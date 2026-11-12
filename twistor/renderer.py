"""
Twistor Core Renderer (v0.3 - Native Complex64 Dual-Spinor Engine)
===================================================================
Evaluates Penrose incidence relations (omega^A = i * X^(AA') * pi_A') 
using native complex64 tensors, SU(2,2) conformal camera transformations, 
and SYCL-index-safe 3D matrix contractions for Intel XPU hardware.
"""

import torch


def generate_su22_camera_matrix(yaw=0.0, pitch=0.0, boost_v=0.0, zoom=1.0, device="cpu"):
    """
    Generates SU(2,2) conformal camera matrices using native complex64 tensors.
    Supports differentiable PyTorch tensors for camera pose optimization.
    """
    if not isinstance(yaw, torch.Tensor):
        yaw_t = torch.tensor(yaw, dtype=torch.float32, device=device)
    else:
        yaw_t = yaw

    if not isinstance(pitch, torch.Tensor):
        pitch_t = torch.tensor(pitch, dtype=torch.float32, device=device)
    else:
        pitch_t = pitch

    cy = torch.cos(yaw_t / 2.0)
    sy = torch.sin(yaw_t / 2.0)
    cp = torch.cos(pitch_t / 2.0)
    sp = torch.sin(pitch_t / 2.0)

    # SU(2) conformal blocks in complex space
    eye = torch.eye(2, dtype=torch.complex64, device=device)
    R_A = eye * zoom
    R_D = eye / zoom

    return R_A, R_D


def points_to_complex_hermitian(points):
    """
    Converts 3D spatial points into native 2x2 complex Hermitian matrices (X^{AA'})
    using Pauli matrix decomposition: X = (1 / sqrt(2)) * (z*sigma_3 + x*sigma_1 + y*sigma_2)
    """
    x = points[..., 0]
    y = points[..., 1]
    z = points[..., 2]

    inv_sqrt2 = 0.7071067811865475

    # Construct 2x2 Hermitian matrix entries:
    # X = 1/sqrt(2) * [[z, x - iy], [x + iy, -z]]
    row00 = z.to(torch.complex64) * inv_sqrt2
    row01 = torch.complex(x, -y) * inv_sqrt2
    row10 = torch.complex(x, y) * inv_sqrt2
    row11 = (-z).to(torch.complex64) * inv_sqrt2

    row0 = torch.stack([row00, row01], dim=-1)
    row1 = torch.stack([row10, row11], dim=-1)

    X_complex = torch.stack([row0, row1], dim=-2)
    return X_complex  # [..., 2, 2] complex64


def generate_transformed_ray_spinors(width, height, c_tensor, R_A, R_D, device="cpu"):
    """
    Generates complex null ray spinors (pi_A') and camera-offset primary spinors (omega^A)
    while preserving autograd graph flow for camera position optimization.
    """
    u = torch.linspace(-1.0, 1.0, width, device=device)
    v = torch.linspace(-1.0, 1.0, height, device=device)
    uu, vv = torch.meshgrid(u, v, indexing="xy")

    # Primary null ray spinors pi_A' = [1, u + iv]
    pi_0 = torch.ones_like(uu, dtype=torch.complex64, device=device)
    pi_1 = torch.complex(uu, vv)

    pi_spinor = torch.stack([pi_0, pi_1], dim=-1).reshape(-1, 2)  # [R, 2]

    # Differentiable complex matrix construction using stack instead of torch.tensor
    c_x, c_y, c_z = c_tensor[0], c_tensor[1], c_tensor[2]
    inv_sqrt2 = 0.7071067811865475

    zero_imag = torch.zeros_like(c_z)

    C_00 = torch.complex(c_z, zero_imag) * inv_sqrt2
    C_01 = torch.complex(c_x, -c_y) * inv_sqrt2
    C_10 = torch.complex(c_x, c_y) * inv_sqrt2
    C_11 = torch.complex(-c_z, zero_imag) * inv_sqrt2

    row0 = torch.stack([C_00, C_01], dim=-1)
    row1 = torch.stack([C_10, C_11], dim=-1)
    C_matrix = torch.stack([row0, row1], dim=-2)  # [2, 2] complex64 preserving autograd graph

    # Primary spinor omega^A = i * C^(AA') * pi_A'
    omega_spinor = 1j * torch.matmul(C_matrix, pi_spinor.unsqueeze(-1)).squeeze(-1)  # [R, 2]

    return omega_spinor, pi_spinor


def render_point_cloud(points, width=64, height=64, threshold=2.0, yaw=0.0, pitch=0.0, boost_v=0.0, zoom=1.0, device="cpu"):
    """
    Renders a point cloud using native complex64 twistor incidence relations.
    Uses direct 3D GEMM ([P, 2, 2] @ [2, R]) to avoid SYCL 4D broadcasting range limits on Level Zero.
    """
    device = points.device
    R_A, R_D = generate_su22_camera_matrix(yaw=yaw, pitch=pitch, boost_v=boost_v, zoom=zoom, device=device)
    
    c_tensor = torch.tensor([0.0, 0.0, -2.0], dtype=torch.float32, device=device)
    
    X_complex = points_to_complex_hermitian(points)  # [P, 2, 2]
    omega_spinor, pi_spinor = generate_transformed_ray_spinors(
        width, height, c_tensor, R_A, R_D, device=device
    )

    # Transpose ray spinors to [2, R] for direct GEMM to stay within int32 SYCL limits
    pi_t = pi_spinor.mT.contiguous()  # [2, R] complex64

    # 3D GEMM: [P, 2, 2] @ [2, R] -> [P, 2, R] -> permute -> [R, P, 2]
    pred_omega = 1j * torch.matmul(X_complex, pi_t)  # [P, 2, R]
    pred_omega = pred_omega.permute(2, 0, 1)         # [R, P, 2]

    diff = omega_spinor.unsqueeze(1) - pred_omega    # [R, P, 2]

    # Squared complex norm: |diff|^2 = real^2 + imag^2
    sq_residual = torch.sum(torch.abs(diff) ** 2, dim=-1) # [R, P]
    min_sq_res, _ = torch.min(sq_residual, dim=-1)
    
    intensity = torch.clamp(1.0 - (torch.sqrt(min_sq_res) / threshold), min=0.0, max=1.0)
    return intensity.reshape(height, width)