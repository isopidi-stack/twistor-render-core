"""
Twistor Core Renderer (v0.7 - Differentiable 6-DoF + SL(2,C) Lorentz Boost Engine)
=================================================================================
Evaluates Penrose eigenspinor incidence relations in CP^3 with optional
differentiable spinor depth relief and relativistic Lorentz aberration (boost_v).
"""

import torch


def generate_su22_camera_matrix(yaw=0.0, pitch=0.0, boost_v=0.0, zoom=1.0, device="cpu"):
    """
    Generates differentiable SL(2, C) spinor rotation (R_A) and Lorentz boost (B_boost) matrices.
    boost_v: relativistic velocity beta = v/c in (-0.995, 0.995).
    """
    yaw_t = yaw if isinstance(yaw, torch.Tensor) else torch.tensor(yaw, dtype=torch.float32, device=device)
    pitch_t = pitch if isinstance(pitch, torch.Tensor) else torch.tensor(pitch, dtype=torch.float32, device=device)
    zoom_t = zoom if isinstance(zoom, torch.Tensor) else torch.tensor(zoom, dtype=torch.float32, device=device)
    beta_t = boost_v if isinstance(boost_v, torch.Tensor) else torch.tensor(boost_v, dtype=torch.float32, device=device)

    beta_safe = torch.clamp(beta_t, -0.995, 0.995)
    rapidity = torch.atanh(beta_safe)
    exp_neg = torch.exp(-0.5 * rapidity)
    exp_pos = torch.exp(0.5 * rapidity)

    cy = torch.cos(yaw_t / 2.0)
    sy = torch.sin(yaw_t / 2.0)
    cp = torch.cos(pitch_t / 2.0)
    sp = torch.sin(pitch_t / 2.0)

    zero = torch.zeros_like(cy)

    # Differentiable SU(2) Yaw matrix
    Uy_00 = torch.complex(cy, zero)
    Uy_01 = torch.complex(-sy, zero)
    Uy_10 = torch.complex(sy, zero)
    Uy_11 = torch.complex(cy, zero)
    U_yaw = torch.stack([
        torch.stack([Uy_00, Uy_01], dim=-1),
        torch.stack([Uy_10, Uy_11], dim=-1)
    ], dim=-2)

    # Differentiable SU(2) Pitch matrix
    Up_00 = torch.complex(cp, zero)
    Up_01 = torch.complex(zero, -sp)
    Up_10 = torch.complex(zero, -sp)
    Up_11 = torch.complex(cp, zero)
    U_pitch = torch.stack([
        torch.stack([Up_00, Up_01], dim=-1),
        torch.stack([Up_10, Up_11], dim=-1)
    ], dim=-2)

    U_rot = torch.matmul(U_pitch, U_yaw)

    zoom_c = torch.complex(torch.sqrt(zoom_t), zero)
    R_A = U_rot * zoom_c

    # SL(2, C) Lorentz boost matrix B_boost (det = 1)
    B_boost = torch.stack([
        torch.stack([torch.complex(exp_neg, zero), torch.complex(zero, zero)], dim=-1),
        torch.stack([torch.complex(zero, zero), torch.complex(exp_pos, zero)], dim=-1)
    ], dim=-2)

    return R_A, B_boost


def points_to_complex_hermitian(points):
    """
    Converts 3D spatial points into native 2x2 complex Hermitian matrices (X^{AA'}).
    """
    x = points[..., 0]
    y = points[..., 1]
    z = points[..., 2]

    inv_sqrt2 = 0.7071067811865475

    row00 = z.to(torch.complex64) * inv_sqrt2
    row01 = torch.complex(x, -y) * inv_sqrt2
    row10 = torch.complex(x, y) * inv_sqrt2
    row11 = (-z).to(torch.complex64) * inv_sqrt2

    row0 = torch.stack([row00, row01], dim=-1)
    row1 = torch.stack([row10, row11], dim=-1)

    return torch.stack([row0, row1], dim=-2)


def generate_transformed_ray_spinors(width, height, c_tensor, R_A, B_boost, device="cpu"):
    """
    Generates normalized projective null ray spinors (pi_A') under SL(2, C) Lorentz boost
    and camera-origin spinors (omega^A).
    """
    u = torch.linspace(-0.65, 0.65, width, device=device)
    v = torch.linspace(-0.65, 0.65, height, device=device)
    uu, vv = torch.meshgrid(u, v, indexing="xy")

    # Projective null ray spinors pi_A' = [1, u + iv]
    pi_0 = torch.ones_like(uu, dtype=torch.complex64, device=device)
    pi_1 = torch.complex(uu, vv)
    pi_raw = torch.stack([pi_0, pi_1], dim=-1).reshape(-1, 2)  # [R, 2]

    # Apply SL(2, C) Lorentz boost before unit spinor normalization
    pi_boosted = torch.matmul(pi_raw, B_boost.mT)
    pi_norm = torch.linalg.norm(pi_boosted, dim=-1, keepdim=True)
    pi_spinor = pi_boosted / pi_norm

    # Differentiable complex camera translation matrix C^(AA')
    c_x, c_y, c_z = c_tensor[0], c_tensor[1], c_tensor[2]
    inv_sqrt2 = 0.7071067811865475
    zero_imag = torch.zeros_like(c_z)

    C_00 = torch.complex(c_z, zero_imag) * inv_sqrt2
    C_01 = torch.complex(c_x, -c_y) * inv_sqrt2
    C_10 = torch.complex(c_x, c_y) * inv_sqrt2
    C_11 = torch.complex(-c_z, zero_imag) * inv_sqrt2

    C_matrix = torch.stack([
        torch.stack([C_00, C_01], dim=-1),
        torch.stack([C_10, C_11], dim=-1)
    ], dim=-2)

    omega_spinor = 1j * torch.matmul(C_matrix, pi_spinor.mT).mT  # [R, 2]
    return omega_spinor, pi_spinor


def render_point_cloud(
    points,
    width=64,
    height=64,
    threshold=0.05,
    yaw=0.0,
    pitch=0.0,
    boost_v=0.0,
    zoom=1.0,
    cam_pos=None,
    tau=0.0005,
    shading=False,
    depth_contrast=2.2,
    device="cpu",
):
    """
    Renders a point cloud using native complex64 twistor eigenspinor incidence relations.
    Supports SL(2, C) relativistic Lorentz aberration via boost_v = v/c.
    """
    device = points.device
    R_A, B_boost = generate_su22_camera_matrix(yaw=yaw, pitch=pitch, boost_v=boost_v, zoom=zoom, device=device)

    if cam_pos is None:
        c_tensor = torch.tensor([0.0, 0.0, -2.0], dtype=torch.float32, device=device)
    else:
        c_tensor = cam_pos

    X_world = points_to_complex_hermitian(points)  # [P, 2, 2]

    # Apply SU(2) spinor conjugation: X_cam = R_A @ X_world @ R_A^dagger
    R_A_dag = R_A.conj().mT
    X_complex = torch.matmul(R_A, torch.matmul(X_world, R_A_dag))  # [P, 2, 2]

    omega_spinor, pi_spinor = generate_transformed_ray_spinors(
        width, height, c_tensor, R_A, B_boost, device=device
    )

    pi_t = pi_spinor.mT.contiguous()  # [2, R] complex64

    # 3D GEMM: [P, 2, 2] @ [2, R] -> [P, 2, R] -> permute -> [R, P, 2]
    pred_omega = 1j * torch.matmul(X_complex, pi_t)
    pred_omega = pred_omega.permute(2, 0, 1)

    # Twistor incidence difference: diff = omega^A - i X^{AA'} pi_{A'}
    diff = omega_spinor.unsqueeze(1) - pred_omega  # [R, P, 2]

    # Eigenspinor perpendicular residual: ||diff||^2 - |<pi | diff>|^2
    total_sq = torch.sum(torch.abs(diff) ** 2, dim=-1)  # [R, P]
    longitudinal = torch.sum(pi_spinor.conj().unsqueeze(1) * diff, dim=-1)  # [R, P] complex64
    perp_sq_residual = torch.relu(total_sq - (torch.abs(longitudinal) ** 2))  # [R, P]

    # Stabilized LogSumExp softmin for silhouette occupancy
    min_val, _ = torch.min(perp_sq_residual, dim=-1, keepdim=True)
    softmin_sq_res = min_val.squeeze(-1) - tau * torch.logsumexp(-(perp_sq_residual - min_val) / tau, dim=-1)
    occupancy = torch.exp(-torch.relu(softmin_sq_res) / (threshold ** 2))

    if not shading:
        return occupancy.reshape(height, width)

    # Differentiable Spinor Depth Relief (static tensor shapes — zero XPU boolean indexing):
    signed_depth = -longitudinal.imag  # [R, P]
    depth_bias = 0.0015 * signed_depth
    z_score = -(perp_sq_residual + depth_bias) / tau
    weights = torch.softmax(z_score, dim=-1)  # [R, P]
    expected_depth = torch.sum(weights * signed_depth, dim=-1)  # [R]

    fg_mask = occupancy > 0.1
    z_near = torch.where(fg_mask, expected_depth, torch.tensor(1e6, device=device)).min().detach()
    z_far = torch.where(fg_mask, expected_depth, torch.tensor(-1e6, device=device)).max().detach()
    z_span = torch.clamp(z_far - z_near, min=1e-4)

    rel_depth = torch.clamp((expected_depth - z_near) / z_span, 0.0, 1.0)
    shade = 0.28 + 0.72 * torch.exp(-depth_contrast * rel_depth)

    intensity = occupancy * shade
    return intensity.reshape(height, width)
