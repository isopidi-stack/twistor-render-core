"""
Twistor Space Renderer Core (Real FP32 Decomposed Kernel + Closest Approach Depth)
=================================================================================
"""

import math
import torch


def points_to_real_hermitian(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Converts 3D spatial points (x, y, z) into real and imaginary parts of X^{AA'}."""
    N = points.shape[0]
    dev = points.device
    scale = 1.0 / 1.41421356

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    R_X = torch.zeros((N, 2, 2), dtype=torch.float32, device=dev)
    R_X[:, 0, 0] = z * scale
    R_X[:, 0, 1] = x * scale
    R_X[:, 1, 0] = x * scale
    R_X[:, 1, 1] = -z * scale

    I_X = torch.zeros((N, 2, 2), dtype=torch.float32, device=dev)
    I_X[:, 0, 1] = -y * scale
    I_X[:, 1, 0] = y * scale

    return R_X, I_X


def pos_to_real_hermitian(pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Converts camera position (cx, cy, cz) into real and imaginary Hermitian components."""
    dev = pos.device
    scale = 1.0 / 1.41421356
    cx, cy, cz = pos[0], pos[1], pos[2]

    R_C = torch.tensor([[cz * scale, cx * scale], [cx * scale, -cz * scale]], dtype=torch.float32, device=dev)
    I_C = torch.tensor([[0.0, -cy * scale], [cy * scale, 0.0]], dtype=torch.float32, device=dev)
    return R_C, I_C


def generate_su22_camera_matrix(
    yaw: float = 0.0,
    pitch: float = 0.0,
    boost_v: float = 0.0,
    zoom: float = 1.0,
    device: str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Constructs real & imaginary parts of SU(2,2) blocks A and D."""
    cy = math.cos(yaw / 2.0)
    sy = math.sin(yaw / 2.0)
    cp = math.cos(pitch / 2.0)
    sp = math.sin(pitch / 2.0)

    R_U = torch.tensor([[cy * cp, -sy * sp], [sy * sp, cy * cp]], dtype=torch.float32, device=device)
    I_U = torch.tensor([[-cy * sp, -sy * cp], [-sy * cp, cy * sp]], dtype=torch.float32, device=device)

    phi = math.atanh(min(max(boost_v, -0.99), 0.99))
    ch = math.cosh(phi / 2.0)
    sh = math.sinh(phi / 2.0)

    L = torch.tensor([[ch - sh, 0.0], [0.0, ch + sh]], dtype=torch.float32, device=device)
    L_inv = torch.tensor([[ch + sh, 0.0], [0.0, ch - sh]], dtype=torch.float32, device=device)

    s_a = math.sqrt(zoom)
    s_d = 1.0 / s_a

    R_A = s_a * (L @ R_U)
    I_A = s_a * (L @ I_U)
    R_D = s_d * (L_inv @ R_U)
    I_D = s_d * (L_inv @ I_U)

    return R_A, I_A, R_D, I_D


def generate_transformed_ray_spinors(
    width: int,
    height: int,
    cam_pos: torch.Tensor,
    R_A: torch.Tensor,
    I_A: torch.Tensor,
    R_D: torch.Tensor,
    I_D: torch.Tensor,
    device: str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generates ray spinors with camera offset c and applies SU(2,2) transformation."""
    u = torch.linspace(-1, 1, width, device=device)
    v = torch.linspace(-1, 1, height, device=device)
    grid_y, grid_x = torch.meshgrid(v, u, indexing="ij")
    M = width * height

    dirs = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(M, 3)
    dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)

    r_pi = torch.zeros((M, 2, 1), dtype=torch.float32, device=device)
    i_pi = torch.zeros((M, 2, 1), dtype=torch.float32, device=device)
    
    r_pi[:, 0, 0] = torch.sqrt(torch.clamp(1.0 + dirs[:, 2], min=1e-7)) * 0.70710678
    r_pi[:, 1, 0] = dirs[:, 0] / (2.0 * torch.clamp(r_pi[:, 0, 0], min=1e-7))
    i_pi[:, 1, 0] = -dirs[:, 1] / (2.0 * torch.clamp(r_pi[:, 0, 0], min=1e-7))

    R_C, I_C = pos_to_real_hermitian(cam_pos)
    r_omega = -torch.matmul(I_C, r_pi) - torch.matmul(R_C, i_pi)
    i_omega =  torch.matmul(R_C, r_pi) - torch.matmul(I_C, i_pi)

    r_omega_trans = torch.matmul(R_A, r_omega) - torch.matmul(I_A, i_omega)
    i_omega_trans = torch.matmul(R_A, i_omega) + torch.matmul(I_A, r_omega)

    r_pi_trans = torch.matmul(R_D, r_pi) - torch.matmul(I_D, i_pi)
    i_pi_trans = torch.matmul(R_D, i_pi) + torch.matmul(I_D, r_pi)

    return r_omega_trans, i_omega_trans, r_pi_trans, i_pi_trans


def render_point_cloud(
    points: torch.Tensor,
    width: int = 32,
    height: int = 32,
    threshold: float = 2.0,
    cam_pos: list[float] = [0.0, 0.0, -2.0],
    yaw: float = 0.0,
    pitch: float = 0.0,
    boost_v: float = 0.0,
    zoom: float = 1.0,
    debug: bool = False
) -> torch.Tensor:
    """Evaluates Twistor Incidence minimum distance per ray to render point silhouettes."""
    device = points.device
    c_tensor = torch.tensor(cam_pos, dtype=torch.float32, device=device)

    R_X, I_X = points_to_real_hermitian(points)
    R_A, I_A, R_D, I_D = generate_su22_camera_matrix(yaw, pitch, boost_v, zoom, device=device)
    
    r_omega, i_omega, r_pi, i_pi = generate_transformed_ray_spinors(
        width, height, c_tensor, R_A, I_A, R_D, I_D, device=device
    )

    R_X_exp = R_X.unsqueeze(0)  # (1, N, 2, 2)
    I_X_exp = I_X.unsqueeze(0)  # (1, N, 2, 2)

    r_pi_exp = r_pi.unsqueeze(1)  # (M, 1, 2, 1)
    i_pi_exp = i_pi.unsqueeze(1)  # (M, 1, 2, 1)

    pred_r_omega = -torch.matmul(I_X_exp, r_pi_exp) - torch.matmul(R_X_exp, i_pi_exp)
    pred_i_omega =  torch.matmul(R_X_exp, r_pi_exp) - torch.matmul(I_X_exp, i_pi_exp)

    diff_r = r_omega.unsqueeze(1) - pred_r_omega
    diff_i = i_omega.unsqueeze(1) - pred_i_omega

    sq_residual = torch.sum(diff_r ** 2 + diff_i ** 2, dim=(-2, -1))  # (M, N)
    
    # Find minimum residual per ray across all points in the cloud
    min_sq_res, _ = torch.min(sq_residual, dim=-1)  # (M,)

    if debug:
        print(f"[DEBUG] min_sq_res min: {min_sq_res.min().item():.4f}, max: {min_sq_res.max().item():.4f}")

    # Map closest distance to intensity (normalized inverted residual)
    intensity = torch.clamp(1.0 - (torch.sqrt(min_sq_res) / threshold), min=0.0, max=1.0)

    return intensity.reshape(height, width)