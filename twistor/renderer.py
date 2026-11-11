"""
Twistor Space Renderer Core (Real FP32 Decomposed Kernel)
==========================================================
Decomposes complex 2-spinor algebra into fused real FP32 tensor contractions
to eliminate PyTorch complex64 sub-kernel dispatch overhead on GPUs.
"""

import torch


def points_to_real_hermitian(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Converts 3D spatial points (x, y, z) into real and imaginary parts of
    a 2x2 Hermitian matrix X^{AA'}. Assumes t = 0.
    
    X^{AA'} = 1/sqrt(2) * [[ t+z,   x-iy ],
                           [ x+iy,  t-z  ]]
    """
    N = points.shape[0]
    dev = points.device
    scale = 1.0 / 1.41421356

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Real part of X^{AA'}
    R_X = torch.zeros((N, 2, 2), dtype=torch.float32, device=dev)
    R_X[:, 0, 0] = z * scale
    R_X[:, 0, 1] = x * scale
    R_X[:, 1, 0] = x * scale
    R_X[:, 1, 1] = -z * scale

    # Imaginary part of X^{AA'}
    I_X = torch.zeros((N, 2, 2), dtype=torch.float32, device=dev)
    I_X[:, 0, 1] = -y * scale
    I_X[:, 1, 0] = y * scale

    return R_X, I_X


def generate_real_ray_spinors(width: int, height: int, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generates real and imaginary components of primary ray twistor spinors (omega, pi)."""
    u = torch.linspace(-1, 1, width, device=device)
    v = torch.linspace(-1, 1, height, device=device)
    grid_y, grid_x = torch.meshgrid(v, u, indexing="ij")
    M = width * height

    # Ray directions mapped to stereographic spinor projection
    dirs = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(M, 3)
    dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)

    # Convert direction unit vectors to pi spinor components
    r_pi = torch.zeros((M, 2), dtype=torch.float32, device=device)
    i_pi = torch.zeros((M, 2), dtype=torch.float32, device=device)
    
    r_pi[:, 0] = torch.sqrt(torch.clamp(1.0 + dirs[:, 2], min=1e-7)) * 0.70710678
    r_pi[:, 1] = dirs[:, 0] / (2.0 * torch.clamp(r_pi[:, 0], min=1e-7))
    i_pi[:, 1] = -dirs[:, 1] / (2.0 * torch.clamp(r_pi[:, 0], min=1e-7))

    # Primary omega spinor components
    r_omega = torch.zeros((M, 2), dtype=torch.float32, device=device)
    i_omega = torch.zeros((M, 2), dtype=torch.float32, device=device)

    return r_omega, i_omega, r_pi, i_pi


def render_point_cloud(points: torch.Tensor, width: int = 32, height: int = 32, threshold: float = 0.08) -> torch.Tensor:
    """
    Evaluates Twistor Incidence Residual in pure real FP32 math.
    """
    device = points.device
    N = points.shape[0]

    R_X, I_X = points_to_real_hermitian(points)
    r_omega, i_omega, r_pi, i_pi = generate_real_ray_spinors(width, height, device=device)

    # Expand dims for batch linear contractions: Rays (M, 1, 2), Points (1, N, 2, 2)
    R_X_exp = R_X.unsqueeze(0)  # (1, N, 2, 2)
    I_X_exp = I_X.unsqueeze(0)  # (1, N, 2, 2)

    r_pi_exp = r_pi.unsqueeze(1).unsqueeze(-1)  # (M, 1, 2, 1)
    i_pi_exp = i_pi.unsqueeze(1).unsqueeze(-1)  # (M, 1, 2, 1)

    # Fused real FP32 tensor contractions
    # Re(i * X @ pi) = -I_X @ r_pi - R_X @ i_pi
    pred_r_omega = -torch.matmul(I_X_exp, r_pi_exp) - torch.matmul(R_X_exp, i_pi_exp)
    # Im(i * X @ pi) =  R_X @ r_pi - I_X @ i_pi
    pred_i_omega =  torch.matmul(R_X_exp, r_pi_exp) - torch.matmul(I_X_exp, i_pi_exp)

    # Calculate squared residual norm || omega - i * X @ pi ||^2
    diff_r = r_omega.unsqueeze(1).unsqueeze(-1) - pred_r_omega
    diff_i = i_omega.unsqueeze(1).unsqueeze(-1) - pred_i_omega

    sq_residual = torch.sum(diff_r ** 2 + diff_i ** 2, dim=(-2, -1))  # (M, N)
    hits = sq_residual < (threshold ** 2)

    return hits.any(dim=-1).reshape(height, width).float()