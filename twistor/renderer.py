"""
Twistor Renderer Pipeline - Phase 3 Camera & Image Buffer Generation
===================================================================
Maps camera pixels to 4-component complex twistors Z^alpha = (omega^A, pi_{A'})
and renders 2D image buffers from 3D point clouds or scene geometry.
"""

from typing import Tuple
import torch
import twistor.geometry as g
import twistor.incidence as inc


def generate_camera_rays(
    width: int,
    height: int,
    fov: float = 60.0,
    device: torch.device = torch.device("cpu")
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates a grid of primary camera rays mapped to twistor components (omega^A, pi_{A'}).
    Rays originate from the camera origin (0, 0, 0) pointing towards the image plane.
    """
    aspect_ratio = width / height
    fov_rad = torch.tensor(fov * (3.141592653589793 / 180.0))
    scale = torch.tan(fov_rad / 2.0)

    y_coords = torch.linspace(1.0, -1.0, height, device=device)
    x_coords = torch.linspace(-1.0 * aspect_ratio, 1.0 * aspect_ratio, width, device=device)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")

    dx = grid_x * scale
    dy = grid_y * scale
    dz = torch.ones_like(dx)

    norm = torch.sqrt(dx**2 + dy**2 + dz**2)
    dir_x, dir_y, dir_z = dx / norm, dy / norm, dz / norm

    denom = torch.sqrt(2.0 * (1.0 + dir_z))
    pi_0 = torch.complex(-dir_x / denom, dir_y / denom)
    pi_1 = torch.complex((1.0 + dir_z) / denom, torch.zeros_like(dir_z))

    pi_grid = torch.stack([pi_0, pi_1], dim=-1).unsqueeze(-1)
    omega_grid = torch.zeros(height, width, 2, 1, dtype=torch.complex64, device=device)

    return omega_grid, pi_grid


def render_point_cloud(
    points_3d: torch.Tensor,
    width: int = 64,
    height: int = 64,
    threshold: float = 0.15
) -> torch.Tensor:
    """
    Renders a 3D point cloud onto a 2D binary image buffer using vectorized
    twistor incidence checks across all rays and points simultaneously.
    """
    device = points_3d.device
    hermitian_pts = g.points_to_hermitian(points_3d)  # Shape (N, 2, 2)

    omega_grid, pi_grid = generate_camera_rays(width, height, device=device)

    omega_flat = omega_grid.view(-1, 2, 1)  # Shape (K, 2, 1)
    pi_flat = pi_grid.view(-1, 2, 1)        # Shape (K, 2, 1)

    i_unit = torch.complex(torch.tensor(0.0, device=device), torch.tensor(1.0, device=device))
    h_pts_exp = hermitian_pts.unsqueeze(0)   # Shape (1, N, 2, 2)
    pi_exp = pi_flat.unsqueeze(1)            # Shape (K, 1, 2, 1)

    omega_pred = i_unit * torch.matmul(h_pts_exp, pi_exp)  # Shape (K, N, 2, 1)
    omega_exp = omega_flat.unsqueeze(1)                   # Shape (K, 1, 2, 1)

    diff = omega_exp - omega_pred
    residuals = torch.norm(diff, dim=(2, 3)).real  # Shape (K, N)

    hits_per_ray = torch.any(residuals < threshold, dim=1)  # Shape (K,)
    image_buffer = hits_per_ray.view(height, width).float()

    return image_buffer