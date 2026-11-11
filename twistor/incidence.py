"""
Twistor Incidence Engine - Phase 2 Collision Kernels
===================================================
Evaluates Penrose's incidence relation omega^A = i * X^{AA'} * pi_{A'}
for batched point clouds, line intersections, and complex spinor collision detection.
"""

from typing import Tuple
import torch


def compute_incidence_residual(
    hermitian_points: torch.Tensor,
    omega: torch.Tensor,
    pi: torch.Tensor
) -> torch.Tensor:
    """
    Computes the Penrose incidence relation residual for a batch of points X^{AA'}
    against a light ray specified by twistor components (omega^A, pi_{A'}).

    Penrose Incidence Relation:
        omega^A = i * X^{AA'} * pi_{A'}

    Args:
        hermitian_points: Tensor of shape (N, 2, 2), complex64 (2x2 Hermitian points)
        omega: Primary position spinor of shape (2, 1) or (N, 2, 1), complex64
        pi: Momentum/direction spinor of shape (2, 1) or (N, 2, 1), complex64

    Returns:
        Tensor of shape (N,) with float32 residual values (0.0 means exact incidence).
    """
    device = hermitian_points.device
    i_unit = torch.complex(torch.tensor(0.0, device=device), torch.tensor(1.0, device=device))

    if omega.dim() == 2:
        omega = omega.unsqueeze(0)  # Shape (1, 2, 1)
    if pi.dim() == 2:
        pi = pi.unsqueeze(0)        # Shape (1, 2, 1)

    omega_pred = i_unit * torch.matmul(hermitian_points, pi)
    diff = omega - omega_pred
    residual = torch.norm(diff, dim=(1, 2)).real

    return residual


def check_point_cloud_incidence(
    hermitian_points: torch.Tensor,
    omega: torch.Tensor,
    pi: torch.Tensor,
    threshold: float = 0.01
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Determines which points in a point cloud lie on or near the given light ray.

    Args:
        hermitian_points: Tensor of shape (N, 2, 2), complex64
        omega: Position spinor of the ray (2, 1), complex64
        pi: Direction spinor of the ray (2, 1), complex64
        threshold: Maximum incidence residual tolerance

    Returns:
        Tuple of (hit_mask [Tensor of bool], residuals [Tensor of float32])
    """
    residuals = compute_incidence_residual(hermitian_points, omega, pi)
    hits = residuals < threshold
    return hits, residuals


def ray_triangle_incidence_check(
    triangle_vertices_hermitian: torch.Tensor,
    omega: torch.Tensor,
    pi: torch.Tensor,
    threshold: float = 0.01
) -> torch.Tensor:
    """
    Evaluates whether a ray intersects a batch of triangles represented by their
    three vertex Hermitian matrices.

    Args:
        triangle_vertices_hermitian: Tensor of shape (M, 3, 2, 2), complex64
        omega: Position spinor of the ray (2, 1), complex64
        pi: Direction spinor of the ray (2, 1), complex64
        threshold: Intersection residual threshold

    Returns:
        Tensor of shape (M,) boolean hits for triangle intersections.
    """
    v0 = triangle_vertices_hermitian[:, 0]
    v1 = triangle_vertices_hermitian[:, 1]
    v2 = triangle_vertices_hermitian[:, 2]

    res0 = compute_incidence_residual(v0, omega, pi)
    res1 = compute_incidence_residual(v1, omega, pi)
    res2 = compute_incidence_residual(v2, omega, pi)

    mean_res = (res0 + res1 + res2) / 3.0
    return mean_res < threshold