"""
Twistor Geometry Engine - Phase 1 Core Mathematics
===================================================
Handles 3D/4D vector conversions to 2x2 complex Hermitian matrices (X^{AA'}),
construction of 4-component complex twistors Z^alpha = (omega^A, pi_{A'}),
and SL(2, C) complex Lorentz/spinor transformations.
"""

from typing import Tuple, Optional
import torch


def get_best_device() -> torch.device:
    """
    Auto-detects the optimal available hardware acceleration backend.
    Supports NVIDIA CUDA, Intel XPU, Apple MPS, and fallback CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def points_to_hermitian(points: torch.Tensor) -> torch.Tensor:
    """
    Converts 3D spatial points (x, y, z) or 4D spacetime points (t, x, y, z)
    into 2x2 complex Hermitian matrices X^{AA'}.

    For 3D points, light-cone time t = sqrt(x^2 + y^2 + z^2) is assigned so that
    null geodesic incidence holds along light paths from the origin.
    """
    num_points = points.shape[0]
    device = points.device

    if points.shape[1] == 3:
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        t = torch.sqrt(x**2 + y**2 + z**2)
    elif points.shape[1] == 4:
        t, x, y, z = points[:, 0], points[:, 1], points[:, 2], points[:, 3]
    else:
        raise ValueError(f"Expected input shape (N, 3) or (N, 4), got {points.shape}")

    hermitian = torch.zeros(num_points, 2, 2, dtype=torch.complex64, device=device)
    hermitian[:, 0, 0] = torch.complex(t + z, torch.zeros_like(z))
    hermitian[:, 0, 1] = torch.complex(x, -y)
    hermitian[:, 1, 0] = torch.complex(x, y)
    hermitian[:, 1, 1] = torch.complex(t - z, torch.zeros_like(z))

    return hermitian


def hermitian_to_points(hermitian: torch.Tensor) -> torch.Tensor:
    """
    Inverse map: Converts 2x2 Hermitian matrices X^{AA'} back to 4D spacetime points (t, x, y, z).
    """
    t_plus_z = hermitian[:, 0, 0].real
    t_minus_z = hermitian[:, 1, 1].real
    x = hermitian[:, 1, 0].real
    y = hermitian[:, 1, 0].imag

    t = 0.5 * (t_plus_z + t_minus_z)
    z = 0.5 * (t_plus_z - t_minus_z)

    return torch.stack([t, x, y, z], dim=-1)


def create_twistor(
    omega: torch.Tensor, 
    pi: torch.Tensor
) -> torch.Tensor:
    """
    Constructs a batch of 4-component complex twistors Z^alpha = (omega^A, pi_{A'}).
    """
    if omega.dim() == 2:
        omega = omega.unsqueeze(-1)
    if pi.dim() == 2:
        pi = pi.unsqueeze(-1)

    return torch.cat([omega, pi], dim=1)


def apply_sl2c_transformation(
    spinor: torch.Tensor, 
    transform_matrix: torch.Tensor
) -> torch.Tensor:
    """
    Applies an SL(2, C) transformation to a batch of 2-component complex spinors.
    """
    return torch.matmul(transform_matrix, spinor)