"""
twistor-render-core
===================
Differentiable 3D rendering engine built on Penrose Twistor Theory (CP^3)
and native complex64 2-spinor algebra.
"""

from .renderer import (
    render_point_cloud,
    generate_su22_camera_matrix,
    points_to_complex_hermitian,
    generate_transformed_ray_spinors,
)
from .io import load_point_cloud, load_ply_points, save_ply_points

__version__ = "0.6.0"
__all__ = [
    "render_point_cloud",
    "generate_su22_camera_matrix",
    "points_to_complex_hermitian",
    "generate_transformed_ray_spinors",
    "load_point_cloud",
    "load_ply_points",
    "save_ply_points",
]