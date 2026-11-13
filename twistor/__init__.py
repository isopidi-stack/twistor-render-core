"""
twistor-vision (twistor)
========================
Sub-millisecond 3D robot localization, batched GPU Spinor RANSAC, and
differentiable SU(2,2) conformal optics on Intel XPU, NVIDIA CUDA, Apple MPS, and CPU.
"""

__version__ = "0.8.0"

import torch
from twistor.geometry import get_best_device
from twistor.renderer import (
    generate_su22_camera_matrix,
    points_to_complex_hermitian,
    render_twistor_splats,
)
from twistor.localization import (
    pixel_uv_to_world_spinor,
    solve_camera_position_closed_form,
)
from twistor.ransac import solve_camera_position_ransac


def localize(
    landmarks_xyz: torch.Tensor,
    pixels_uv: torch.Tensor,
    yaw: float = 0.0,
    pitch: float = 0.0,
    R_A: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    One-shot closed-form 3-DoF camera position from 3D landmarks and 2D stereographic
    (u, v) rays given known IMU orientation (yaw, pitch or SU(2) matrix R_A).
    """
    device = landmarks_xyz.device
    if R_A is None:
        R_A, _, _ = generate_su22_camera_matrix(yaw=yaw, pitch=pitch, device=device)
    pi_world = pixel_uv_to_world_spinor(pixels_uv[:, 0], pixels_uv[:, 1], R_A)
    return solve_camera_position_closed_form(landmarks_xyz, pi_world)


def localize_ransac(
    landmarks_xyz: torch.Tensor,
    pixels_uv: torch.Tensor,
    yaw: float = 0.0,
    pitch: float = 0.0,
    R_A: torch.Tensor | None = None,
    num_hypotheses: int = 512,
    inlier_threshold: float = 0.008,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Outlier-robust 3-DoF camera localization via batched GPU Spinor RANSAC.
    Returns (position_xyz [3], inlier_mask [N]).
    """
    device = landmarks_xyz.device
    if R_A is None:
        R_A, _, _ = generate_su22_camera_matrix(yaw=yaw, pitch=pitch, device=device)
    pi_world = pixel_uv_to_world_spinor(pixels_uv[:, 0], pixels_uv[:, 1], R_A)
    return solve_camera_position_ransac(
        landmarks_xyz,
        pi_world,
        num_hypotheses=num_hypotheses,
        inlier_threshold=inlier_threshold,
    )


def calibrate_su22(
    landmarks_xyz: torch.Tensor,
    target_uv: torch.Tensor,
    cam_pos: torch.Tensor,
    steps: int = 120,
    lr: float = 0.03,
) -> dict[str, torch.Tensor]:
    """
    Differentiable single-image SU(2,2) conformal self-calibration.
    Jointly recovers 3D orientation (yaw, pitch), optical zoom dilation, and
    wide-angle/fisheye conformal distortion (bx, by) from 2D-3D correspondences.
    """
    device = landmarks_xyz.device
    yaw = torch.zeros(1, device=device, requires_grad=True)
    pitch = torch.zeros(1, device=device, requires_grad=True)
    zoom = torch.zeros(1, device=device, requires_grad=True)
    b_conf = torch.zeros(2, device=device, requires_grad=True)

    opt = torch.optim.Adam([yaw, pitch, zoom, b_conf], lr=lr)
    X_world = points_to_complex_hermitian(landmarks_xyz)
    C_world = points_to_complex_hermitian(cam_pos.unsqueeze(0))[0]

    for _ in range(steps):
        opt.zero_grad()
        R_A, _, _ = generate_su22_camera_matrix(yaw=yaw, pitch=pitch, device=device)
        R_dag = R_A.conj().mT
        X_rel = torch.matmul(R_A, torch.matmul(X_world - C_world, R_dag))

        # Extract camera-frame coordinates & apply conformal dilation + special conformal boost
        x_c = X_rel[:, 1, 0].real * 1.41421356
        y_c = X_rel[:, 1, 0].imag * 1.41421356
        z_c = torch.clamp(X_rel[:, 0, 0].real * 1.41421356, min=1e-4)

        scale = torch.exp(zoom)
        u_pin = scale * (x_c / z_c)
        v_pin = scale * (y_c / z_c)

        r2 = u_pin * u_pin + v_pin * v_pin
        bx, by = b_conf[0], b_conf[1]
        denom = torch.clamp(1.0 - 2.0 * (bx * u_pin + by * v_pin) + (bx * bx + by * by) * r2, min=1e-4)
        u_pred = (u_pin - bx * r2) / denom
        v_pred = (v_pin - by * r2) / denom

        loss = torch.mean((u_pred - target_uv[:, 0]) ** 2 + (v_pred - target_uv[:, 1]) ** 2)
        loss.backward()
        opt.step()

    return {
        "yaw": yaw.detach(),
        "pitch": pitch.detach(),
        "zoom": zoom.detach(),
        "conformal_b": b_conf.detach(),
        "final_mse": loss.detach(),
    }


__all__ = [
    "__version__",
    "get_best_device",
    "localize",
    "localize_ransac",
    "calibrate_su22",
    "pixel_uv_to_world_spinor",
    "solve_camera_position_closed_form",
    "solve_camera_position_ransac",
    "generate_su22_camera_matrix",
    "points_to_complex_hermitian",
    "render_twistor_splats",
]
