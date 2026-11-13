"""
Outlier Stress-Test Benchmark: Raw Least-Squares vs. Batched Spinor RANSAC
==========================================================================
Tests 3D camera localization with 40% corrupted false matches (outliers)
and sensor noise across N = 500 to N = 10,000 landmarks on GPU/XPU.
"""

import time
import torch
import numpy as np
from twistor.geometry import get_best_device
from twistor.renderer import generate_su22_camera_matrix, points_to_complex_hermitian
from twistor.ransac import solve_camera_position_ransac
from examples.instant_robot_snap import (
    project_landmarks_to_uv,
    pixel_uv_to_world_spinor,
    solve_camera_position_closed_form,
)


def sync_device(device):
    d = str(device)
    if d == "xpu":
        torch.xpu.synchronize()
    elif d == "cuda":
        torch.cuda.synchronize()


def run_ransac_benchmark():
    device = get_best_device()
    print(f"=== Batched Spinor RANSAC Stress-Test (40% Outliers + Sensor Noise) [{device}] ===")

    torch.manual_seed(42)
    true_pos_world = torch.tensor([0.35, -0.25, -2.10], dtype=torch.float32, device=device)
    yaw, pitch = 0.18, -0.12

    R_A, _, _ = generate_su22_camera_matrix(yaw=yaw, pitch=pitch, device=device)
    R_A_dag = R_A.conj().mT
    C_world_herm = points_to_complex_hermitian(true_pos_world.unsqueeze(0))[0]
    C_cam_herm = torch.matmul(R_A, torch.matmul(C_world_herm, R_A_dag))
    true_cam_pos = torch.stack([
        C_cam_herm[1, 0].real * np.sqrt(2.0),
        C_cam_herm[1, 0].imag * np.sqrt(2.0),
        C_cam_herm[0, 0].real * np.sqrt(2.0),
    ])

    for n_pts in [500, 2000, 10000]:
        landmarks = torch.empty((n_pts, 3), dtype=torch.float32, device=device)
        landmarks[:, 0].uniform_(-2.0, 2.0)
        landmarks[:, 1].uniform_(-2.0, 2.0)
        landmarks[:, 2].uniform_(0.4, 3.0)

        u_obs, v_obs = project_landmarks_to_uv(landmarks, true_cam_pos, yaw, pitch)

        # Add realistic pixel jitter to all points
        u_obs = u_obs + 0.0015 * torch.randn_like(u_obs)
        v_obs = v_obs + 0.0015 * torch.randn_like(v_obs)

        # Corrupt 40% of observations with wild false-match outliers
        n_outliers = int(0.40 * n_pts)
        outlier_idx = torch.randperm(n_pts, device=device)[:n_outliers]
        u_obs[outlier_idx] = torch.empty(n_outliers, device=device).uniform_(-1.5, 1.5)
        v_obs[outlier_idx] = torch.empty(n_outliers, device=device).uniform_(-1.5, 1.5)

        pi_world = pixel_uv_to_world_spinor(u_obs, v_obs, R_A)

        # Raw least-squares (no RANSAC)
        pos_raw = solve_camera_position_closed_form(landmarks, pi_world)
        err_raw = torch.linalg.norm(pos_raw - true_pos_world).item()

        # Warmup RANSAC
        for _ in range(5):
            _, _ = solve_camera_position_ransac(landmarks, pi_world, num_hypotheses=512)

        sync_device(device)
        t0 = time.perf_counter()
        for _ in range(30):
            pos_ransac, inliers = solve_camera_position_ransac(
                landmarks, pi_world, num_hypotheses=512, inlier_threshold=0.015
            )
        sync_device(device)
        dt_ms = ((time.perf_counter() - t0) / 30.0) * 1000.0
        err_ransac = torch.linalg.norm(pos_ransac - true_pos_world).item()

        print(
            f"N={n_pts:5,d} (40% outliers) | "
            f"Raw LS Err: {err_raw:.3f}m | "
            f"Spinor RANSAC Err: {err_ransac:.5f}m | "
            f"Inliers Found: {inliers.sum().item()}/{n_pts} | "
            f"Time: {dt_ms:.3f} ms"
        )


if __name__ == "__main__":
    run_ransac_benchmark()
