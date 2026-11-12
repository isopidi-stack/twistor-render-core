"""
Scaling Benchmark: Twistor Spinor Localization vs. Classical Ray Triangulation
==============================================================================
Tests throughput, latency, and numerical agreement from N = 6 up to N = 50,000
3D landmarks on GPU/XPU/CPU.
"""

import time
import torch
import numpy as np
from twistor.geometry import get_best_device
from twistor.renderer import generate_su22_camera_matrix, points_to_complex_hermitian
from examples.instant_robot_snap import (
    project_landmarks_to_uv,
    pixel_uv_to_world_spinor,
    solve_camera_position_closed_form,
)
from examples.verify_classical_vs_twistor import solve_classical_ray_triangulation


def sync_device(device):
    d = str(device)
    if d == "xpu":
        torch.xpu.synchronize()
    elif d == "cuda":
        torch.cuda.synchronize()


def run_scaling_benchmark():
    device = get_best_device()
    print(f"=== Scaling Benchmark: Twistor vs. Classical ({device}) ===")
    print(f"{'Landmarks (N)':>14} | {'Twistor (ms)':>13} | {'Classical (ms)':>15} | {'Speedup':>9} | {'Max Diff':>11}")
    print("-" * 74)

    torch.manual_seed(123)
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

    for n_pts in [6, 100, 1000, 10000, 50000, 100000]:
        # Random 3D landmarks in front of the camera
        landmarks = torch.empty((n_pts, 3), dtype=torch.float32, device=device)
        landmarks[:, 0].uniform_(-2.0, 2.0)
        landmarks[:, 1].uniform_(-2.0, 2.0)
        landmarks[:, 2].uniform_(0.3, 3.0)

        u_obs, v_obs = project_landmarks_to_uv(landmarks, true_cam_pos, yaw, pitch)
        pi_world = pixel_uv_to_world_spinor(u_obs, v_obs, R_A)

        # Warmup
        for _ in range(10):
            _ = solve_camera_position_closed_form(landmarks, pi_world)
            _ = solve_classical_ray_triangulation(landmarks, u_obs, v_obs, yaw, pitch)

        iters = 100 if n_pts <= 10000 else 50

        sync_device(device)
        t0 = time.perf_counter()
        for _ in range(iters):
            pos_twistor = solve_camera_position_closed_form(landmarks, pi_world)
        sync_device(device)
        dt_twistor = ((time.perf_counter() - t0) / iters) * 1000.0

        sync_device(device)
        t0 = time.perf_counter()
        for _ in range(iters):
            pos_classical = solve_classical_ray_triangulation(landmarks, u_obs, v_obs, yaw, pitch)
        sync_device(device)
        dt_classical = ((time.perf_counter() - t0) / iters) * 1000.0

        speedup = dt_classical / dt_twistor
        diff = torch.linalg.norm(pos_twistor - pos_classical).item()
        print(f"{n_pts:14,d} | {dt_twistor:13.4f} | {dt_classical:15.4f} | {speedup:8.2f}x | {diff:11.2e}")


if __name__ == "__main__":
    run_scaling_benchmark()

