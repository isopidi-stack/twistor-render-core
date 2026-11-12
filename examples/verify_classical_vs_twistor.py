"""
Verification Benchmark: Twistor Spinor Localization vs. Classical Ray Triangulation
===================================================================================
Audits whether Twistor closed-form localization matches classical 3D projective
ray triangulation in accuracy and speed when camera orientation (IMU) is known.
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


def solve_classical_ray_triangulation(landmarks_xyz, u_obs, v_obs, yaw, pitch):
    """
    Standard Euclidean 3D ray triangulation (minimizing point-to-ray perpendicular distance):
    sum_i || (I - d_i d_i^T) (X_i - C) ||^2
    """
    device = landmarks_xyz.device
    # Match the exact spinor stereographic ray direction from (u, v)
    # pi = [1, w] where w = u + i*v -> unit 3D ray in camera frame:
    # x = 2u / (1 + u^2 + v^2), y = 2v / (1 + u^2 + v^2), z = (1 - u^2 - v^2) / (1 + u^2 + v^2)
    r2 = u_obs ** 2 + v_obs ** 2
    denom = 1.0 + r2
    dx_c = (2.0 * u_obs) / denom
    dy_c = (2.0 * v_obs) / denom
    dz_c = (1.0 - r2) / denom
    d_cam = torch.stack([dx_c, dy_c, dz_c], dim=-1)  # [N, 3]

    # Rotate camera ray directions into world frame using the same SU(2) rotation
    R_A, _, _ = generate_su22_camera_matrix(yaw=yaw, pitch=pitch, device=device)
    R_A_dag = R_A.conj().mT
    D_cam_herm = points_to_complex_hermitian(d_cam)
    D_world_herm = torch.matmul(R_A_dag, torch.matmul(D_cam_herm, R_A))
    d_world = torch.stack([
        D_world_herm[:, 1, 0].real * np.sqrt(2.0),
        D_world_herm[:, 1, 0].imag * np.sqrt(2.0),
        D_world_herm[:, 0, 0].real * np.sqrt(2.0),
    ], dim=-1)  # [N, 3] unit world rays

    # Build 3x3 normal equations: sum_i (I - d_i d_i^T) C = sum_i (I - d_i d_i^T) X_i
    eye3 = torch.eye(3, device=device).unsqueeze(0)  # [1, 3, 3]
    P_perp = eye3 - torch.einsum("ni,nj->nij", d_world, d_world)  # [N, 3, 3]

    A = torch.sum(P_perp, dim=0) + 1e-9 * torch.eye(3, device=device)  # [3, 3]
    b = torch.sum(torch.einsum("nij,nj->ni", P_perp, landmarks_xyz), dim=0)  # [3]
    return torch.linalg.solve(A, b)


def run_audit():
    device = get_best_device()
    print(f"=== Rigorous Audit: Twistor vs. Classical Localization ({device}) ===")

    landmarks = torch.tensor([
        [-0.75, -0.55,  0.60],
        [ 0.80, -0.45,  0.85],
        [-0.60,  0.70,  0.50],
        [ 0.65,  0.60,  0.95],
        [ 0.00, -0.10,  1.20],
        [-0.30,  0.25,  0.35],
    ], dtype=torch.float32, device=device)

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

    u_obs, v_obs = project_landmarks_to_uv(landmarks, true_cam_pos, yaw, pitch)
    pi_world = pixel_uv_to_world_spinor(u_obs, v_obs, R_A)

    # Warmup both
    for _ in range(10):
        _ = solve_camera_position_closed_form(landmarks, pi_world)
        _ = solve_classical_ray_triangulation(landmarks, u_obs, v_obs, yaw, pitch)

    # Time Twistor (100 runs)
    if str(device) == "xpu": torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(100):
        pos_twistor = solve_camera_position_closed_form(landmarks, pi_world)
    if str(device) == "xpu": torch.xpu.synchronize()
    dt_twistor = (time.perf_counter() - t0) * 10.0  # ms per call

    # Time Classical (100 runs)
    t0 = time.perf_counter()
    for _ in range(100):
        pos_classical = solve_classical_ray_triangulation(landmarks, u_obs, v_obs, yaw, pitch)
    if str(device) == "xpu": torch.xpu.synchronize()
    dt_classical = (time.perf_counter() - t0) * 10.0  # ms per call

    diff_between_methods = torch.linalg.norm(pos_twistor - pos_classical).item()
    err_twistor = torch.linalg.norm(pos_twistor - true_pos_world).item()
    err_classical = torch.linalg.norm(pos_classical - true_pos_world).item()

    print(f"Twistor Position   : {pos_twistor.tolist()} | Err: {err_twistor:.2e} | Avg Time: {dt_twistor:.4f} ms")
    print(f"Classical Position : {pos_classical.tolist()} | Err: {err_classical:.2e} | Avg Time: {dt_classical:.4f} ms")
    print(f"Difference between Twistor & Classical solutions : {diff_between_methods:.2e} units")


if __name__ == "__main__":
    run_audit()
