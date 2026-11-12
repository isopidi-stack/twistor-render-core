"""
Kidnapped Robot Closed-Form 3D Localization via Twistor Duality
==============================================================
Demonstrates Sir Roger Penrose's Twistor Duality (Klein correspondence):
Because light rays are encoded as points in projective twistor space (CP^3),
matching 3+ pixel rays to known 3D map landmarks solves the camera's exact
3D position C = (x, y, z) in ONE closed-form linear matrix step -- with zero
initial guess and zero iterative optimization.
"""

import os
import time
import torch
import numpy as np
from PIL import Image, ImageDraw
from twistor.geometry import get_best_device
from twistor.renderer import (
    generate_su22_camera_matrix,
    points_to_complex_hermitian,
    render_point_cloud,
)


def pixel_uv_to_world_spinor(u, v, R_A):
    """
    Converts normalized camera sensor coordinates (u, v) into a unit world-frame
    ray 2-spinor pi_world = R_A^dagger @ pi_cam.
    """
    pi_0 = torch.ones_like(u, dtype=torch.complex64)
    pi_1 = torch.complex(u, v)
    pi_cam = torch.stack([pi_0, pi_1], dim=-1)  # [N, 2]
    pi_cam = pi_cam / torch.linalg.norm(pi_cam, dim=-1, keepdim=True)

    # Since X_cam = R_A @ X_world @ R_A^dagger, eigenspinor incidence
    # X_cam @ pi_cam = lambda * pi_cam transforms to world frame as:
    # X_world @ (R_A^dagger @ pi_cam) = lambda * (R_A^dagger @ pi_cam)
    R_A_dag = R_A.conj().mT
    pi_world = torch.matmul(pi_cam, R_A_dag.mT)  # [N, 2]
    pi_world = pi_world / torch.linalg.norm(pi_world, dim=-1, keepdim=True)
    return pi_world


def solve_camera_position_closed_form(landmarks_xyz, ray_spinors_pi):
    """
    Fused float32 normal-equation solver for projective twistor incidence:
        <pi_perp | (X_i - C) | pi> = 0
    Eliminates [N, 2, 2] complex Hermitian allocations and complex concatenations
    for high-throughput scaling up to 100,000+ landmarks.
    """
    # Extract real & imaginary parts of pi0 = a0 + i*b0, pi1 = a1 + i*b1
    a0 = ray_spinors_pi[:, 0].real
    b0 = ray_spinors_pi[:, 0].imag
    a1 = ray_spinors_pi[:, 1].real
    b1 = ray_spinors_pi[:, 1].imag

    # Quadratic spinor basis terms (factor 1/sqrt(2) cancels on both sides)
    # p0_sq = pi0^2 = (a0^2 - b0^2) + i*(2*a0*b0)
    # p1_sq = pi1^2 = (a1^2 - b1^2) + i*(2*a1*b1)
    # p0_p1 = pi0*pi1 = (a0*a1 - b0*b1) + i*(a0*b1 + b0*a1)
    re_p0_sq = a0 * a0 - b0 * b0
    im_p0_sq = 2.0 * a0 * b0
    re_p1_sq = a1 * a1 - b1 * b1
    im_p1_sq = 2.0 * a1 * b1
    re_p0_p1 = a0 * a1 - b0 * b1
    im_p0_p1 = a0 * b1 + b0 * a1

    # Real and imaginary rows of [coeff_cx, coeff_cy, coeff_cz]:
    # coeff_cx = pi0^2 - pi1^2
    # coeff_cy = i * (pi0^2 + pi1^2) -> Re = -(im_p0_sq + im_p1_sq), Im = (re_p0_sq + re_p1_sq)
    # coeff_cz = -2 * pi0 * pi1
    r_cx = re_p0_sq - re_p1_sq
    r_cy = -(im_p0_sq + im_p1_sq)
    r_cz = -2.0 * re_p0_p1

    i_cx = im_p0_sq - im_p1_sq
    i_cy = re_p0_sq + re_p1_sq
    i_cz = -2.0 * im_p0_p1

    x = landmarks_xyz[:, 0]
    y = landmarks_xyz[:, 1]
    z = landmarks_xyz[:, 2]

    # RHS scalar <pi_perp | X_i | pi> = r_cx*x + r_cy*y + r_cz*z + i*(i_cx*x + i_cy*y + i_cz*z)
    rhs_r = r_cx * x + r_cy * y + r_cz * z
    rhs_i = i_cx * x + i_cy * y + i_cz * z

    # Stack [N, 3] real and [N, 3] imag into [2N, 3] directly in float32
    A_real = torch.cat([
        torch.stack([r_cx, r_cy, r_cz], dim=-1),
        torch.stack([i_cx, i_cy, i_cz], dim=-1),
    ], dim=0)
    b_real = torch.cat([rhs_r, rhs_i], dim=0).unsqueeze(-1)

    ATA = torch.matmul(A_real.mT, A_real) + 1e-9 * torch.eye(3, device=A_real.device)
    ATb = torch.matmul(A_real.mT, b_real)
    sol = torch.linalg.solve(ATA, ATb).squeeze(-1)
    return sol


def project_landmarks_to_uv(landmarks_xyz, true_cam_pos, yaw, pitch):
    """
    Computes exact sensor (u, v) coordinates where 3D landmarks hit the camera sensor.
    """
    device = landmarks_xyz.device
    R_A, _, _ = generate_su22_camera_matrix(yaw=yaw, pitch=pitch, boost_v=0.0, zoom=1.0, device=device)
    R_A_dag = R_A.conj().mT

    # Rotate world landmarks into camera spinor frame, then subtract camera-frame origin
    X_world = points_to_complex_hermitian(landmarks_xyz)
    X_cam_pts = torch.matmul(R_A, torch.matmul(X_world, R_A_dag))
    C_cam = points_to_complex_hermitian(true_cam_pos.unsqueeze(0))[0]
    X_rel_cam = X_cam_pts - C_cam.unsqueeze(0)

    z_c = X_rel_cam[:, 0, 0].real * np.sqrt(2.0)
    x_plus_iy = X_rel_cam[:, 1, 0] * np.sqrt(2.0)
    r_norm = torch.sqrt(z_c ** 2 + torch.abs(x_plus_iy) ** 2)

    w = x_plus_iy / (r_norm + z_c + 1e-8)
    return w.real, w.imag


def run_kidnapped_robot_demo():
    device = get_best_device()
    print(f"=== Kidnapped Robot Closed-Form Localization Demo ({device}) ===")

    landmarks = torch.tensor([
        [-0.75, -0.55,  0.60],
        [ 0.80, -0.45,  0.85],
        [-0.60,  0.70,  0.50],
        [ 0.65,  0.60,  0.95],
        [ 0.00, -0.10,  1.20],
        [-0.30,  0.25,  0.35],
    ], dtype=torch.float32, device=device)

    # Robot wakes up completely lost at an unknown 3D coordinate
    true_pos_world = torch.tensor([0.35, -0.25, -2.10], dtype=torch.float32, device=device)
    known_yaw = 0.18
    known_pitch = -0.12

    # Express camera translation in camera spinor frame (matching renderer.py convention)
    R_A, _, _ = generate_su22_camera_matrix(yaw=known_yaw, pitch=known_pitch, device=device)
    R_A_dag = R_A.conj().mT
    C_world_herm = points_to_complex_hermitian(true_pos_world.unsqueeze(0))[0]
    C_cam_herm = torch.matmul(R_A, torch.matmul(C_world_herm, R_A_dag))
    true_cam_pos = torch.stack([
        C_cam_herm[1, 0].real * np.sqrt(2.0),
        C_cam_herm[1, 0].imag * np.sqrt(2.0),
        C_cam_herm[0, 0].real * np.sqrt(2.0),
    ])

    u_obs, v_obs = project_landmarks_to_uv(landmarks, true_cam_pos, known_yaw, known_pitch)
    pi_world = pixel_uv_to_world_spinor(u_obs, v_obs, R_A)

    # Warmup
    _ = solve_camera_position_closed_form(landmarks, pi_world)

    t0 = time.perf_counter()
    est_pos_world = solve_camera_position_closed_form(landmarks, pi_world)
    if str(device) == "xpu":
        torch.xpu.synchronize()
    elif str(device) == "cuda":
        torch.cuda.synchronize()
    dt_ms = (time.perf_counter() - t0) * 1000.0

    err_exact = torch.linalg.norm(est_pos_world - true_pos_world).item()
    print(f"\n[Test 1: Exact Sensor Rays -- Zero Initial Guess]")
    print(f"  Ground Truth World XYZ : ({true_pos_world[0]:+.4f}, {true_pos_world[1]:+.4f}, {true_pos_world[2]:+.4f})")
    print(f"  Recovered World XYZ    : ({est_pos_world[0]:+.4f}, {est_pos_world[1]:+.4f}, {est_pos_world[2]:+.4f})")
    print(f"  3D Position Error      : {err_exact:.2e} units")
    print(f"  Solve Time             : {dt_ms:.3f} ms (1 closed-form matrix step)")

    # Test 2: Noisy camera sensor (+/- 1.5 px jitter)
    torch.manual_seed(42)
    pixel_noise_std = 1.3 / 128.0
    u_noisy = u_obs + torch.randn_like(u_obs) * pixel_noise_std
    v_noisy = v_obs + torch.randn_like(v_obs) * pixel_noise_std
    pi_noisy = pixel_uv_to_world_spinor(u_noisy, v_noisy, R_A)
    est_pos_noisy = solve_camera_position_closed_form(landmarks, pi_noisy)
    err_noisy = torch.linalg.norm(est_pos_noisy - true_pos_world).item()

    print(f"\n[Test 2: Noisy Camera Sensor (+/- 1.5 px jitter)]")
    print(f"  Recovered World XYZ    : ({est_pos_noisy[0]:+.4f}, {est_pos_noisy[1]:+.4f}, {est_pos_noisy[2]:+.4f})")
    print(f"  3D Position Error      : {err_noisy:.4f} units")

    # Convert recovered world position back to camera frame for renderer.py
    C_est_world = points_to_complex_hermitian(est_pos_noisy.unsqueeze(0))[0]
    C_est_cam = torch.matmul(R_A, torch.matmul(C_est_world, R_A_dag))
    est_cam_pos = torch.stack([
        C_est_cam[1, 0].real * np.sqrt(2.0),
        C_est_cam[1, 0].imag * np.sqrt(2.0),
        C_est_cam[0, 0].real * np.sqrt(2.0),
    ])

    # Render side-by-side visual proof
    os.makedirs("assets", exist_ok=True)
    edges = []
    corners = [
        (-0.8, -0.6, 0.4), (0.8, -0.6, 0.4), (0.8, 0.6, 0.4), (-0.8, 0.6, 0.4),
        (-0.8, -0.6, 1.2), (0.8, -0.6, 1.2), (0.8, 0.6, 1.2), (-0.8, 0.6, 1.2),
    ]
    edge_pairs = [
        (0,1),(1,2),(2,3),(3,0), (4,5),(5,6),(6,7),(7,4), (0,4),(1,5),(2,6),(3,7)
    ]
    t_vals = torch.linspace(0, 1, 45)
    for i, j in edge_pairs:
        p0 = torch.tensor(corners[i], dtype=torch.float32)
        p1 = torch.tensor(corners[j], dtype=torch.float32)
        edges.append(p0 + t_vals.unsqueeze(-1) * (p1 - p0))
    scene_pts = torch.cat([torch.cat(edges, dim=0).to(device), landmarks], dim=0)

    with torch.no_grad():
        img_gt = render_point_cloud(
            scene_pts, width=192, height=192, threshold=0.038,
            yaw=known_yaw, pitch=known_pitch, cam_pos=true_cam_pos, shading=True, device=device
        )
        img_rec = render_point_cloud(
            scene_pts, width=192, height=192, threshold=0.038,
            yaw=known_yaw, pitch=known_pitch, cam_pos=est_cam_pos, shading=True, device=device
        )

    def to_rgb_panel(tensor_2d, title, subtitle, tint=(0.4, 0.95, 1.15)):
        arr = (tensor_2d.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        rgb = np.zeros((192, 192, 3), dtype=np.uint8)
        rgb[..., 0] = np.clip(arr * tint[0], 0, 255).astype(np.uint8)
        rgb[..., 1] = np.clip(arr * tint[1], 0, 255).astype(np.uint8)
        rgb[..., 2] = np.clip(arr * tint[2], 0, 255).astype(np.uint8)
        im = Image.fromarray(rgb).resize((320, 320), Image.Resampling.BILINEAR)
        draw = ImageDraw.Draw(im)
        draw.text((12, 10), title, fill=(255, 255, 255))
        draw.text((12, 296), subtitle, fill=(190, 235, 255))
        return im

    panel_left = to_rgb_panel(
        img_gt,
        "1. Lost Robot Camera Snap",
        f"True XYZ: ({true_pos_world[0]:.2f}, {true_pos_world[1]:.2f}, {true_pos_world[2]:.2f})",
        tint=(1.15, 0.85, 0.45),
    )
    panel_right = to_rgb_panel(
        img_rec,
        "2. One-Shot Twistor Solve",
        f"Est XYZ:  ({est_pos_noisy[0]:.2f}, {est_pos_noisy[1]:.2f}, {est_pos_noisy[2]:.2f}) [{dt_ms:.2f}ms]",
        tint=(0.45, 1.05, 1.15),
    )

    canvas = Image.new("RGB", (648, 320), (15, 18, 24))
    canvas.paste(panel_left, (0, 0))
    canvas.paste(panel_right, (328, 0))
    out_img = "assets/kidnapped_robot_snap.png"
    canvas.save(out_img)
    print(f"\nSaved visual proof to {out_img}")


if __name__ == "__main__":
    run_kidnapped_robot_demo()

