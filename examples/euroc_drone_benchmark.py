"""
EuRoC MAV Drone Flight Trajectory Benchmark for twistor-vision
==============================================================
Evaluates end-to-end 3D drone localization across a multi-waypoint MAV flight
using IMU quaternions mapped to SU(2) spinor matrices and batched GPU Spinor
RANSAC under 35% false feature matches and wide-angle optical noise.

Supports:
  1. Built-in realistic EuRoC-profile 3D figure-8 MAV flight (default, zero download)
  2. Real unpacked EuRoC MAV sequence folder via --euroc-dir path/to/mav0
"""

import argparse
import csv
import os
import time
import numpy as np
import torch
import twistor


def quat_wxyz_to_su2(q_wxyz: torch.Tensor) -> torch.Tensor:
    """
    Converts unit quaternion(s) [..., 4] (w, x, y, z) into unitary SU(2)
    spinor transformation matrix R_A [..., 2, 2] compatible with
    X_cam = R_A @ X_world @ R_A^dagger.
    """
    q = q_wxyz / torch.linalg.norm(q_wxyz, dim=-1, keepdim=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    # Standard SO(3) rotation from world to camera/body frame: X_cam = R_wc @ X_world
    # Conjugate transpose gives R_cw = R_wc^T; SU(2) double-cover of R_wc:
    # U = w I - i (x sigma_x + y sigma_y + z sigma_z)
    # Exact inverse quaternion (w, -x, -y, -z) mapped via U = q_w I - i(q_x sigma_x + q_y sigma_y + q_z sigma_z)
    u00 = torch.complex(w, z)
    u01 = torch.complex(y, x)
    u10 = torch.complex(-y, x)
    u11 = torch.complex(w, -z)

    row0 = torch.stack([u00, u01], dim=-1)
    row1 = torch.stack([u10, u11], dim=-1)
    return torch.stack([row0, row1], dim=-2).to(torch.complex64)


def load_euroc_groundtruth_csv(csv_path: str, max_frames: int = 600):
    """Loads timestamped 3D positions (p_x, p_y, p_z) and quaternions (q_w, q_x, q_y, q_z)."""
    positions, quats = [], []
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            # EuRoC format: timestamp, p_RS_R_x, y, z, q_RS_w, x, y, z, ...
            px, py, pz = float(row[1]), float(row[2]), float(row[3])
            qw, qx, qy, qz = float(row[4]), float(row[5]), float(row[6]), float(row[7])
            positions.append([px, py, pz])
            quats.append([qw, qx, qy, qz])
            if len(positions) >= max_frames:
                break
    return np.array(positions, dtype=np.float32), np.array(quats, dtype=np.float32)


def generate_euroc_profile_flight(num_frames: int = 600):
    """
    Generates a realistic 30-second (20 Hz) indoor Machine Hall MAV flight
    with aggressive banking roll/pitch, heading sweeps, and 3D altitude changes.
    """
    t = np.linspace(0.0, 30.0, num_frames, dtype=np.float32)
    # 3D Lissajous / figure-8 trajectory inside a 10m x 8m x 3m hall
    px = 3.2 * np.sin(0.42 * t)
    py = 2.1 * np.sin(0.84 * t)
    pz = 1.4 + 0.55 * np.cos(0.63 * t)
    positions = np.stack([px, py, pz], axis=-1)

    # Dynamic drone attitude: roll, pitch, yaw
    roll = 0.22 * np.cos(0.84 * t)
    pitch = 0.18 * np.sin(0.42 * t)
    yaw = 0.65 * np.sin(0.21 * t)

    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    quats = np.stack([qw, qx, qy, qz], axis=-1).astype(np.float32)
    return positions, quats


def project_world_to_camera_uv(landmarks_world: torch.Tensor, cam_pos_world: torch.Tensor, R_A: torch.Tensor):
    """Projects 3D world landmarks into stereographic camera coordinates (u, v) via SU(2)."""
    X_w = twistor.points_to_complex_hermitian(landmarks_world)
    C_w = twistor.points_to_complex_hermitian(cam_pos_world.unsqueeze(0))[0]
    R_dag = R_A.conj().mT
    X_c = torch.matmul(R_A, torch.matmul(X_w - C_w, R_dag))

    sqrt2 = 1.41421356
    x_c = X_c[:, 1, 0].real * sqrt2
    y_c = X_c[:, 1, 0].imag * sqrt2
    z_c = X_c[:, 0, 0].real * sqrt2

    in_front = z_c > 0.35
    r_c = torch.sqrt(x_c * x_c + y_c * y_c + z_c * z_c)
    denom = torch.clamp(r_c + z_c, min=0.35)
    u = x_c / denom
    v = y_c / denom
    in_fov = in_front & (u.abs() < 1.15) & (v.abs() < 0.95)
    return torch.stack([u, v], dim=-1), in_fov


def sync_device(device):
    d = str(device)
    if d == "xpu":
        torch.xpu.synchronize()
    elif d == "cuda":
        torch.cuda.synchronize()


def run_euroc_benchmark(euroc_dir: str | None = None, num_frames: int = 600, outlier_ratio: float = 0.35, imu_noise: float = 0.0, save_plot: bool = True):
    device = twistor.get_best_device()
    torch.manual_seed(7)
    np.random.seed(7)

    if euroc_dir and os.path.exists(os.path.join(euroc_dir, "state_groundtruth_estimate0", "data.csv")):
        gt_csv = os.path.join(euroc_dir, "state_groundtruth_estimate0", "data.csv")
        gt_pos_np, gt_quat_np = load_euroc_groundtruth_csv(gt_csv, max_frames=num_frames)
        source_label = f"EuRoC MAV Dataset ({euroc_dir})"
    else:
        gt_pos_np, gt_quat_np = generate_euroc_profile_flight(num_frames=num_frames)
        source_label = "EuRoC-Profile 3D MAV Flight (600 frames @ 20 Hz)"

    gt_pos = torch.tensor(gt_pos_np, dtype=torch.float32, device=device)
    gt_quat = torch.tensor(gt_quat_np, dtype=torch.float32, device=device)

    # Build a dense 3D room landmark map surrounding the flight envelope
    mins = gt_pos.min(dim=0).values - torch.tensor([3.5, 3.5, 1.5], device=device)
    maxs = gt_pos.max(dim=0).values + torch.tensor([3.5, 3.5, 3.5], device=device)
    map_landmarks = mins + torch.rand((2500, 3), device=device) * (maxs - mins)

    print(f"=== twistor-vision v{twistor.__version__} Drone Flight Benchmark [{device}] ===")
    print(f"Trajectory: {source_label} | Frames: {len(gt_pos)} | Outliers: {int(outlier_ratio * 100)}% | IMU Noise: {imu_noise}")

    est_positions = []
    frame_times_ms = []
    inlier_ratios = []

    # Warmup XPU/CUDA kernels
    R_warm = quat_wxyz_to_su2(gt_quat[0])
    uv_warm, mask_warm = project_world_to_camera_uv(map_landmarks, gt_pos[0], R_warm)
    for _ in range(5):
        _, _ = twistor.localize_ransac(map_landmarks[mask_warm], uv_warm[mask_warm], R_A=R_warm)

    for i in range(len(gt_pos)):
        # Simulate slight IMU attitude noise (~0.15 deg)
        q_noisy = gt_quat[i] + imu_noise * torch.randn(4, device=device) if imu_noise > 0 else gt_quat[i]
        R_A = quat_wxyz_to_su2(q_noisy)

        uv_all, visible = project_world_to_camera_uv(map_landmarks, gt_pos[i], quat_wxyz_to_su2(gt_quat[i]))
        pts_vis = map_landmarks[visible]
        uv_vis = uv_all[visible].clone()

        # Add pixel sensor jitter + 35% corrupted false feature matches
        uv_vis += 0.0015 * torch.randn_like(uv_vis)
        n_vis = pts_vis.shape[0]
        n_out = int(outlier_ratio * n_vis)
        if n_out > 0:
            out_idx = torch.randperm(n_vis, device=device)[:n_out]
            uv_vis[out_idx] = torch.empty((n_out, 2), device=device).uniform_(-1.2, 1.2)

        sync_device(device)
        t0 = time.perf_counter()
        pos_est, inliers = twistor.localize_ransac(
            pts_vis, uv_vis, R_A=R_A, num_hypotheses=256, inlier_threshold=0.009
        )
        sync_device(device)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        est_positions.append(pos_est)
        frame_times_ms.append(dt_ms)
        inlier_ratios.append(inliers.float().mean().item())

    est_pos = torch.stack(est_positions, dim=0)
    errors_m = torch.linalg.norm(est_pos - gt_pos, dim=-1)
    ate_rmse_mm = torch.sqrt(torch.mean(errors_m ** 2)).item() * 1000.0
    max_err_mm = errors_m.max().item() * 1000.0
    mean_ms = float(np.mean(frame_times_ms))
    fps = 1000.0 / mean_ms

    print(f"ATE RMSE:         {ate_rmse_mm:.2f} mm")
    print(f"Max Frame Error:  {max_err_mm:.2f} mm")
    print(f"Mean Latency:     {mean_ms:.3f} ms / frame ({fps:.0f} Hz)")
    print(f"Avg Inlier Ratio: {np.mean(inlier_ratios) * 100.0:.1f}% (true inliers: {(1.0 - outlier_ratio) * 100.0:.1f}%)")

    # Save 3D trajectory verification plot if matplotlib is available
    if save_plot:
      try:
        import matplotlib.pyplot as plt
        os.makedirs("assets", exist_ok=True)
        gt_np = gt_pos.cpu().numpy()
        est_np = est_pos.cpu().numpy()

        fig = plt.figure(figsize=(10, 4.5), dpi=140)
        ax1 = fig.add_subplot(121, projection="3d")
        ax1.plot(gt_np[:, 0], gt_np[:, 1], gt_np[:, 2], "k-", lw=2.0, label="Ground Truth MAV")
        ax1.plot(est_np[:, 0], est_np[:, 1], est_np[:, 2], "--", color="#00a8e8", lw=1.4, label="Twistor Spinor RANSAC")
        ax1.set_title(f"3D MAV Trajectory (ATE RMSE: {ate_rmse_mm:.2f} mm)")
        ax1.set_xlabel("X (m)")
        ax1.set_ylabel("Y (m)")
        ax1.set_zlabel("Z (m)")
        ax1.legend(loc="upper right", fontsize=8)

        ax2 = fig.add_subplot(122)
        ax2.plot(errors_m.cpu().numpy() * 1000.0, color="#ff595e", lw=1.2)
        ax2.axhline(ate_rmse_mm, color="#1982c4", ls="--", lw=1.2, label=f"ATE RMSE = {ate_rmse_mm:.2f} mm")
        ax2.set_title(f"Per-Frame Error @ {fps:.0f} Hz ({mean_ms:.2f} ms/frame)")
        ax2.set_xlabel("Flight Frame (20 Hz)")
        ax2.set_ylabel("3D Position Error (mm)")
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=8)

        plt.tight_layout()
        out_path = os.path.join("assets", "euroc_drone_trajectory.png")
        plt.savefig(out_path)
        print(f"Saved trajectory plot to {out_path}")
      except ImportError:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--euroc-dir", type=str, default=None, help="Optional path to unpacked EuRoC mav0 folder")
    parser.add_argument("--frames", type=int, default=600)
    args = parser.parse_args()
    run_euroc_benchmark(euroc_dir=args.euroc_dir, num_frames=args.frames, imu_noise=0.0, save_plot=True)
    print()
    run_euroc_benchmark(euroc_dir=args.euroc_dir, num_frames=args.frames, imu_noise=0.0013, save_plot=False)
