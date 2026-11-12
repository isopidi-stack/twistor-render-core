"""
Twistor Real 3D Scan 6-DoF Pose Recovery (demos/scan_pose_recovery.py)
======================================================================
Loads an external .ply or .obj 3D scan (automatically fetches the Stanford 
Bunny scan on first run) and optimizes a misaligned 6-DoF camera pose using 
native complex64 twistor eigenspinor incidence relations.

Usage:
    python demos/scan_pose_recovery.py [optional_path_to_scan.ply_or_obj]
"""

import sys
import os
import time
import urllib.request
import torch
import torchvision.utils as vutils

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import twistor.renderer as twistor_r
from twistor.io import load_point_cloud, save_ply_points


SAMPLE_SCAN_URL = "https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/stanford-bunny.obj"
DEFAULT_PLY_PATH = os.path.join(os.path.dirname(__file__), "stanford_bunny.ply")
TEMP_OBJ_PATH = os.path.join(os.path.dirname(__file__), "stanford_bunny.obj")


def get_scan_points(cli_arg=None, max_points=3500, device="cpu"):
    if cli_arg and os.path.exists(cli_arg):
        print(f"Loading user 3D scan: {cli_arg}")
        return load_point_cloud(cli_arg, max_points=max_points, normalize=True, device=device)

    if not os.path.exists(DEFAULT_PLY_PATH):
        try:
            print("Downloading Stanford Bunny 3D scan...")
            urllib.request.urlretrieve(SAMPLE_SCAN_URL, TEMP_OBJ_PATH)
            pts = load_point_cloud(TEMP_OBJ_PATH, max_points=max_points, normalize=True, device=device)
            save_ply_points(DEFAULT_PLY_PATH, pts)
            if os.path.exists(TEMP_OBJ_PATH):
                os.remove(TEMP_OBJ_PATH)
            print(f"Saved normalized PLY scan to: {DEFAULT_PLY_PATH}")
            return pts
        except Exception as e:
            print(f"Download skipped ({e}); generating asymmetric 3D trefoil scan...")
            t = torch.linspace(0, 2 * torch.pi, max_points, device=device)
            x = torch.sin(t) + 2.0 * torch.sin(2.0 * t)
            y = torch.cos(t) - 2.0 * torch.cos(2.0 * t)
            z = -torch.sin(3.0 * t)
            pts = torch.stack([x, y, z], dim=-1)
            pts = pts - pts.mean(dim=0, keepdim=True)
            return pts / torch.linalg.norm(pts, dim=-1).max()

    print(f"Loading cached 3D scan: {DEFAULT_PLY_PATH}")
    return load_point_cloud(DEFAULT_PLY_PATH, max_points=max_points, normalize=True, device=device)


def main():
    device = "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    cli_path = sys.argv[1] if len(sys.argv) > 1 else None
    pts = get_scan_points(cli_path, max_points=3500, device=device)
    print(f"Running 6-DoF Scan Pose Recovery on {device.upper()} ({pts.shape[0]} vertices)...")

    width, height = 160, 160
    sharp_thresh = 0.042
    sharp_tau = 0.0003

    # Ground truth camera pose
    true_pos = torch.tensor([0.12, -0.10, -1.85], device=device)
    true_yaw = torch.tensor(0.45, device=device)
    true_pitch = torch.tensor(-0.30, device=device)

    target_img = twistor_r.render_point_cloud(
        pts, width, height, threshold=sharp_thresh,
        yaw=true_yaw, pitch=true_pitch, cam_pos=true_pos, tau=sharp_tau, device=device
    ).detach()

    # Initial misaligned camera guess
    est_pos = torch.tensor([-0.15, 0.14, -2.30], device=device, requires_grad=True)
    est_yaw = torch.tensor(-0.25, device=device, requires_grad=True)
    est_pitch = torch.tensor(0.20, device=device, requires_grad=True)

    with torch.no_grad():
        initial_img = twistor_r.render_point_cloud(
            pts, width, height, threshold=sharp_thresh,
            yaw=est_yaw, pitch=est_pitch, cam_pos=est_pos, tau=sharp_tau, device=device
        )

    optimizer = torch.optim.Adam([
        {"params": [est_pos], "lr": 0.015},
        {"params": [est_yaw, est_pitch], "lr": 0.025}
    ])

    start_time = time.perf_counter()
    mid_img = None
    total_epochs = 150

    for epoch in range(total_epochs + 1):
        optimizer.zero_grad()

        progress = epoch / total_epochs
        curr_thresh = 0.22 * ((sharp_thresh / 0.22) ** progress)
        curr_tau = 0.008 * ((sharp_tau / 0.008) ** progress)

        target_curr = twistor_r.render_point_cloud(
            pts, width, height, threshold=curr_thresh,
            yaw=true_yaw, pitch=true_pitch, cam_pos=true_pos, tau=curr_tau, device=device
        ).detach()

        pred_img = twistor_r.render_point_cloud(
            pts, width, height, threshold=curr_thresh,
            yaw=est_yaw, pitch=est_pitch, cam_pos=est_pos, tau=curr_tau, device=device
        )

        loss = torch.nn.functional.mse_loss(pred_img, target_curr)
        loss.backward()
        optimizer.step()

        if epoch == 45:
            with torch.no_grad():
                mid_img = twistor_r.render_point_cloud(
                    pts, width, height, threshold=sharp_thresh,
                    yaw=est_yaw, pitch=est_pitch, cam_pos=est_pos, tau=sharp_tau, device=device
                )

        if epoch % 30 == 0:
            print(
                f"Epoch {epoch:03d} | Loss: {loss.item():.6f} | "
                f"Pos: [{est_pos[0].item():.2f}, {est_pos[1].item():.2f}, {est_pos[2].item():.2f}] | "
                f"Yaw: {est_yaw.item():.3f} (Target {true_yaw.item():.3f}) | "
                f"Pitch: {est_pitch.item():.3f} (Target {true_pitch.item():.3f})"
            )

    elapsed = time.perf_counter() - start_time

    with torch.no_grad():
        final_img = twistor_r.render_point_cloud(
            pts, width, height, threshold=sharp_thresh,
            yaw=est_yaw, pitch=est_pitch, cam_pos=est_pos, tau=sharp_tau, device=device
        )

    grid = torch.stack([
        initial_img.cpu().unsqueeze(0),
        mid_img.cpu().unsqueeze(0),
        final_img.cpu().unsqueeze(0),
        target_img.cpu().unsqueeze(0)
    ], dim=0)

    out_path = "scan_pose_recovery.png"
    vutils.save_image(grid, out_path, nrow=4, padding=6, pad_value=0.35)
    print(f"\nScan pose recovery finished in {elapsed:.2f}s!")
    print(f"Saved 4-panel scan progression to: {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()