"""
Twistor Visual 6-DoF Pose Recovery Showcase (v0.4)
=================================================
Optimizes camera position AND SU(2) spinor orientation (yaw, pitch) using 
SU(2) spinor conjugation and eigenspinor incidence annealing.
"""

import sys
import os
import time
import torch
import torchvision.utils as vutils

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import twistor.renderer as twistor_r


def generate_torus_point_cloud(num_points=2500, major_r=0.85, minor_r=0.20, device="cpu"):
    """Generates a high-contrast 3D torus point cloud with a clear center hole."""
    u = torch.rand(num_points, device=device) * 2.0 * torch.pi
    v = torch.rand(num_points, device=device) * 2.0 * torch.pi

    x = (major_r + minor_r * torch.cos(v)) * torch.cos(u)
    y = (major_r + minor_r * torch.cos(v)) * torch.sin(u)
    z = minor_r * torch.sin(v)

    return torch.stack([x, y, z], dim=-1)


def main():
    device = "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu"
    print(f"Running 6-DoF Visual Pose Recovery on {device.upper()}...")

    width, height = 160, 160
    sharp_thresh = 0.045
    sharp_tau = 0.0003
    pts = generate_torus_point_cloud(num_points=2500, device=device)

    # Ground truth camera pose (tilted and offset)
    true_pos = torch.tensor([0.18, -0.14, -2.0], device=device)
    true_yaw = torch.tensor(0.55, device=device)
    true_pitch = torch.tensor(-0.45, device=device)

    target_img = twistor_r.render_point_cloud(
        pts, width, height, threshold=sharp_thresh,
        yaw=true_yaw, pitch=true_pitch, cam_pos=true_pos, tau=sharp_tau, device=device
    ).detach()

    print(f"Target image peak intensity: {target_img.max().item():.3f}, mean: {target_img.mean().item():.3f}")

    # Initial misaligned camera guess
    est_pos = torch.tensor([-0.15, 0.15, -2.4], device=device, requires_grad=True)
    est_yaw = torch.tensor(-0.25, device=device, requires_grad=True)
    est_pitch = torch.tensor(0.25, device=device, requires_grad=True)

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

        # Anneal both kernel bandwidth (threshold) and softmin sharpness (tau)
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

    out_path = "pose_recovery_showcase.png"
    vutils.save_image(grid, out_path, nrow=4, padding=6, pad_value=0.35)
    print(f"\nOptimization finished in {elapsed:.2f}s!")
    print(f"Saved crisp 4-panel visual progression to: {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()