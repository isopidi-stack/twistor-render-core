"""
Full SU(2,2) Conformal Robot Camera Self-Calibration (v0.8 Demo)
================================================================
Demonstrates joint recovery of:
1. Camera Orientation (SU(2) yaw & pitch)
2. Optical Focal Zoom (Conformal Dilation D)
3. Wide-Angle / Fisheye Lens Distortion (Transverse Special Conformal K(b_x, b_y))
simultaneously from a single 2D image via differentiable twistor rendering.
"""

import os
import torch
import numpy as np
from PIL import Image, ImageDraw
from twistor.geometry import get_best_device
from twistor.renderer import render_point_cloud


def build_calibration_grid(device):
    """Builds an asymmetric 3D calibration target (outer ring + inner triangle + crossbars)."""
    pts = []
    # Outer ring at z = 0.7
    theta = torch.linspace(0, 2 * torch.pi, 140)[:-1]
    pts.append(torch.stack([0.65 * torch.cos(theta), 0.65 * torch.sin(theta), torch.full_like(theta, 0.7)], dim=-1))

    # Asymmetric inner triangle at z = 0.45 (breaks rotational & zoom self-similarity)
    tri_verts = [
        torch.tensor([0.0, 0.36, 0.45]),
        torch.tensor([-0.32, -0.22, 0.45]),
        torch.tensor([0.34, -0.18, 0.45]),
    ]
    t_edge = torch.linspace(0, 1, 50)
    for i in range(3):
        p0, p1 = tri_verts[i], tri_verts[(i + 1) % 3]
        pts.append(p0 + t_edge.unsqueeze(-1) * (p1 - p0))

    # Crossbars at z = 0.7
    line = torch.linspace(-0.65, 0.65, 65)
    zeros = torch.zeros_like(line)
    pts.append(torch.stack([line, zeros, torch.full_like(line, 0.7)], dim=-1))
    pts.append(torch.stack([zeros, line, torch.full_like(line, 0.7)], dim=-1))
    return torch.cat(pts, dim=0).to(device)


def run_su22_calibration_demo():
    device = get_best_device()
    print(f"=== Full SU(2,2) Conformal Robot Self-Calibration ({device}) ===")

    pts = build_calibration_grid(device)

    # Unknown ground-truth camera + lens parameters
    true_yaw = 0.24
    true_pitch = -0.18
    true_zoom = 1.38
    true_b = torch.tensor([0.22, -0.19, 0.00], dtype=torch.float32, device=device)

    with torch.no_grad():
        target_img = render_point_cloud(
            pts, width=112, height=112, threshold=0.042,
            yaw=true_yaw, pitch=true_pitch, zoom=true_zoom, conformal_b=true_b,
            shading=True, device=device
        )

    # Initialize uncalibrated camera (wrong angle, wrong zoom, zero lens distortion)
    est_yaw = torch.tensor(0.0, dtype=torch.float32, device=device, requires_grad=True)
    est_pitch = torch.tensor(0.0, dtype=torch.float32, device=device, requires_grad=True)
    est_zoom = torch.tensor(1.0, dtype=torch.float32, device=device, requires_grad=True)
    est_b_xy = torch.tensor([0.0, 0.0], dtype=torch.float32, device=device, requires_grad=True)

    optimizer = torch.optim.Adam([
        {"params": [est_yaw, est_pitch], "lr": 0.025},
        {"params": [est_zoom], "lr": 0.025},
        {"params": [est_b_xy], "lr": 0.02},
    ])

    snapshots = []
    for epoch in range(1, 121):
        optimizer.zero_grad()
        # Coarse-to-fine ray incidence radius for wide capture basin
        progress = min(1.0, epoch / 60.0)
        curr_thresh = 0.085 * (1.0 - progress) + 0.042 * progress

        with torch.no_grad():
            target_curr = render_point_cloud(
                pts, width=112, height=112, threshold=curr_thresh,
                yaw=true_yaw, pitch=true_pitch, zoom=true_zoom, conformal_b=true_b,
                shading=True, device=device
            )

        est_b = torch.stack([est_b_xy[0], est_b_xy[1], torch.zeros_like(est_b_xy[0])])
        pred_img = render_point_cloud(
            pts, width=112, height=112, threshold=curr_thresh,
            yaw=est_yaw, pitch=est_pitch, zoom=est_zoom, conformal_b=est_b,
            shading=True, device=device
        )
        loss = torch.nn.functional.mse_loss(pred_img, target_curr)
        loss.backward()
        optimizer.step()

        if epoch in [1, 15, 30, 50, 75, 100, 120]:
            b_np = est_b.detach().cpu().numpy().copy()
            snapshots.append((epoch, loss.item(), pred_img.detach().clone(), est_zoom.item(), b_np))
            print(
                f"Epoch {epoch:03d} | Loss: {loss.item():.6f} | "
                f"Zoom: {est_zoom.item():.2f} (GT {true_zoom:.2f}) | "
                f"Conformal b_xy: [{b_np[0]:+.2f}, {b_np[1]:+.2f}] (GT [+0.22, -0.19])"
            )

    # Export side-by-side calibration GIF
    os.makedirs("assets", exist_ok=True)
    gt_arr = (target_img.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    gt_rgb = np.stack([
        np.clip(gt_arr * 1.15, 0, 255).astype(np.uint8),
        np.clip(gt_arr * 0.85, 0, 255).astype(np.uint8),
        np.clip(gt_arr * 0.45, 0, 255).astype(np.uint8),
    ], axis=-1)
    gt_pil = Image.fromarray(gt_rgb).resize((280, 280), Image.Resampling.BILINEAR)
    ImageDraw.Draw(gt_pil).text((10, 10), "Target Warped Sensor View", fill=(255, 230, 150))

    frames = []
    for ep, l_val, p_img, z_val, b_val in snapshots:
        p_arr = (p_img.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        p_rgb = np.stack([
            np.clip(p_arr * 0.45, 0, 255).astype(np.uint8),
            np.clip(p_arr * 1.05, 0, 255).astype(np.uint8),
            np.clip(p_arr * 1.15, 0, 255).astype(np.uint8),
        ], axis=-1)
        p_pil = Image.fromarray(p_rgb).resize((280, 280), Image.Resampling.BILINEAR)
        draw = ImageDraw.Draw(p_pil)
        draw.text((10, 10), f"SU(2,2) Calibrating | Ep {ep:03d}", fill=(255, 255, 255))
        draw.text((10, 258), f"Zoom:{z_val:.2f} | b_xy:[{b_val[0]:+.2f},{b_val[1]:+.2f}]", fill=(185, 240, 255))

        frame = Image.new("RGB", (568, 280), (14, 17, 22))
        frame.paste(gt_pil, (0, 0))
        frame.paste(p_pil, (288, 0))
        frames.append(frame)

    frames.extend([frames[-1]] * 6)
    out_gif = "assets/su22_robot_calibration.gif"
    frames[0].save(out_gif, save_all=True, append_images=frames[1:], duration=220, loop=0)
    print(f"\nSaved SU(2,2) calibration animation to {out_gif}")


if __name__ == "__main__":
    run_su22_calibration_demo()
