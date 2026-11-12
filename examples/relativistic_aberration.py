"""
Relativistic Lorentz Aberration & Inverse Velocity Recovery (v0.7 Demo)
=======================================================================
Demonstrates SL(2, C) spinor Lorentz boosts on celestial ray spinors:
1. Recovers unknown relativistic velocity (beta = v/c) via 2D pixel gradients.
2. Exports an animated GIF (assets/relativistic_aberration.gif) showing
   Penrose-Terrell relativistic aberration from -0.82c to +0.88c.
"""

import os
import math
import torch
import numpy as np
from PIL import Image, ImageDraw
from twistor.geometry import get_best_device
from twistor.renderer import render_point_cloud


def build_starfield_lattice(device):
    """Creates a 3D tunnel of rings + central torus to visualize relativistic aberration."""
    pts = []
    # 5 concentric depth rings along the Z-axis
    for z in torch.linspace(-1.2, 1.6, 6):
        theta = torch.linspace(0, 2 * math.pi, 90)[:-1]
        r = 0.85
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)
        zz = torch.full_like(x, z.item())
        pts.append(torch.stack([x, y, zz], dim=-1))

    # Central tilted torus
    u = torch.linspace(0, 2 * math.pi, 60)[:-1]
    v = torch.linspace(0, 2 * math.pi, 24)[:-1]
    uu, vv = torch.meshgrid(u, v, indexing="ij")
    R, r = 0.42, 0.14
    tx = (R + r * torch.cos(vv)) * torch.cos(uu)
    ty = (R + r * torch.cos(vv)) * torch.sin(uu)
    tz = r * torch.sin(vv)
    torus = torch.stack([tx.reshape(-1), ty.reshape(-1), tz.reshape(-1)], dim=-1)
    pts.append(torus)

    return torch.cat(pts, dim=0).to(device)


def run_inverse_velocity_test(points, device):
    print("\n--- Part 1: Differentiable Relativistic Velocity Recovery ---")
    true_beta = 0.78
    with torch.no_grad():
        target_img = render_point_cloud(
            points, width=96, height=96, threshold=0.045,
            yaw=0.25, pitch=-0.15, boost_v=true_beta, shading=True, device=device
        )

    est_beta = torch.tensor(0.0, dtype=torch.float32, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([est_beta], lr=0.03)

    for epoch in range(1, 81):
        optimizer.zero_grad()
        pred_img = render_point_cloud(
            points, width=96, height=96, threshold=0.045,
            yaw=0.25, pitch=-0.15, boost_v=est_beta, shading=True, device=device
        )
        loss = torch.nn.functional.mse_loss(pred_img, target_img)
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d} | Loss: {loss.item():.6f} | Recovered v/c: {est_beta.item():.4f}c (Target: {true_beta:.4f}c)")


def generate_aberration_gif(points, device, out_path="assets/relativistic_aberration.gif"):
    print("\n--- Part 2: Rendering Relativistic Aberration GIF ---")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Smooth sinusoidal sweep: 0 -> +0.88c -> 0 -> -0.82c -> 0
    num_frames = 48
    betas = []
    for i in range(num_frames):
        phase = 2 * math.pi * i / num_frames
        b = 0.85 * math.sin(phase)
        betas.append(b)

    frames = []
    for idx, beta in enumerate(betas):
        with torch.no_grad():
            img_t = render_point_cloud(
                points, width=192, height=192, threshold=0.042,
                yaw=0.22 * math.cos(2 * math.pi * idx / num_frames),
                pitch=-0.12,
                boost_v=beta,
                shading=True,
                depth_contrast=2.4,
                device=device
            )

        arr = (img_t.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        # Apply cyan-gold relativistic colormap
        rgb = np.zeros((192, 192, 3), dtype=np.uint8)
        rgb[..., 0] = np.clip(arr * 1.15, 0, 255).astype(np.uint8)
        rgb[..., 1] = np.clip(arr * 0.85, 0, 255).astype(np.uint8)
        rgb[..., 2] = np.clip(arr * 0.45 + (arr > 15) * 40, 0, 255).astype(np.uint8)

        pil_img = Image.fromarray(rgb).resize((384, 384), Image.Resampling.BILINEAR)
        draw = ImageDraw.Draw(pil_img)
        gamma = 1.0 / math.sqrt(max(1e-4, 1.0 - beta ** 2))
        draw.text((14, 12), "SL(2,C) Relativistic Lorentz Aberration", fill=(255, 225, 140))
        draw.text((14, 354), f"Velocity: {beta:+.2f}c   |   Lorentz Factor y: {gamma:.2f}", fill=(180, 230, 255))
        frames.append(pil_img)

    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=75,
        loop=0
    )
    print(f"Saved GIF to {out_path}")


if __name__ == "__main__":
    device = get_best_device()
    print(f"Running v0.7 SL(2,C) Lorentz Boost Demo on: {device}")
    pts = build_starfield_lattice(device)
    run_inverse_velocity_test(pts, device)
    generate_aberration_gif(pts, device)
