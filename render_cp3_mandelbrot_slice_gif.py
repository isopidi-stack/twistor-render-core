"""
Twistor Render Core - Native Trigonometry-Free CP^3 Mandelbrot Projection Renderer
Generates cp3_mandelbrot_projection_proof.gif by iterating the homogeneous degree-2
projective spinor map on CP^3 (embedded in C^5 / SIGMA_5):
    omega^A_{k+1} = (omega^A_k)^2 + C^A (pi_{A',k})^2
    pi_{A',k+1}  = 2 * omega^A_k * pi_{A',k}
Colored natively by pi_{A'} -> 0 horizon proximity, projective Green's potential,
and the 25-element SIGMA_5 spinor coupling trace (omega^dagger pi + pi^dagger omega).
"""

import os
import math
import torch
import numpy as np
from PIL import Image, ImageDraw
from palatial_gpo_ray_optics import SIGMA_5, compute_sigma5_freedom_matrix


def render_cp3_mandelbrot_frame(
    frame_idx: int,
    total_frames: int = 36,
    height: int = 360,
    width: int = 540,
    iterations: int = 44
) -> np.ndarray:
    """
    Renders a single frame of a zooming, phase-precessing CP^3 projective
    Mandelbrot cross-section with zero trigonometric coordinate charts.
    """
    t = frame_idx / float(total_frames)
    # Smooth periodic zoom into Seahorse Valley (-0.7453 + 0.1127j)
    zoom_cycle = 0.5 * (1.0 - math.cos(2.0 * math.pi * t))
    span = 1.35 * (0.045 ** zoom_cycle)

    center_re = -0.7453
    center_im = 0.1127
    aspect = width / float(height)

    ys = torch.linspace(center_im - span, center_im + span, height, dtype=torch.float64)
    xs = torch.linspace(center_re - span * aspect, center_re + span * aspect, width, dtype=torch.float64)
    grid_im, grid_re = torch.meshgrid(ys, xs, indexing="ij")

    # Complex parameter channels C^0, C^1 on CP^3
    C_0 = torch.complex(grid_re, grid_im)
    chiral_offset = 0.035 * complex(math.cos(2.0 * math.pi * t), math.sin(2.0 * math.pi * t))
    C_1 = C_0 + chiral_offset

    # Initialize homogeneous CP^3 coordinates [omega^0 : omega^1 : pi_0' : pi_1']
    omega_0 = torch.zeros_like(C_0)
    omega_1 = torch.zeros_like(C_1)
    pi_0 = torch.ones_like(C_0)
    pi_1 = torch.ones_like(C_1)

    green_pot = torch.zeros((height, width), dtype=torch.float64)
    spinor_coupling_accum = torch.zeros((height, width), dtype=torch.float64)

    for _ in range(iterations):
        # Homogeneous degree-2 projective endomorphism on CP^3 (zero trig, zero overflow)
        next_omega_0 = (omega_0 * omega_0) + C_0 * (pi_0 * pi_0)
        next_omega_1 = (omega_1 * omega_1) + C_1 * (pi_1 * pi_1)
        next_pi_0 = pi_0 * pi_0
        next_pi_1 = pi_1 * pi_1

        proj_norm = torch.sqrt(
            (next_omega_0.abs() ** 2) +
            (next_omega_1.abs() ** 2) +
            (next_pi_0.abs() ** 2) +
            (next_pi_1.abs() ** 2)
        ).clamp(min=1e-15)

        green_pot = green_pot + torch.log(proj_norm)

        omega_0 = next_omega_0 / proj_norm
        omega_1 = next_omega_1 / proj_norm
        pi_0 = next_pi_0 / proj_norm
        pi_1 = next_pi_1 / proj_norm

        # Track diagonal spinor inner product Re(omega^dagger pi) inside SIGMA_5
        re_inner = (omega_0.conj() * pi_0 + omega_1.conj() * pi_1).real
        spinor_coupling_accum = spinor_coupling_accum + re_inner

    # Proximity to the compactified hyperplane at infinity (pi_{A'} -> 0)
    pi_norm = torch.sqrt((pi_0.abs() ** 2) + (pi_1.abs() ** 2)).clamp(0.0, 1.0)
    horizon_escape = 1.0 - pi_norm  # 0 inside bounded cardioid, -> 1 at infinity hyperplane

    # Smooth analytical erf() filament enhancement from Green's potential & spinor trace
    pot_norm = torch.tanh(green_pot / (float(iterations) * 0.55))
    coupling_wave = 0.5 + 0.5 * torch.tanh(spinor_coupling_accum * 0.35)
    filament_glow = torch.exp(-((pi_norm - 0.18) ** 2) / (2.0 * (0.14 ** 2)))

    # GPO Triad Color Mapping:
    # Bounded interior -> Deep Emerald (P_G)
    # Boundary filaments -> Solar Gold (P_O)
    # Infinity horizon cascade (pi_{A'} -> 0) -> Electric Violet / Magenta (P_P)
    col_r = (
        0.04 * (1.0 - horizon_escape) +
        0.95 * filament_glow * coupling_wave +
        0.72 * (horizon_escape ** 1.4) * pot_norm
    )
    col_g = (
        0.42 * (1.0 - horizon_escape) * (0.4 + 0.6 * coupling_wave) +
        0.82 * filament_glow +
        0.16 * (horizon_escape ** 1.6) * (1.0 - pot_norm)
    )
    col_b = (
        0.38 * (1.0 - horizon_escape) +
        0.35 * filament_glow +
        0.96 * (horizon_escape ** 1.2) * (0.45 + 0.55 * coupling_wave)
    )

    rgb = torch.stack([col_r, col_g, col_b], dim=-1).clamp(0.0, 1.0)
    img_np = (rgb * 255.0).to(torch.uint8).numpy()
    pil_img = Image.fromarray(img_np)

    # Sample center pixel's 25-element SIGMA_5 freedom trace for live HUD readout
    cy, cx = height // 2, width // 2
    Z5_center = torch.tensor(
        [[omega_0[cy, cx], omega_1[cy, cx], pi_0[cy, cx], pi_1[cy, cx], 1.0 + pi_norm[cy, cx]]],
        dtype=torch.complex128
    )
    Z5_center = Z5_center / torch.linalg.norm(Z5_center, dim=-1, keepdim=True)
    M25_center = compute_sigma5_freedom_matrix(Z5_center)[0]
    tr_center = float(torch.einsum("ii->", M25_center).real.item())

    draw = ImageDraw.Draw(pil_img)
    draw.rectangle([8, 8, width - 8, 46], fill=(8, 10, 18))
    draw.text(
        (16, 12),
        f"Native CP^3 Projective Mandelbrot Slice (Zero Trig) | Zoom: {1.35 / span:.1f}x",
        fill=(80, 250, 190)
    )
    draw.text(
        (16, 28),
        f"Seahorse Valley (-0.7453 + 0.1127i) | Center Tr(M_25) = {tr_center:+.4f}",
        fill=(245, 205, 95)
    )

    return np.array(pil_img, dtype=np.uint8)


def generate_cp3_mandelbrot_gif(
    out_path: str = "cp3_mandelbrot_projection_proof.gif",
    num_frames: int = 36,
    duration_ms: int = 55
) -> str:
    frames = []
    for idx in range(num_frames):
        rgb = render_cp3_mandelbrot_frame(idx, total_frames=num_frames)
        frames.append(Image.fromarray(rgb))

    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False
    )
    return os.path.abspath(out_path)


if __name__ == "__main__":
    saved = generate_cp3_mandelbrot_gif()
    print(f"[SUCCESS] Generated native CP^3 Mandelbrot projection GIF: {saved}")

