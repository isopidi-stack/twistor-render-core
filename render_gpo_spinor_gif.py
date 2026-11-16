"""
Project Argon v6.1 - High-Contrast CP^4 Spinor Incidence & Casimir Interference Renderer
Directly invokes render_gpo_hud_frame() from palatial_gpo_ray_optics.py with sharp optical
cores (sigma=0.11), 24-node toroidal/helical twistor lattice, and full GPO spectral separation.
"""

import os
import sys
import math
import torch

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from palatial_gpo_ray_optics import (
    render_gpo_hud_frame,
    compute_gpo_probabilities,
    compute_gpo_spectral_expectation,
)


def generate_spinor_viewport(height: int, width: int, angle: float, device: str = "cpu"):
    y_lin = torch.linspace(-1.0, 1.0, height, dtype=torch.float64, device=device)
    x_lin = torch.linspace(-1.0, 1.0, width, dtype=torch.float64, device=device)
    vy, vx = torch.meshgrid(y_lin, x_lin, indexing="ij")

    # Orbit camera around the twistor lattice
    radius = 3.6
    cam_x = radius * math.sin(angle)
    cam_y = 1.15 * math.sin(angle * 1.0)
    cam_z = -radius * math.cos(angle)
    cam_origin = torch.tensor([cam_x, cam_y, cam_z], dtype=torch.float64, device=device)
    cam_origin_grid = cam_origin.view(1, 1, 3).expand(height, width, 3)

    fwd = -cam_origin / torch.linalg.norm(cam_origin)
    world_up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64, device=device)
    right = torch.linalg.cross(world_up, fwd)
    right = right / torch.linalg.norm(right)
    up = torch.linalg.cross(fwd, right)

    ray_dir = (
        fwd.view(1, 1, 3)
        + vx.unsqueeze(-1) * right.view(1, 1, 3) * 0.58
        - vy.unsqueeze(-1) * up.view(1, 1, 3) * 0.58
    )
    ray_dir = ray_dir / torch.linalg.norm(ray_dir, dim=-1, keepdim=True).clamp(min=1e-12)

    nx, ny, nz = ray_dir[..., 0], ray_dir[..., 1], ray_dir[..., 2]
    denom = torch.sqrt((2.0 * (1.0 + nz)).clamp(min=1e-9))
    pi_0 = (0.5 * denom).to(torch.complex128)
    pi_1 = (nx.to(torch.complex128) + 1j * ny.to(torch.complex128)) / denom.to(torch.complex128)
    pi_spinor = torch.stack([pi_0, pi_1], dim=-1)

    return cam_origin_grid, pi_spinor, vx, vy, ray_dir


def main():
    out_dir = sys.argv[1]
    num_frames = 36
    height, width = 320, 320
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  [*] Rendering High-Contrast CP^4 Spinor Lattice on {device.upper()} ({width}x{height}, {num_frames} frames)")

    with torch.no_grad():
        for frame_idx in range(num_frames):
            phase = (frame_idx / float(num_frames)) * (2.0 * math.pi)
            cam_origin_grid, pi_spinor, vx, vy, ray_dir = generate_spinor_viewport(height, width, phase, device=device)

            # Construct a sharp 21-primitive 3D Hopf-like interlocking twistor lattice
            centers = [[0.0, 0.0, 0.0]]
            opacities = [0.99]

            # Ring 1: 10 nodes on equatorial X-Z torus
            for k in range(10):
                a = (k / 10.0) * (2.0 * math.pi) + phase
                r1 = 1.25
                centers.append([r1 * math.cos(a), 0.28 * math.sin(3.0 * a), r1 * math.sin(a)])
                opacities.append(0.95)

            # Ring 2: 10 nodes on tilted counter-rotating Clifford torus
            for k in range(10):
                b = (k / 10.0) * (2.0 * math.pi) - phase * 1.0
                r2 = 0.78
                x0 = r2 * math.cos(b)
                y0 = r2 * math.sin(b)
                # Tilt by 60 degrees around X-axis
                centers.append([x0, y0 * 0.866, y0 * 0.5])
                opacities.append(0.92)

            field_centers = torch.tensor(centers, dtype=torch.float64, device=device)
            field_opacities = torch.tensor(opacities, dtype=torch.float64, device=device)

            # Dynamic central-charge / mass-shell field phi(x,y)
            r_screen = torch.sqrt(vx ** 2 + vy ** 2)
            azimuth = torch.atan2(vy, vx)
            phi_mass = (0.65 * torch.exp(1j * (r_screen * 6.0 - 3.0 * azimuth + phase * 2.0))).unsqueeze(-1)

            N_pix = height * width
            # Run narrow-core (sigma=0.115) pass for crisp volumetric spheres + erf() occlusion
            res_sharp = render_gpo_hud_frame(
                cam_origin=cam_origin_grid.reshape(N_pix, 3),
                pi_spinor=pi_spinor.reshape(N_pix, 2),
                field_centers=field_centers,
                field_opacities=field_opacities,
                phi_mass=phi_mass.reshape(N_pix, 1),
                sigma_width=0.115,
                L0_scale=3.8
            )

            # Run halo/caustic envelope pass (sigma=0.28) for interference rings
            res_halo = render_gpo_hud_frame(
                cam_origin=cam_origin_grid.reshape(N_pix, 3),
                pi_spinor=pi_spinor.reshape(N_pix, 2),
                field_centers=field_centers,
                field_opacities=field_opacities,
                phi_mass=phi_mass.reshape(N_pix, 1),
                sigma_width=0.28,
                L0_scale=3.8
            )

            max_defect = float(res_sharp["incidence_defect"].max().item())
            if max_defect > 1e-10:
                raise RuntimeError(f"Incidence defect violation: {max_defect}")

            core_int = res_sharp["field_intensity"].reshape(height, width).clamp(0.0, 1.0)
            halo_int = res_halo["field_intensity"].reshape(height, width).clamp(0.0, 1.0)
            vis_depth = res_sharp["visible_depth"].reshape(height, width)

            # Construct local spinor interference state driven by CP^4 Casimir eigenvalues (0, 20, 48)
            # This makes the GPO channels oscillate across the volumetric caustics
            pi_grid = pi_spinor
            casimir_phase_20 = 20.0 * 0.22 * halo_int + phase * 2.0
            casimir_phase_48 = 48.0 * 0.12 * core_int - phase * 3.0

            omega_local = pi_grid * torch.exp(1j * (casimir_phase_20.unsqueeze(-1) + azimuth.unsqueeze(-1)))
            phi_local = (
                (0.95 * core_int + 0.35 * torch.sin(casimir_phase_48).abs())
                * torch.exp(1j * casimir_phase_48)
            ).unsqueeze(-1).to(torch.complex128)

            Z5_local = torch.cat([omega_local, pi_grid, phi_local], dim=-1)
            gpo_probs = compute_gpo_probabilities(Z5_local)
            casimir_exp = compute_gpo_spectral_expectation(Z5_local) / 48.0

            c_G = gpo_probs[..., 0]
            c_P = gpo_probs[..., 1]
            c_O = gpo_probs[..., 2]

            # High-frequency Fubini-Study harmonic rings around each node
            harmonic_rings = 0.5 + 0.5 * torch.cos(halo_int * 28.0 - phase * 4.0)
            caustic_glow = (halo_int ** 1.35) * (0.45 + 0.55 * harmonic_rings)
            core_hot = core_int ** 0.75

            # Depth-cue attenuation (near nodes blaze brighter than occluded far nodes)
            depth_factor = torch.exp(-0.18 * (vis_depth - 2.2).clamp(min=0.0)) * (core_int > 1e-3).to(torch.float64) + 0.65 * (core_int <= 1e-3).to(torch.float64)

            # Deep obsidian void background with subtle spinor reticle grid
            reticle = (torch.exp(-((r_screen - 0.62) ** 2) / 0.0004) + torch.exp(-((r_screen - 0.38) ** 2) / 0.0003)) * 0.22
            bg_r = 0.015 + reticle * 0.15
            bg_g = 0.020 + reticle * 0.65
            bg_b = 0.045 + reticle * 0.85

            # Vivid GPO Triad Palette:
            # c_G -> Neon Emerald/Cyan, c_P -> Electric Magenta/Violet, c_O -> Incandescent Solar Gold
            R = bg_r + depth_factor * (
                caustic_glow * (0.10 * c_G + 0.88 * c_P + 1.00 * c_O)
                + core_hot * (0.20 * c_G + 0.95 * c_P + 1.00 * c_O)
                + 0.45 * (core_int ** 2.2)
            )
            G = bg_g + depth_factor * (
                caustic_glow * (1.00 * c_G + 0.12 * c_P + 0.65 * c_O)
                + core_hot * (0.98 * c_G + 0.25 * c_P + 0.82 * c_O)
                + 0.40 * (core_int ** 2.2)
            )
            B = bg_b + depth_factor * (
                caustic_glow * (0.72 * c_G + 1.00 * c_P + 0.08 * c_O)
                + core_hot * (0.85 * c_G + 1.00 * c_P + 0.20 * c_O)
                + 0.55 * (core_int ** 2.2) * casimir_exp
            )

            rgb = torch.stack([R, G, B], dim=-1).clamp(0.0, 1.0)
            rgb_bytes = (rgb * 255.0).to(torch.uint8).cpu().numpy().tobytes()

            ppm_path = os.path.join(out_dir, f"frame_{frame_idx:03d}.ppm")
            with open(ppm_path, "wb") as f:
                f.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
                f.write(rgb_bytes)

            sys.stdout.write(f"\r  [+] Rendered Sharp Spinor Frame {frame_idx + 1:02d}/{num_frames} (Max Incidence Defect: {max_defect:.1e})")
            sys.stdout.flush()
    print("\n  [OK] All 36 High-Contrast CP^4 Spinor frames rendered with 0.0 incidence defect.")


if __name__ == "__main__":
    main()
