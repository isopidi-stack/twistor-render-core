"""
Project Argon v6.1 - True CP^4 Spinor Incidence & erf() Radiative Transfer Renderer
Directly invokes render_gpo_hud_frame() from palatial_gpo_ray_optics.py to synthesize
36 frames of a rotating 3D volumetric twistor lattice with zero Z-buffer.
"""

import os
import sys
import math
import torch

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from palatial_gpo_ray_optics import render_gpo_hud_frame


def generate_spinor_viewport(height: int, width: int, angle: float, device: str = "cpu"):
    """
    Constructs an orbiting camera origin X_cam and a 2-spinor ray bundle pi_{A'} across the screen.
    """
    y_lin = torch.linspace(-1.15, 1.15, height, dtype=torch.float64, device=device)
    x_lin = torch.linspace(-1.15, 1.15, width, dtype=torch.float64, device=device)
    vy, vx = torch.meshgrid(y_lin, x_lin, indexing="ij")

    # Orbit camera in X-Z plane looking toward the origin
    radius = 4.2
    cam_x = radius * math.sin(angle)
    cam_y = 0.85 * math.sin(angle * 2.0)
    cam_z = -radius * math.cos(angle)
    cam_origin = torch.tensor([cam_x, cam_y, cam_z], dtype=torch.float64, device=device)
    cam_origin_grid = cam_origin.view(1, 1, 3).expand(height, width, 3)

    # Forward, Right, Up basis for camera orientation toward (0,0,0)
    fwd = -cam_origin / torch.linalg.norm(cam_origin)
    world_up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64, device=device)
    right = torch.linalg.cross(world_up, fwd)
    right = right / torch.linalg.norm(right)
    up = torch.linalg.cross(fwd, right)

    # 3D unit ray direction n(y, x)
    ray_dir = (
        fwd.view(1, 1, 3)
        + vx.unsqueeze(-1) * right.view(1, 1, 3) * 0.55
        - vy.unsqueeze(-1) * up.view(1, 1, 3) * 0.55
    )
    ray_dir = ray_dir / torch.linalg.norm(ray_dir, dim=-1, keepdim=True).clamp(min=1e-12)

    nx = ray_dir[..., 0]
    ny = ray_dir[..., 1]
    nz = ray_dir[..., 2]

    # Exact inverse Bloch map: unit vector (nx, ny, nz) -> normalized 2-spinor pi_{A'}
    # cos(theta/2) = sqrt((1 + nz)/2), sin(theta/2)*exp(i*phi) = (nx + i*ny)/sqrt(2*(1 + nz))
    denom = torch.sqrt((2.0 * (1.0 + nz)).clamp(min=1e-9))
    pi_0 = 0.5 * denom.to(torch.complex128)
    pi_1 = (nx.to(torch.complex128) + 1j * ny.to(torch.complex128)) / denom.to(torch.complex128)
    pi_spinor = torch.stack([pi_0, pi_1], dim=-1)

    return cam_origin_grid, pi_spinor, vx, vy


def main():
    out_dir = sys.argv[1]
    num_frames = 36
    height, width = 256, 256
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  [*] Executing v6.1 CP^4 Spinor Ray-Optics on device: {device.upper()} ({width}x{height}, {num_frames} frames)")

    with torch.no_grad():
        for frame_idx in range(num_frames):
            phase = (frame_idx / float(num_frames)) * (2.0 * math.pi)
            cam_origin_grid, pi_spinor, vx, vy = generate_spinor_viewport(height, width, phase, device=device)

            # Construct 9 dynamic 3D Gaussian twistor primitives (central core + 2 counter-rotating rings)
            centers = [[0.0, 0.0, 0.0]]
            opacities = [0.96]
            for k in range(8):
                theta_k = (k / 8.0) * (2.0 * math.pi) + (0.5 * phase if k % 2 == 0 else -0.5 * phase)
                r_ring = 1.15 if k % 2 == 0 else 0.78
                cx = r_ring * math.cos(theta_k)
                cy = 0.45 * math.sin(2.0 * theta_k + phase)
                cz = r_ring * math.sin(theta_k)
                centers.append([cx, cy, cz])
                opacities.append(0.88 if k % 2 == 0 else 0.75)

            field_centers = torch.tensor(centers, dtype=torch.float64, device=device)
            field_opacities = torch.tensor(opacities, dtype=torch.float64, device=device)

            # Spatially varying mass-shell / central charge coordinate phi in CP^4
            radial_screen = torch.sqrt(vx ** 2 + vy ** 2)
            phi_mass = (0.55 * torch.exp(1j * (radial_screen * 2.2 - phase))).unsqueeze(-1)

            # Flatten pixel batch (H*W) for vectorized v6.1 evaluation
            N_pix = height * width
            res = render_gpo_hud_frame(
                cam_origin=cam_origin_grid.reshape(N_pix, 3),
                pi_spinor=pi_spinor.reshape(N_pix, 2),
                field_centers=field_centers,
                field_opacities=field_opacities,
                phi_mass=phi_mass.reshape(N_pix, 1),
                sigma_width=0.38,
                L0_scale=1.5
            )

            # Verify strict Penrose incidence defect == 0 on every rendered frame
            max_defect = float(res["incidence_defect"].max().item())
            if max_defect > 1e-10:
                raise RuntimeError(f"Penrose incidence defect exceeded tolerance: {max_defect}")

            # Extract GPO triad probabilities (c_G, c_P, c_O), field intensity, and H_GPO expectation
            gpo_hud = res["gpo_hud_probs"].reshape(height, width, 3)
            gpo_phys = res["gpo_physical_probs"].reshape(height, width, 3)
            intensity = res["field_intensity"].reshape(height, width)
            casimir_exp = res["gpo_spectral_expectation"].reshape(height, width) / 48.0

            c_G = gpo_hud[..., 0]
            c_P = gpo_hud[..., 1]
            c_O = gpo_hud[..., 2]

            # Map exact GPO triad channels + H_GPO spectral expectation to RGB
            # Green channel -> Emerald/Cyan, Purple channel -> Violet/Magenta, Orange channel -> Amber/Gold
            lum = (0.18 + 0.82 * intensity.clamp(0.0, 1.0))
            shimmer = 0.5 + 0.5 * torch.cos(casimir_exp * math.pi * 4.0 - phase)

            R = lum * (0.15 * c_G + 0.72 * c_P + 1.00 * c_O) + 0.18 * intensity * c_O
            G = lum * (0.95 * c_G + 0.18 * c_P + 0.58 * c_O) + 0.12 * shimmer * intensity
            B = lum * (0.55 * c_G + 0.96 * c_P + 0.12 * c_O) + 0.15 * gpo_phys[..., 1]

            rgb = torch.stack([R, G, B], dim=-1).clamp(0.0, 1.0)
            rgb_bytes = (rgb * 255.0).to(torch.uint8).cpu().numpy().tobytes()

            ppm_path = os.path.join(out_dir, f"frame_{frame_idx:03d}.ppm")
            with open(ppm_path, "wb") as f:
                f.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
                f.write(rgb_bytes)

            sys.stdout.write(f"\r  [+] Rendered Spinor Frame {frame_idx + 1:02d}/{num_frames} (Max Incidence Defect: {max_defect:.1e})")
            sys.stdout.flush()
    print("\n  [OK] All 36 CP^4 spinor frames rendered with 0.0 incidence defect.")


if __name__ == "__main__":
    main()
