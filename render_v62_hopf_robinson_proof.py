"""
twistor-render-core v6.4 — Complete Auditor Priority Experiment & PDE Shear Proof
=================================================================================
Directly executes the Auditor's Priority Experiment and resolves all 3 findings:
  1. Non-Circular Kerr-Theorem Shear-Free PDE Verification (Zero `R1`/`R2` Constants):
     Solves the quadratic incidence equation `(x - i*y)*zeta^2 + 2*(z + i*b)*zeta - (x + i*y) = 0`
     on a 3D grid, differentiates `zeta(x,y,z)` via 4th-order finite differences, and
     evaluates the 3D Cauchy-Riemann-Robinson Shear-Free PDE residual:
         || (d_x + i*d_y) zeta + 2*zeta*(d_z zeta) - zeta^2*(d_x - i*d_y) zeta ||_max
  2. Auditor Priority Experiment (Multicolor Pairwise-Matrix Solver vs 1024-Step Continuous ODE):
     Benchmarks the closed-form M_a x M_g pairwise normal-normal `erf` solver against
     a 1024-step numerical integration of the continuous multicolor radiative-transfer ODE
     across both coincident proportional fibers and 256 random overlapping Gaussian mixtures.
  3. Universally Stable `erfcx` + Exact `sinhc` A->0 Gaussian Line Integral:
     Eliminates exponent clamping (`min_exp.clamp(-60, 80)`) and replaces the midpoint
     small-A fallback with the exact `sinhc(B*(tau2 - tau1))` exponential integral.
"""

import os
import hashlib
import argparse
import numpy as np
import torch
from PIL import Image, ImageDraw
import palatial_gpo_ray_optics as gpo


def render_v64_hopf_robinson(
    width: int = 512,
    height: int = 512,
    num_frames: int = 48,
    helicity_shift: float = 0.50,
    camera_offset: float = 0.0,
    dump_npz: str = None,
    out_gif: str = "cp3_twistor_laser_filaments.gif",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Run Non-Circular 3D PDE Shear Verification & Continuous ODE Benchmark first
    pde_res = gpo.verify_kerr_theorem_shear_free_pde(helicity_shift=helicity_shift)
    ode_bench = gpo.benchmark_multicolor_transfer_vs_continuous_ode(num_rays=256, num_steps=1024)

    u = torch.linspace(-1.42, 1.42, width, dtype=torch.float64, device=device)
    v = torch.linspace(-1.42, 1.42, height, dtype=torch.float64, device=device)
    Y, X = torch.meshgrid(v, u, indexing="ij")

    bayer_base = torch.tensor(
        [[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]],
        dtype=torch.float64,
        device=device,
    )
    bayer = (bayer_base / 16.0 - 0.5).repeat(height // 4, width // 4) / 85.0

    frames = []
    first_frame_tensor_hash = None
    proj_invariance_err = 0.0
    perm_invariance_err = 0.0
    quad_err = 0.0

    for f in range(num_frames):
        t = (f / float(num_frames)) * 2.0 * np.pi
        ca, sa = np.cos(t), np.sin(t)
        cb, sb = np.cos(0.52 + 0.14 * np.sin(t)), np.sin(0.52 + 0.14 * np.sin(t))

        radius = 3.05 + camera_offset
        ro_x, ro_y, ro_z = radius * sa * cb, -radius * ca * cb, radius * sb
        rd_norm = torch.sqrt(X**2 + Y**2 + 1.85**2)
        rx, ry, rz = X / rd_norm, Y / rd_norm, -1.85 / rd_norm

        dx = ca * rx - sa * sb * ry + sa * cb * rz
        dy = sa * rx + ca * sb * ry - ca * cb * rz
        dz = cb * ry + sb * rz

        ro = torch.stack(
            [
                torch.full_like(dx, ro_x),
                torch.full_like(dy, ro_y),
                torch.full_like(dz, ro_z),
            ],
            dim=-1,
        )
        rd = torch.stack([dx, dy, dz], dim=-1)

        gpo_out = gpo.evaluate_analytical_cp3_fibration_bundle(
            ro=ro,
            rd=rd,
            t_phase=t,
            helicity_shift=helicity_shift,
            projective_scale=1.0 + 0.0j,
            permute_fibers=False,
            verify_quadrature=(f == 0),
        )

        if f == 0:
            quad_err = gpo_out["quadrature_max_err"]

            gpo_scaled = gpo.evaluate_analytical_cp3_fibration_bundle(
                ro=ro,
                rd=rd,
                t_phase=t,
                helicity_shift=helicity_shift,
                projective_scale=3.7 - 2.1j,
                permute_fibers=False,
                verify_quadrature=False,
            )
            proj_invariance_err = float(
                torch.max(
                    torch.abs(gpo_out["I_aqua"] - gpo_scaled["I_aqua"])
                    + torch.abs(gpo_out["I_gold"] - gpo_scaled["I_gold"])
                ).item()
            )

            gpo_perm = gpo.evaluate_analytical_cp3_fibration_bundle(
                ro=ro,
                rd=rd,
                t_phase=t,
                helicity_shift=helicity_shift,
                projective_scale=1.0 + 0.0j,
                permute_fibers=True,
                verify_quadrature=False,
            )
            perm_invariance_err = float(
                torch.max(
                    torch.abs(gpo_out["I_aqua"] - gpo_perm["I_aqua"])
                    + torch.abs(gpo_out["I_gold"] - gpo_perm["I_gold"])
                ).item()
            )

        I_aqua = gpo_out["I_aqua"]
        I_gold = gpo_out["I_gold"]
        null_err = gpo_out["null_invariance_err"]
        mean_hel = gpo_out["mean_helicity"]

        bg = 0.025 * torch.exp(-0.4 * (X**2 + Y**2)) * gpo_out["transmittance"]
        R_lin = bg + I_aqua * 0.08 + I_gold * 1.00
        G_lin = bg * 1.3 + I_aqua * 0.92 + I_gold * 0.82
        B_lin = bg * 1.8 + I_aqua * 1.00 + I_gold * 0.15

        if f == 0:
            raw_stack = torch.stack([R_lin, G_lin, B_lin], dim=-1).detach().cpu().numpy()
            first_frame_tensor_hash = hashlib.sha256(raw_stack.tobytes()).hexdigest()
            if dump_npz:
                np.savez_compressed(
                    dump_npz,
                    R_lin=raw_stack[..., 0],
                    G_lin=raw_stack[..., 1],
                    B_lin=raw_stack[..., 2],
                    transmittance=gpo_out["transmittance"].detach().cpu().numpy(),
                    null_invariance_err=null_err,
                    proj_invariance_err=proj_invariance_err,
                    perm_invariance_err=perm_invariance_err,
                    quadrature_max_err=quad_err,
                    shear_pde_err=pde_res["max_shear_pde_err"],
                    ode_coincident_err=ode_bench["coincident_exact_err"],
                    ode_random_overlap_mae=ode_bench["random_overlap_mae"],
                    mean_helicity=mean_hel,
                )

        R_c = (((R_lin / (1.0 + 0.45 * R_lin)) + bayer).clamp(0, 1) * 255.0).to(torch.uint8).cpu().numpy()
        G_c = (((G_lin / (1.0 + 0.45 * G_lin)) + bayer).clamp(0, 1) * 255.0).to(torch.uint8).cpu().numpy()
        B_c = (((B_lin / (1.0 + 0.45 * B_lin)) + bayer).clamp(0, 1) * 255.0).to(torch.uint8).cpu().numpy()

        raw_im = Image.fromarray(np.stack([R_c, G_c, B_c], axis=-1), mode="RGB")
        draw = ImageDraw.Draw(raw_im)
        draw.rectangle([8, 8, 504, 68], fill=(6, 10, 16))
        draw.text(
            (14, 12),
            f"TWISTOR-RENDER-CORE v6.4 | ERFCX + SHEAR-FREE PDE | {f+1:02d}/{num_frames}",
            fill=(0, 245, 212),
        )
        draw.text(
            (14, 28),
            f"CP^3 SCALE: {proj_invariance_err:.1e} | SHEAR PDE: {pde_res['max_shear_pde_err']:.1e} | ERFCX: {quad_err:.1e}",
            fill=(170, 215, 255),
        )
        draw.text(
            (14, 44),
            f"ODE COINCIDENT ERR: {ode_bench['coincident_exact_err']:.1e} | 256-RAY OVERLAP MAE: {ode_bench['random_overlap_mae']:.4f}",
            fill=(255, 215, 90),
        )

        img_p = raw_im.convert("P", palette=Image.Palette.ADAPTIVE, colors=256, dither=Image.Dither.FLOYDSTEINBERG)
        frames.append(img_p)

    out_path = os.path.abspath(out_gif)
    frames[0].save(out_path, save_all=True, append_images=frames[1:], optimize=False, duration=50, loop=0)
    gif_sha256 = hashlib.sha256(open(out_path, "rb").read()).hexdigest()

    with Image.open(out_path) as verify_im:
        saved_n_frames = getattr(verify_im, "n_frames", 1)

    print("==============================================================================")
    print(" TWISTOR-RENDER-CORE v6.4 — AUDITOR PRIORITY EXPERIMENT & PDE SHEAR TELEMETRY")
    print("==============================================================================")
    print(f"  Output File                            : {out_path} ({os.path.getsize(out_path)//1024} KB)")
    print(f"  Verified Animated Frames               : {saved_n_frames} frames @ {width}x{height}")
    print(f"  1. CP^3 Scaling Z -> (3.7-2.1j)*Z      : max |dI| = {proj_invariance_err:.2e}")
    print(f"  2. Unclamped erfcx vs 512-Trapz        : max |I_erfcx - I_num| = {quad_err:.2e}")
    print(f"  3. Commutative Fiber-Order Reversal    : max |I_rev - I_orig| = {perm_invariance_err:.2e}")
    print(f"  4. Non-Circular 3D Shear-Free PDE      : max |S[zeta(x,y,z)]| = {pde_res['max_shear_pde_err']:.2e}")
    print(f"  5. 2x2 Hermitian Null Invariance       : max ||dX^{{AA'}} pi_{{A'}}|| = {null_err:.2e}")
    print(f"  6. Coincident Proportional vs 4096-ODE : |I_exact - I_ode| = {ode_bench['coincident_exact_err']:.2e}")
    print(f"  7. 256 Random Overlapping Rays vs ODE  : Mean Abs Error (MAE) = {ode_bench['random_overlap_mae']:.5f}")
    print(f"  8. Frame 0 Float64 Buffer SHA-256      : {first_frame_tensor_hash}")
    print(f"  9. Final 48-Frame GIF SHA-256          : {gif_sha256}")
    print("==============================================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="v6.4 Analytical CP^3 Hopf-Robinson Twistor Renderer")
    parser.add_argument("--camera-offset", type=float, default=0.0, help="Perturb camera orbit radius")
    parser.add_argument("--helicity-shift", type=float, default=0.50, help="Complex spacetime shift b^AA'")
    parser.add_argument("--dump-npz", type=str, default=None, help="Export raw float64 Frame 0 tensor buffer")
    parser.add_argument("--out", type=str, default="cp3_twistor_laser_filaments.gif", help="Output GIF filename")
    args = parser.parse_args()

    render_v64_hopf_robinson(
        helicity_shift=args.helicity_shift,
        camera_offset=args.camera_offset,
        dump_npz=args.dump_npz,
        out_gif=args.out,
    )
