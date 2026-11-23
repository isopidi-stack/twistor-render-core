"""
Reproducible regression harness for Bridge Lemma 3 (conditioning, cancellation,
sheet-continuation margin, and equatorial chart-handoff continuity) on CPU
float64 / complex128 benchmarked against an 80-digit Decimal exact-float64 reference.
"""

from __future__ import annotations

from decimal import Decimal, getcontext
import math
import pathlib
import sys
import unittest
import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from twistor import congruence as cg

getcontext().prec = 80

# Recorded reference values from the reported CPU float64/complex128 stress run
# (Python 3.13.1, PyTorch 2.15.0+cpu, commit 2c2e627).
RECORDED_CPU_REFERENCE = {
    "k2_sweep_growth_factor": 102.47,
    "rho_grad_sweep_growth_factor": 1.023e6,
    "worst_case_delta_rel_err": 3.327e-11,
    "disk_crossing_min_mu": 1.3515,
    "equatorial_max_norm_err": 3.33e-16,
    "equatorial_phase_cov_err": 2.71e-16,
}


def _delta_exact_decimal(x_f64: float, y_f64: float, z_f64: float, b_f64: float) -> tuple[Decimal, Decimal]:
    x = Decimal.from_float(float(x_f64))
    y = Decimal.from_float(float(y_f64))
    z = Decimal.from_float(float(z_f64))
    b = Decimal.from_float(float(b_f64))
    return (x * x + y * y + z * z - b * b), (Decimal(2) * b * z)


def _dec_complex_rel_err(comp: complex, ref_re: Decimal, ref_im: Decimal) -> Decimal:
    c_re = Decimal.from_float(float(comp.real))
    c_im = Decimal.from_float(float(comp.imag))
    num = ((c_re - ref_re) ** 2 + (c_im - ref_im) ** 2).sqrt()
    den = (ref_re * ref_re + ref_im * ref_im).sqrt()
    return num if den == 0 else num / den


class TestBridgeLemma3Numerics(unittest.TestCase):
    def test_1_conditioning_sweep_k_and_rho(self):
        b = 1.0
        dr_values = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5)
        k2_exact_list = []
        rho_exact_list = []

        for dr in dr_values:
            pos = torch.tensor([1.0 + dr, 0.0, 0.0], dtype=torch.float64)
            w = cg.principal_w(pos, b, sheet_sign=+1.0)
            w_abs = float(torch.abs(w).item())
            M = float((pos[0] ** 2 + pos[1] ** 2 + pos[2] ** 2 + b * b).item())

            k2_exact = 1.0 / w_abs
            rho_exact = math.sqrt(M) / (w_abs ** 3)
            k2_exact_list.append(k2_exact)
            rho_exact_list.append(rho_exact)

            h_k = min(1e-6, 0.02 * dr)
            h_rho = min(1e-7, 0.01 * (w_abs ** 2))

            cols_k, cols_re, cols_im = [], [], []
            for ax in range(3):
                step_k = torch.zeros(3, dtype=torch.float64)
                step_k[ax] = h_k
                kp = cg.tangent_field(pos + step_k, cg.continue_sheet_step(pos + step_k, w, b), b)
                km = cg.tangent_field(pos - step_k, cg.continue_sheet_step(pos - step_k, w, b), b)
                cols_k.append((kp - km) / (2.0 * h_k))

                step_r = torch.zeros(3, dtype=torch.float64)
                step_r[ax] = h_rho
                rp = complex(cg.optical_scalar(cg.continue_sheet_step(pos + step_r, w, b)).item())
                rm = complex(cg.optical_scalar(cg.continue_sheet_step(pos - step_r, w, b)).item())
                drho = (rp - rm) / (2.0 * h_rho)
                cols_re.append(drho.real)
                cols_im.append(drho.imag)

            k2_fd = float(torch.linalg.svdvals(torch.stack(cols_k, dim=-1))[0].item())
            rho_fd = float(torch.linalg.norm(torch.tensor([cols_re, cols_im], dtype=torch.float64), ord="fro").item())

            self.assertLess(abs(k2_fd - k2_exact) / k2_exact, 1e-3)
            self.assertLess(abs(rho_fd - rho_exact) / rho_exact, 1e-6)

        k2_growth = k2_exact_list[-1] / k2_exact_list[0]
        rho_growth = rho_exact_list[-1] / rho_exact_list[0]
        self.assertAlmostEqual(k2_growth, RECORDED_CPU_REFERENCE["k2_sweep_growth_factor"], places=1)
        self.assertAlmostEqual(rho_growth / 1e6, RECORDED_CPU_REFERENCE["rho_grad_sweep_growth_factor"] / 1e6, places=2)

    def test_2_cancellation_bound_against_exact_float64_oracle(self):
        b = 1.0
        sample_pts = [
            (1.0 + 1e-1, 0.0, 0.0),
            (1.0 + 1e-3, 0.0, 0.0),
            (1.0 + 1e-5, 0.0, 0.0),
            (0.8, 0.6, 1e-3),
            (0.8, 0.6, 1e-6),
        ]
        u_mach = Decimal.from_float(math.ulp(1.0)) / Decimal(2)
        worst_rel_err = 0.0

        for x_f, y_f, z_f in sample_pts:
            pos = torch.tensor([x_f, y_f, z_f], dtype=torch.float64)
            d_comp = complex(cg.delta(pos, b).item())
            ref_re, ref_im = _delta_exact_decimal(x_f, y_f, z_f, b)
            M_dec = (
                Decimal.from_float(x_f) ** 2
                + Decimal.from_float(y_f) ** 2
                + Decimal.from_float(z_f) ** 2
                + Decimal.from_float(b) ** 2
            )
            abs_delta_dec = (ref_re * ref_re + ref_im * ref_im).sqrt()
            rel_err = float(_dec_complex_rel_err(d_comp, ref_re, ref_im))
            cond_bound = float((Decimal(8) * u_mach * M_dec) / abs_delta_dec)

            worst_rel_err = max(worst_rel_err, rel_err)
            self.assertLess(rel_err, 1e-9)
            self.assertLessEqual(rel_err, cond_bound)

        self.assertLessEqual(worst_rel_err, RECORDED_CPU_REFERENCE["worst_case_delta_rel_err"] * 1.05)

    def test_3_sheet_continuation_margin_and_perturbations(self):
        b = 1.25
        x0 = torch.tensor([0.3, -0.4, 0.5], dtype=torch.float64)
        w0 = cg.principal_w(x0, b, sheet_sign=-1.0)
        k0 = cg.tangent_field(x0, w0, b)
        s_star = float((-w0.real).item())

        w_prev = w0
        min_mu = float("inf")
        steps = 200
        ds = (2.2 * s_star) / steps

        for idx in range(1, steps + 1):
            s = idx * ds
            xs = x0 + s * k0
            w_exact = w0 + s
            w_prev_exact = w0 + (idx - 1) * ds
            mu_exact = float((w_exact.real * w_prev_exact.real + w_exact.imag * w_prev_exact.imag).item())
            min_mu = min(min_mu, mu_exact)
            self.assertGreater(mu_exact, 0.0)

            w_sel = cg.continue_sheet_step(xs, w_prev, b)
            self.assertLess(float(torch.abs(w_sel - w_exact).item()), 1e-13)
            w_prev = w_sel

        self.assertGreaterEqual(min_mu, RECORDED_CPU_REFERENCE["disk_crossing_min_mu"] * 0.99)

        # Controlled perturbation test around pi/2 boundary
        xs_probe = x0 + 0.5 * s_star * k0
        w_true_probe = w0 + 0.5 * s_star
        w_ref_safe = (w_true_probe * complex(math.cos(0.49 * math.pi), math.sin(0.49 * math.pi))).detach().clone()
        w_ref_flip = (w_true_probe * complex(math.cos(0.51 * math.pi), math.sin(0.51 * math.pi))).detach().clone()

        sel_safe = cg.continue_sheet_step(xs_probe, w_ref_safe, b)
        sel_flip = cg.continue_sheet_step(xs_probe, w_ref_flip, b)
        self.assertLess(float(torch.abs(sel_safe - w_true_probe).item()), 1e-13)
        self.assertLess(float(torch.abs(sel_flip + w_true_probe).item()), 1e-13)

    def test_4_equatorial_handoff_and_optical_scalar_continuity(self):
        b = 1.25
        # Outside spanning disk: r^2 = 1.6^2 + (-1.2)^2 = 4.0 > 1.25^2
        zs = torch.linspace(-2.5, 2.5, 501, dtype=torch.float64)
        pts = torch.stack([torch.full_like(zs, 1.6), torch.full_like(zs, -1.2), zs], dim=-1)

        w_list = [cg.principal_w(pts[0], b, sheet_sign=+1.0)]
        for i in range(1, len(pts)):
            w_list.append(cg.continue_sheet_step(pts[i], w_list[-1], b))
        w_cont = torch.stack(w_list, dim=0)

        res = cg.bounded_chart_and_screen(pts, w_cont, b)
        k = res["k"]
        use_south = res["use_south"]
        switches = torch.where(use_south[:-1] != use_south[1:])[0]
        self.assertEqual(len(switches), 1)
        idx = int(switches[0].item())

        max_norm_err = float(torch.max(torch.abs(torch.linalg.norm(k, dim=-1) - 1.0)).item())
        self.assertLess(max_norm_err, 1e-14)
        self.assertLessEqual(max_norm_err, RECORDED_CPU_REFERENCE["equatorial_max_norm_err"] * 1.1)

        rho_vals = cg.optical_scalar(w_cont)
        rho_jump = float(torch.abs(rho_vals[idx + 1] - rho_vals[idx]).item())
        self.assertLess(rho_jump, 1e-2)

        kx_s, ky_s, kz_s = float(k[idx, 0].item()), float(k[idx, 1].item()), float(k[idx, 2].item())
        zeta_s = complex(kx_s, ky_s) / (1.0 + kz_s)
        eta_s = complex(kx_s, -ky_s) / (1.0 - kz_s)
        m_s_vec = cg.screen_vector_south(torch.tensor(zeta_s, dtype=torch.complex128))
        m_n_vec = cg.screen_vector_north(torch.tensor(eta_s, dtype=torch.complex128))
        phase_factor = zeta_s.conjugate() / zeta_s
        cov_err = float(torch.linalg.norm(m_n_vec - phase_factor * m_s_vec).item())

        self.assertLess(cov_err, 1e-14)
        self.assertLessEqual(cov_err, RECORDED_CPU_REFERENCE["equatorial_phase_cov_err"] * 1.1)


if __name__ == "__main__":
    unittest.main(verbosity=2)