import math
import unittest
import torch

from twistor import congruence as cg


def _diff_4th(fn, pos: torch.Tensor, axis: int, h: float) -> torch.Tensor:
    step = torch.zeros_like(pos)
    step[..., axis] = h
    return (-fn(pos + 2.0 * step) + 8.0 * fn(pos + step) - 8.0 * fn(pos - step) + fn(pos - 2.0 * step)) / (12.0 * h)


def _right_handed_screen_basis(k0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    ref = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64) if abs(float(k0[0])) < 0.8 else torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    e1 = torch.linalg.cross(k0, ref)
    e1 = e1 / torch.linalg.norm(e1)
    e2 = torch.linalg.cross(k0, e1)
    e2 = e2 / torch.linalg.norm(e2)
    return e1, e2


class TestCongruenceReference(unittest.TestCase):
    def test_1_unit_norm_null_lift_and_regularity(self):
        b = 1.25
        pts = torch.tensor([
            [1.4, -0.7, 0.9],      # general point
            [0.0, 0.0, 0.6],       # z-axis (+z)
            [0.0, 0.0, -0.6],      # z-axis (-z)
            [0.3, -0.4, 0.0],      # lifted spanning disk interior (alpha = 0)
            [0.0, 0.0, 0.0],       # disk center (alpha = 0, x = y = 0)
        ], dtype=torch.float64)
        for sign in (+1.0, -1.0):
            w = cg.principal_w(pts, b, sheet_sign=sign)
            k = cg.tangent_field(pts, w, b)
            K = cg.minkowski_null_lift(k)
            self.assertTrue(torch.isfinite(k).all().item())
            norms = torch.linalg.norm(k, dim=-1)
            self.assertLess(float(torch.max(torch.abs(norms - 1.0)).item()), 1e-14)
            # Minkowski null condition K^mu K_mu = K0^2 - |k|^2 = 0
            g = torch.tensor([1.0, -1.0, -1.0, -1.0], dtype=torch.float64)
            null_norm = torch.sum((K * K) * g, dim=-1)
            self.assertLess(float(torch.max(torch.abs(null_norm)).item()), 1e-14)

    def test_2_geodesicity_optical_scalars_and_frobenius_norm(self):
        b = 0.85
        h = 1e-5
        test_pts = [
            torch.tensor([0.55, 0.45, 0.50], dtype=torch.float64),
            torch.tensor([1.20, -0.65, -0.35], dtype=torch.float64),
            torch.tensor([0.30, -0.40, 0.00], dtype=torch.float64),  # alpha = 0 disk interior
        ]
        for pos in test_pts:
            for sign in (+1.0, -1.0):
                w0 = cg.principal_w(pos, b, sheet_sign=sign)
                k0 = cg.tangent_field(pos, w0, b)

                def eval_k_local(p: torch.Tensor) -> torch.Tensor:
                    w_loc = cg.continue_sheet_step(p, w0, b)
                    return cg.tangent_field(p, w_loc, b)

                dk_dx = _diff_4th(eval_k_local, pos, 0, h)
                dk_dy = _diff_4th(eval_k_local, pos, 1, h)
                dk_dz = _diff_4th(eval_k_local, pos, 2, h)
                # M_{i,j} = d_j k_i
                M = torch.stack([dk_dx, dk_dy, dk_dz], dim=-1)

                # Spatial & stationary Minkowski null geodesicity: (k . grad) k = M @ k = 0
                geod_err = float(torch.linalg.norm(M @ k0).item())
                self.assertLess(geod_err, 1e-10)

                # Frobenius norm ||grad k||_F^2 = 2 / |w|^2
                frob_num = float(torch.sum(M * M).item())
                frob_pred = 2.0 / float((torch.abs(w0) ** 2).item())
                self.assertLess(abs(frob_num - frob_pred), 1e-9)

                # Screen-projected optical tensor B_AB = e_A^i e_B^j d_j k_i
                e1, e2 = _right_handed_screen_basis(k0)
                E = torch.stack([e1, e2], dim=0)  # (2, 3)
                B = E @ M @ E.T
                theta = 0.5 * (B[0, 0] + B[1, 1])
                omega = 0.5 * (B[0, 1] - B[1, 0])
                sigma = complex(0.5 * float((B[0, 0] - B[1, 1]).item()), 0.5 * float((B[0, 1] + B[1, 0]).item()))
                rho_num = complex(float(theta.item()), float(omega.item()))
                rho_pred = complex(cg.optical_scalar(w0).item())

                self.assertLess(abs(sigma), 1e-10)
                self.assertLess(abs(rho_num - rho_pred), 1e-10)

    def test_3_reversed_kz_regression_detects_shear_breakage(self):
        """Regression test: flipping kz -> -kz breaks shear-freeness (reproducing legacy bug)."""
        b = 0.85
        h = 1e-5
        pos = torch.tensor([0.55, 0.45, 0.50], dtype=torch.float64)
        w0 = cg.principal_w(pos, b, sheet_sign=+1.0)

        def eval_k_flipped(p: torch.Tensor) -> torch.Tensor:
            w_loc = cg.continue_sheet_step(p, w0, b)
            k = cg.tangent_field(p, w_loc, b)
            return torch.stack([k[..., 0], k[..., 1], -k[..., 2]], dim=-1)

        k_bad = eval_k_flipped(pos)
        M_bad = torch.stack([
            _diff_4th(eval_k_flipped, pos, 0, h),
            _diff_4th(eval_k_flipped, pos, 1, h),
            _diff_4th(eval_k_flipped, pos, 2, h),
        ], dim=-1)
        e1, e2 = _right_handed_screen_basis(k_bad)
        E = torch.stack([e1, e2], dim=0)
        B_bad = E @ M_bad @ E.T
        sigma_bad = complex(0.5 * float((B_bad[0, 0] - B_bad[1, 1]).item()), 0.5 * float((B_bad[0, 1] + B_bad[1, 0]).item()))
        self.assertGreater(abs(sigma_bad), 0.2)

    def test_4_non_antipodal_deck_transformation_and_sheet_mismatch(self):
        b = 1.0
        pos = torch.tensor([2.0, 0.0, 0.0], dtype=torch.float64)
        w_p = cg.principal_w(pos, b, sheet_sign=+1.0)
        w_m = -w_p
        k_p = cg.tangent_field(pos, w_p, b)
        k_m = cg.tangent_field(pos, w_m, b)

        # At (2, 0, 0), b=1, w=sqrt(3): k_+ = (sqrt(3)/2, -1/2, 0), k_- = (-sqrt(3)/2, -1/2, 0)
        self.assertGreater(float(torch.linalg.norm(k_p + k_m).item()), 0.9)  # not antipodal
        self.assertLess(abs(complex(cg.optical_scalar(w_p).item()) + complex(cg.optical_scalar(w_m).item())), 1e-14)

        # Global mismatch norm formula D(x) = 2|w|/sqrt(alpha^2 + b^2)
        gen = torch.Generator().manual_seed(42)
        pts = (torch.rand((200, 3), generator=gen, dtype=torch.float64) - 0.5) * 6.0
        w = cg.principal_w(pts, b=1.25, sheet_sign=+1.0)
        mask = torch.abs(w) > 1e-3
        pts, w = pts[mask], w[mask]
        num_diff = torch.linalg.norm(cg.tangent_field(pts, w, 1.25) - cg.tangent_field(pts, -w, 1.25), dim=-1)
        pred_diff = cg.sheet_mismatch_norm(pts, w, 1.25)
        self.assertLess(float(torch.max(torch.abs(num_diff - pred_diff)).item()), 1e-14)

    def test_5_bounded_two_chart_atlas_and_screen_transition(self):
        b = 1.25
        gen = torch.Generator().manual_seed(7)
        pts = (torch.rand((100, 3), generator=gen, dtype=torch.float64) - 0.5) * 4.0
        pts = torch.cat([pts, torch.tensor([[0.0, 0.0, 0.5], [0.0, 0.0, -0.5]], dtype=torch.float64)], dim=0)
        w = cg.principal_w(pts, b, sheet_sign=+1.0)
        res = cg.bounded_chart_and_screen(pts, w, b)

        # Bounded coordinate |coord| <= 1 everywhere
        self.assertLessEqual(float(torch.max(torch.abs(res["coord"])).item()), 1.0 + 1e-14)

        # Orthonormality: k.m = 0, m.m = 0, m.conj(m) = 1
        k_c = res["k"].to(dtype=torch.complex128)
        m = res["m"]
        self.assertLess(float(torch.max(torch.abs(torch.sum(k_c * m, dim=-1))).item()), 1e-14)
        self.assertLess(float(torch.max(torch.abs(torch.sum(m * m, dim=-1))).item()), 1e-14)
        self.assertLess(float(torch.max(torch.abs(torch.sum(m * torch.conj(m), dim=-1) - 1.0)).item()), 1e-14)

        # Exact overlap transition m_N(1/zeta) = (conj(zeta)/zeta) * m_S(zeta)
        zeta_ov = torch.tensor([0.6 + 0.7j, -0.9 + 0.4j, 1.2 - 0.5j], dtype=torch.complex128)
        lhs = cg.screen_vector_north(1.0 / zeta_ov)
        rhs = (torch.conj(zeta_ov) / zeta_ov).unsqueeze(-1) * cg.screen_vector_south(zeta_ov)
        self.assertLess(float(torch.max(torch.abs(lhs - rhs)).item()), 1e-14)

    def test_6_conformal_footprint_transport_and_lifted_disk_crossing(self):
        b = 1.25
        x0 = torch.tensor([0.3, -0.4, 0.5], dtype=torch.float64)
        w0 = cg.principal_w(x0, b, sheet_sign=-1.0)  # alpha_0 < 0
        k0 = cg.tangent_field(x0, w0, b)
        ds = 2.8

        # Verified lifted-disk puncture inside {z=0, x^2+y^2 < b^2}
        s_star = float((-w0.real).item())
        self.assertTrue(0.0 < s_star < ds)
        x_star = x0 + s_star * k0
        r_star = math.hypot(float(x_star[0].item()), float(x_star[1].item()))
        self.assertLess(abs(float(x_star[2].item())), 1e-14)
        self.assertLess(r_star, abs(b))

        # Incremental continuation across the lifted disk
        w_curr = w0
        for s in torch.linspace(0.0, ds, 201, dtype=torch.float64)[1:]:
            xs = x0 + s * k0
            w_curr = cg.continue_sheet_step(xs, w_curr, b)
            self.assertLess(float(torch.abs(w_curr - (w0 + s)).item()), 1e-13)
            self.assertLess(float(torch.linalg.norm(cg.tangent_field(xs, w_curr, b) - k0).item()), 1e-13)

        # Conformal footprint transport F(s) <-> q = (w0 + ds) / w0 with O(eps^2) check
        e1, e2 = _right_handed_screen_basis(k0)
        q = complex(((w0 + ds) / w0).item())
        F_pred = torch.tensor([[q.real, -q.imag], [q.imag, q.real]], dtype=torch.float64)

        errs = []
        for eps in (1e-4, 1e-5, 1e-6):
            F_num = torch.zeros((2, 2), dtype=torch.float64)
            for col, e_vec in enumerate((e1, e2)):
                xp, xm = x0 + eps * e_vec, x0 - eps * e_vec
                kp = cg.tangent_field(xp, cg.continue_sheet_step(xp, w0, b), b)
                km = cg.tangent_field(xm, cg.continue_sheet_step(xm, w0, b), b)
                sep = ((xp + ds * kp) - (xm + ds * km)) / (2.0 * eps)
                F_num[0, col] = torch.dot(sep, e1)
                F_num[1, col] = torch.dot(sep, e2)
            errs.append(float(torch.linalg.norm(F_num - F_pred).item()))
            if eps == 1e-6:
                sv = torch.linalg.svdvals(F_num)
                self.assertLess(abs(float((sv[0] / sv[1]).item()) - 1.0), 1e-8)
                area_pred = abs(complex((w0 + ds).item())) ** 2 / (abs(complex(w0.item())) ** 2)
                self.assertLess(abs(abs(float(torch.linalg.det(F_num).item())) - area_pred), 1e-7)

        self.assertGreater(errs[0] / max(errs[1], 1e-16), 50.0)
        self.assertLess(errs[-1], 1e-8)

    def test_7_theorem2_robinson_villarceau_roundtrip_and_domain_guards(self):
        for theta, phi, b_signed in [(0.72, 0.48, +0.65), (0.95, 2.10, -0.75), (1.30, -1.10, +1.40)]:
            k = torch.tensor([
                math.sin(theta) * math.cos(phi),
                math.sin(theta) * math.sin(phi),
                math.cos(theta),
            ], dtype=torch.float64)
            x0 = torch.tensor([-b_signed * float(k[1].item()), b_signed * float(k[0].item()), 0.0], dtype=torch.float64)
            fwd = cg.robinson_ray_to_villarceau_circle(x0, k, num_points=128)
            # Permute points to verify order-independence
            perm = torch.randperm(128, generator=torch.Generator().manual_seed(123))
            inv = cg.villarceau_circle_to_robinson_ray(fwd["points"][perm])

            self.assertLess(float(torch.linalg.norm(inv["k"] - k).item()), 1e-13)
            self.assertLess(float(torch.linalg.norm(inv["x0"] - x0).item()), 1e-13)
            self.assertLess(abs(inv["b"] - b_signed), 1e-13)
            self.assertLess(float(torch.linalg.norm(inv["center"] - fwd["center"]).item()), 1e-13)
            self.assertLess(abs(inv["radius"] - fwd["radius"]), 1e-13)

        # +b and -b point sets have opposite nonzero centers
        k_sep = torch.tensor([0.6, 0.0, 0.8], dtype=torch.float64)
        c_pos = cg.robinson_ray_to_villarceau_circle(torch.tensor([0.0, 0.6, 0.0], dtype=torch.float64), k_sep)["center"]
        c_neg = cg.robinson_ray_to_villarceau_circle(torch.tensor([0.0, -0.6, 0.0], dtype=torch.float64), k_sep)["center"]
        self.assertLess(float(torch.linalg.norm(c_pos + c_neg).item()), 1e-14)
        self.assertGreater(float(torch.linalg.norm(c_pos).item()), 0.1)

        # Domain guard checks (outside D_Rob must raise ValueError)
        with self.assertRaises(ValueError):
            cg.robinson_ray_to_params(torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64), k_sep)  # x0 == 0
        with self.assertRaises(ValueError):
            cg.robinson_ray_to_params(torch.tensor([0.5, 0.0, 0.1], dtype=torch.float64), k_sep)  # z0 != 0
        with self.assertRaises(ValueError):
            cg.robinson_ray_to_params(torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64), torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64))  # kz == 1


if __name__ == "__main__":
    unittest.main(verbosity=2)