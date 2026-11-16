import unittest
import torch
from twistor.cohomology_penrose import TwistorCohomologyEngine, CechCohomologyClassH1
from twistor.penrose_core import PenroseTwistorCore
from twistor.harmonics import PenroseTwistorHarmonicEngine

class TestRigorousPenroseTransformAndBaselines(unittest.TestCase):

    def test_1_strict_homogeneity_of_actual_integrand(self):
        torch.manual_seed(42)
        Z = torch.randn((50, 4), dtype=torch.complex128)
        is_homog, rel_err = TwistorCohomologyEngine.check_homogeneity(
            TwistorCohomologyEngine.elementary_twistor_function, Z, k=-2, tol=1e-14
        )
        print(f"  -> [RIGOROUS AUDIT 1] Elementary State O(-2) Homogeneity: {is_homog} | Rel Error: {rel_err:.2e}")
        self.assertTrue(is_homog)
        self.assertLess(rel_err, 1e-14)

    def test_2_pole_separation_and_non_vanishing_wave_solution(self):
        x4 = torch.tensor([0.15, -0.10, 0.12, 0.08], dtype=torch.float64, requires_grad=True)
        _, _, _, _, zeta_A, zeta_B = TwistorCohomologyEngine.linear_forms_on_cp1(x4)

        mag_A = torch.abs(zeta_A).item()
        mag_B = torch.abs(zeta_B).item()

        phi_contour = TwistorCohomologyEngine.contour_integral_transform(x4, num_pts=256)
        phi_exact   = TwistorCohomologyEngine.analytical_residue_transform(x4)
        field_mag   = torch.abs(phi_contour).item()
        quad_err    = torch.abs(phi_contour - phi_exact).item()

        self.assertGreater(field_mag, 0.1)
        self.assertLess(quad_err, 1e-15)

        def eval_re(pt):
            return TwistorCohomologyEngine.contour_integral_transform(pt, num_pts=256).real

        def eval_im(pt):
            return TwistorCohomologyEngine.contour_integral_transform(pt, num_pts=256).imag

        hess_re = torch.autograd.functional.hessian(eval_re, x4)
        hess_im = torch.autograd.functional.hessian(eval_im, x4)
        minkowski_diag = torch.tensor([1.0, -1.0, -1.0, -1.0], dtype=torch.float64)

        box_re = torch.sum(torch.diag(hess_re) * minkowski_diag).item()
        box_im = torch.sum(torch.diag(hess_im) * minkowski_diag).item()
        relative_wave_residual = (abs(box_re) + abs(box_im)) / field_mag

        print(f"  -> [RIGOROUS AUDIT 2] Poles: |zeta_A|={mag_A:.3f}, |zeta_B|={mag_B:.3f} | |phi(x)|={field_mag:.4f} | Wave Err: {relative_wave_residual:.2e}")
        self.assertLess(relative_wave_residual, 1e-13)

    def test_3_pole_centered_contour_on_wide_domain_and_tight_poles(self):
        """
        Addresses Reviewer 1's findings #1 and #2:
          - Tests 5,000 random points in the wide domain [-10, 10]^4 (where |zeta_A| > |zeta_B|
            on ~34% of points and |zeta_A|/|zeta_B| approaches 1).
          - Tests a custom tight-pole pair (A_tight, B_tight) with modulus ratio r = |zeta_A|/|zeta_B| = 0.967.
          - Proves that the pole-centered contour Gamma_x = { zeta_A + 0.5*|zeta_A - zeta_B|*e^{i*phi} }
            succeeds on 100% of points and achieves < 1e-14 relative error everywhere!
        """
        torch.manual_seed(2026)
        pts_wide = (torch.rand((5000, 4), dtype=torch.float64) * 20.0 - 10.0)
        phi_exact_wide = TwistorCohomologyEngine.analytical_residue_transform(pts_wide)
        phi_contour_wide = TwistorCohomologyEngine.contour_integral_transform(pts_wide, num_pts=256, contour_mode="pole_centered")
        rel_err_wide = torch.max(torch.abs(phi_contour_wide - phi_exact_wide) / torch.abs(phi_exact_wide)).item()

        # Tight-pole configuration with |zeta_A| / |zeta_B| = 0.967
        A_tight = torch.tensor([0.10 + 0.05j, -0.10 + 0.05j, -0.967 + 0.00j, 1.00 + 0.00j], dtype=torch.complex128)
        B_tight = torch.tensor([0.10 - 0.05j,  0.10 + 0.05j, -1.000 + 0.00j, 1.00 + 0.00j], dtype=torch.complex128)
        x_origin = torch.zeros(4, dtype=torch.float64)
        phi_exact_tight = TwistorCohomologyEngine.analytical_residue_transform(x_origin, A=A_tight, B=B_tight)
        phi_contour_tight = TwistorCohomologyEngine.contour_integral_transform(x_origin, num_pts=256, A=A_tight, B=B_tight, contour_mode="pole_centered")
        rel_err_tight = (torch.abs(phi_contour_tight - phi_exact_tight) / torch.abs(phi_exact_tight)).item()

        print(f"  -> [RIGOROUS AUDIT 3: Pole-Centered Contour] [-10,10]^4 (5000 pts) Max Rel Err: {rel_err_wide:.2e} | Tight Poles (r=0.967) Rel Err: {rel_err_tight:.2e}")
        self.assertLess(rel_err_wide, 1e-14)
        self.assertLess(rel_err_tight, 1e-14)

    def test_4_non_vacuous_simple_pole_cech_coboundary_and_negative_control(self):
        """
        Addresses Reviewer 1's finding #3 (vacuous double poles) and Reviewer 2's Priority #1:
          1. Uses simple-pole sections h_0(Z) = c0 / ((C1.Z)(C2.Z)) with non-zero residues at two
             distinct poles outside Gamma_x (holomorphic on V_0) and h_1(Z) = c1 / ((D1.Z)(D2.Z))
             with two distinct simple poles inside Gamma_x (holomorphic on V_1, opposite residues cancel).
             Verifies that adding delta(h) = h_0 - h_1 leaves phi(x) invariant to ~1e-16.
          2. Negative Control: Adds g(Z) = w / ((D1_in.Z)(C1_out.Z)) which has one simple pole INSIDE
             Gamma_x and one simple pole OUTSIDE Gamma_x (so g is NOT a coboundary on V_0 + V_1).
             Proves that:
               (a) verify_patch_holomorphicity strictly rejects (D1_in, C1_out) on both V_0 and V_1, and
               (b) integrating f + g shifts phi(x) by the exact non-zero residue Delta_phi = w / (p1*q0 - p0*q1)!
        """
        x4 = torch.tensor([0.15, -0.10, 0.12, 0.08], dtype=torch.float64)
        center_A, radius_R = TwistorCohomologyEngine.compute_pole_centered_contour(x4)

        # Two distinct simple poles outside Gamma_x (|zeta - zeta_A| > R ~ 1.40) -> valid V_0 section
        C1_out = torch.tensor([0.10 - 0.05j, 0.20 + 0.10j,  3.20 + 0.40j, 1.00 + 0.00j], dtype=torch.complex128)
        C2_out = torch.tensor([-0.10 + 0.05j, 0.15 - 0.10j, 4.10 - 0.30j, 1.00 + 0.00j], dtype=torch.complex128)

        # Two distinct simple poles inside Gamma_x (|zeta - zeta_A| < R ~ 1.40) -> valid V_1 section
        D1_in = torch.tensor([0.25 + 0.10j, -0.15 + 0.10j, -0.20 + 0.05j, 1.00 + 0.00j], dtype=torch.complex128)
        D2_in = torch.tensor([0.20 - 0.05j, -0.10 + 0.05j, -0.35 - 0.05j, 1.00 + 0.00j], dtype=torch.complex128)

        phi_orig = TwistorCohomologyEngine.contour_integral_transform(x4, num_pts=256)
        phi_cob  = TwistorCohomologyEngine.contour_integral_transform(
            x4, num_pts=256, cech_coboundary_v0=(C1_out, C2_out), cech_coboundary_v1=(D1_in, D2_in)
        )
        cob_diff = torch.abs(phi_cob - phi_orig).item()
        self.assertLess(cob_diff, 1e-14)

        # Negative Control: (D1_in, C1_out) has one pole inside Gamma_x and one pole outside Gamma_x!
        with self.assertRaises(ValueError):
            CechCohomologyClassH1.verify_patch_holomorphicity(x4, center_A, radius_R, D1_in, C1_out, patch="V0")
        with self.assertRaises(ValueError):
            CechCohomologyClassH1.verify_patch_holomorphicity(x4, center_A, radius_R, D1_in, C1_out, patch="V1")

        weight = 1.25 + 0.0j
        phi_shifted = TwistorCohomologyEngine.contour_integral_transform(
            x4, num_pts=256, extra_cocycle_term=(D1_in, C1_out, weight)
        )
        measured_shift = phi_shifted - phi_orig
        expected_shift = weight * TwistorCohomologyEngine.analytical_residue_transform(x4, A=D1_in, B=C1_out)
        neg_ctrl_err = torch.abs(measured_shift - expected_shift).item()

        print(f"  -> [RIGOROUS AUDIT 4: Non-Vacuous Cech] Coboundary Diff: {cob_diff:.2e} | Negative-Control Shift |Delta phi|={torch.abs(measured_shift).item():.4f} (Match Err: {neg_ctrl_err:.2e})")
        self.assertGreater(torch.abs(measured_shift).item(), 0.2)
        self.assertLess(neg_ctrl_err, 1e-14)

    def test_5_special_conformal_k_matrix_su22_invariance(self):
        b_4d = torch.tensor([0.8, -1.2, 0.5, -0.9], dtype=torch.float64)
        K = PenroseTwistorCore.build_special_conformal_su22_matrix(b_4d)
        Sigma = PenroseTwistorCore.su22_metric(dtype=torch.complex128)
        err = torch.max(torch.abs(torch.conj(K.mT) @ Sigma @ K - Sigma)).item()
        print(f"  -> [RIGOROUS AUDIT 5] Special Conformal K(b) SU(2,2) Metric Error ||K^dagger Sigma K - Sigma||: {err:.2e}")
        self.assertLess(err, 1e-15)

    def test_6_spinor_transverse_projector_equals_euclidean_projector(self):
        torch.manual_seed(77)
        pts = torch.randn((100, 3), dtype=torch.float64)
        pi = PenroseTwistorHarmonicEngine.points_to_celestial_spinors(pts)
        n = PenroseTwistorHarmonicEngine.spinor_to_unit_ray(pi)

        p0, p1 = pi[:, 0], pi[:, 1]
        q0, q1 = -torch.conj(p1), torch.conj(p0)
        q0_bar, q1_bar = torch.conj(q0), torch.conj(q1)

        mx = q0_bar * p1 + q1_bar * p0
        my = -1j * (q0_bar * p1 - q1_bar * p0)
        mz = q0_bar * p0 - q1_bar * p1
        norm_sq = (torch.abs(p0)**2 + torch.abs(p1)**2).unsqueeze(-1)
        m_vec = torch.stack([mx, my, mz], dim=-1) / norm_sq
        a, b = m_vec.real, m_vec.imag

        P_spinor = torch.einsum("ni,nj->nij", a, a) + torch.einsum("ni,nj->nij", b, b)
        I3 = torch.eye(3, dtype=torch.float64).unsqueeze(0)
        P_euclid = I3 - torch.einsum("ni,nj->nij", n, n)

        proj_diff = torch.max(torch.abs(P_spinor - P_euclid)).item()
        print(f"  -> [RIGOROUS AUDIT 6] Algebraic Identity ||(a a^T + b b^T) - (I - n n^T)||_inf: {proj_diff:.2e}")
        self.assertLess(proj_diff, 1e-14)

if __name__ == "__main__":
    unittest.main(verbosity=2)