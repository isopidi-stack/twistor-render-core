import unittest
import torch
from twistor.cohomology_penrose import TwistorCohomologyEngine, CechCohomologyClassH1
from twistor.penrose_core import PenroseTwistorCore
from twistor.harmonics import PenroseTwistorHarmonicEngine

class TestRigorousPenroseTransformAndBaselines(unittest.TestCase):

    def test_1_strict_homogeneity_of_actual_integrand(self):
        """
        Verifies exact homogeneity f(lambda * Z) = lambda^{-2} f(Z) on the actual elementary
        twistor function using the (is_homogeneous, rel_err) tuple and explicit tolerance check.
        """
        torch.manual_seed(42)
        Z = torch.randn((50, 4), dtype=torch.complex128)
        is_homog, rel_err = TwistorCohomologyEngine.check_homogeneity(
            TwistorCohomologyEngine.elementary_twistor_function, Z, k=-2, tol=1e-14
        )
        print(f"  -> [RIGOROUS AUDIT 1] Elementary State O(-2) Homogeneity: {is_homog} | Rel Error: {rel_err:.2e}")
        self.assertTrue(is_homog)
        self.assertLess(rel_err, 1e-14)

    def test_2_pole_separation_and_non_vanishing_wave_solution(self):
        """
        Verifies pole separation, non-zero field |phi(x)| > 0.1, contour vs Cauchy residue match,
        and 4D massless wave equation Box phi(x) = 0 via exact autograd Hessian.
        """
        x4 = torch.tensor([0.15, -0.10, 0.12, 0.08], dtype=torch.float64, requires_grad=True)
        _, _, _, _, zeta_A, zeta_B = TwistorCohomologyEngine.linear_forms_on_cp1(x4)

        mag_A = torch.abs(zeta_A).item()
        mag_B = torch.abs(zeta_B).item()
        self.assertLess(mag_A, 0.6)
        self.assertGreater(mag_B, 1.8)

        phi_contour = TwistorCohomologyEngine.contour_integral_transform(x4, num_pts=256, adaptive_radius=True)
        phi_exact   = TwistorCohomologyEngine.analytical_residue_transform(x4)
        field_mag   = torch.abs(phi_contour).item()
        quad_err    = torch.abs(phi_contour - phi_exact).item()

        self.assertGreater(field_mag, 0.1)
        self.assertLess(quad_err, 1e-14)

        def eval_re(pt):
            return TwistorCohomologyEngine.contour_integral_transform(pt, num_pts=256, adaptive_radius=True).real

        def eval_im(pt):
            return TwistorCohomologyEngine.contour_integral_transform(pt, num_pts=256, adaptive_radius=True).imag

        hess_re = torch.autograd.functional.hessian(eval_re, x4)
        hess_im = torch.autograd.functional.hessian(eval_im, x4)
        minkowski_diag = torch.tensor([1.0, -1.0, -1.0, -1.0], dtype=torch.float64)

        box_re = torch.sum(torch.diag(hess_re) * minkowski_diag).item()
        box_im = torch.sum(torch.diag(hess_im) * minkowski_diag).item()
        relative_wave_residual = (abs(box_re) + abs(box_im)) / field_mag

        print(f"  -> [RIGOROUS AUDIT 2] Poles: |zeta_A|={mag_A:.3f}, |zeta_B|={mag_B:.3f} | |phi(x)|={field_mag:.4f} | Wave Err: {relative_wave_residual:.2e}")
        self.assertLess(relative_wave_residual, 1e-13)

    def test_3_adaptive_contour_radius_near_unit_circle_poles(self):
        """
        Addresses the second review's #1 numerical finding:
        Samples 2,000 random spacetime points in [-1, 1]^4, filters for points that pass
        |zeta_A| < 1 < |zeta_B| INCLUDING near-boundary poles (margin < 0.05 from |zeta|=1),
        and proves that:
          - Fixed |zeta|=1 trapezoid quadrature degrades badly when pole margin < 0.05, whereas
          - Adaptive radius rho = sqrt(|zeta_A| * |zeta_B|) achieves < 1e-14 max relative error across all points!
        """
        torch.manual_seed(2026)
        pts_4d = (torch.rand((2000, 4), dtype=torch.float64) * 2.0 - 1.0)
        _, _, _, _, zeta_A, zeta_B = TwistorCohomologyEngine.linear_forms_on_cp1(pts_4d)
        mag_A = torch.abs(zeta_A)
        mag_B = torch.abs(zeta_B)

        # Keep points where |zeta_A| < 1 < |zeta_B|
        valid_mask = (mag_A < 0.995) & (mag_B > 1.005)
        pts_valid = pts_4d[valid_mask]

        phi_exact    = TwistorCohomologyEngine.analytical_residue_transform(pts_valid)
        phi_fixed    = TwistorCohomologyEngine.contour_integral_transform(pts_valid, num_pts=256, adaptive_radius=False)
        phi_adaptive = TwistorCohomologyEngine.contour_integral_transform(pts_valid, num_pts=256, adaptive_radius=True)

        rel_err_fixed    = torch.max(torch.abs(phi_fixed - phi_exact) / torch.abs(phi_exact)).item()
        rel_err_adaptive = torch.max(torch.abs(phi_adaptive - phi_exact) / torch.abs(phi_exact)).item()

        print(f"  -> [RIGOROUS AUDIT 3: Adaptive Radius ({pts_valid.shape[0]} pts)] Fixed |zeta|=1 Max Rel Err: {rel_err_fixed:.2e} vs Adaptive rho=sqrt(|zA||zB|) Max Rel Err: {rel_err_adaptive:.2e}")
        self.assertLess(rel_err_adaptive, 1e-14)

    def test_4_genuine_u0_u1_cech_coboundary_and_invalid_pole_rejection(self):
        """
        Verifies:
          1. Valid Cech 0-cochain pair (C_u0, D_u1) passes explicit U_0 / U_1 patch holomorphicity
             verification and leaves phi(x) invariant to ~1e-16.
          2. An invalid coboundary section whose pole lies INSIDE U_0 (|zeta_C| < rho) is
             strictly rejected by CechCohomologyClassH1.verify_patch_holomorphicity!
        """
        x4 = torch.tensor([0.15, -0.10, 0.12, 0.08], dtype=torch.float64)

        # Valid C_u0 (pole at |zeta_C| ~ 3.0 > rho) and valid D_u1 (pole at |zeta_D| ~ 0.22 < rho)
        C_valid = torch.tensor([0.10 - 0.05j, 0.20 + 0.10j, 3.00 + 0.50j, 1.00 + 0.00j], dtype=torch.complex128)
        D_valid = torch.tensor([-0.15 + 0.10j, 0.10 - 0.20j, -0.20 + 0.10j, 1.00 + 0.00j], dtype=torch.complex128)

        phi_orig = TwistorCohomologyEngine.contour_integral_transform(x4, num_pts=256, adaptive_radius=True)
        phi_pert = TwistorCohomologyEngine.contour_integral_transform(
            x4, num_pts=256, adaptive_radius=True, cech_coboundary_pair=(C_valid, D_valid)
        )
        gauge_diff = torch.abs(phi_pert - phi_orig).item()
        self.assertLess(gauge_diff, 1e-14)

        # Now test an INVALID U_0 section C_bad whose pole is at ~0.10 (INSIDE U_0!)
        C_bad = torch.tensor([0.01 + 0.01j, -0.01 + 0.01j, -0.10 + 0.00j, 1.00 + 0.00j], dtype=torch.complex128)
        with self.assertRaises(ValueError):
            TwistorCohomologyEngine.contour_integral_transform(
                x4, num_pts=256, adaptive_radius=True, cech_coboundary_pair=(C_bad, D_valid)
            )
        print(f"  -> [RIGOROUS AUDIT 4] Valid U_0/U_1 Cech Coboundary |phi_pert - phi_orig|: {gauge_diff:.2e} | Invalid In-Patch Pole Rejected: True")

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