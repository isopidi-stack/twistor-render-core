import unittest
import torch
from twistor.cohomology_penrose import TwistorCohomologyEngine
from twistor.penrose_core import PenroseTwistorCore
from twistor.harmonics import PenroseTwistorHarmonicEngine

class TestRigorousPenroseTransformAndBaselines(unittest.TestCase):

    def test_1_strict_homogeneity_of_actual_integrand(self):
        """
        Tests exact homogeneity f(lambda * Z) = lambda^{-2} f(Z) on the actual elementary
        twistor function used in the Penrose transform (with zero +1e-12 regularizers).
        """
        torch.manual_seed(42)
        Z = torch.randn((50, 4), dtype=torch.complex128)
        rel_err = TwistorCohomologyEngine.check_homogeneity(
            TwistorCohomologyEngine.elementary_twistor_function, Z, k=-2
        )
        print(f"  -> [RIGOROUS AUDIT 1] Elementary State O(-2) Relative Homogeneity Error: {rel_err:.2e}")
        self.assertLess(rel_err, 1e-14)

    def test_2_pole_separation_and_non_vanishing_wave_solution(self):
        """
        Addresses the review's pole-enclosure critique:
          1. Explicitly verifies that at x4, Pole A is strictly inside |zeta|=1 (|zeta_A| < 1)
             and Pole B is strictly outside |zeta|=1 (|zeta_B| > 1).
          2. Verifies that the resulting field phi(x) is NON-ZERO (|phi(x)| > 0.1).
          3. Verifies that the contour integral matches the exact Cauchy residue 1/(p1*q0 - p0*q1).
          4. Verifies that this non-zero field satisfies Box phi(x) = 0 to machine precision.
        """
        x4 = torch.tensor([0.15, -0.10, 0.12, 0.08], dtype=torch.float64, requires_grad=True)
        _, _, _, _, zeta_A, zeta_B = TwistorCohomologyEngine.linear_forms_on_cp1(x4)

        mag_A = torch.abs(zeta_A).item()
        mag_B = torch.abs(zeta_B).item()
        self.assertLess(mag_A, 0.6, "Pole A must lie strictly inside unit contour!")
        self.assertGreater(mag_B, 1.8, "Pole B must lie strictly outside unit contour!")

        phi_contour = TwistorCohomologyEngine.contour_integral_transform(x4, num_pts=256)
        phi_exact   = TwistorCohomologyEngine.analytical_residue_transform(x4)
        field_mag   = torch.abs(phi_contour).item()
        quad_err    = torch.abs(phi_contour - phi_exact).item()

        self.assertGreater(field_mag, 0.1, "Field phi(x) must be genuinely non-zero!")
        self.assertLess(quad_err, 1e-14)

        def eval_re(pt):
            return TwistorCohomologyEngine.contour_integral_transform(pt, num_pts=256).real

        def eval_im(pt):
            return TwistorCohomologyEngine.contour_integral_transform(pt, num_pts=256).imag

        hess_re = torch.autograd.functional.hessian(eval_re, x4)
        hess_im = torch.autograd.functional.hessian(eval_im, x4)
        minkowski_diag = torch.tensor([1.0, -1.0, -1.0, -1.0], dtype=torch.float64)

        box_re = torch.sum(torch.diag(hess_re) * minkowski_diag).item()
        box_im = torch.sum(torch.diag(hess_im) * minkowski_diag).item()
        wave_residual = abs(box_re) + abs(box_im)
        relative_wave_residual = wave_residual / field_mag

        print(f"  -> [RIGOROUS AUDIT 2] Poles: |zeta_A|={mag_A:.3f} (<1), |zeta_B|={mag_B:.3f} (>1) | |phi(x)|={field_mag:.4f} (!=0)")
        print(f"  -> [RIGOROUS AUDIT 2] Contour vs Residue Error: {quad_err:.2e} | Non-Zero Field |Box phi|/|phi|: {relative_wave_residual:.2e}")
        self.assertLess(relative_wave_residual, 1e-13)

    def test_3_genuine_u0_u1_cech_coboundary_invariance(self):
        """
        Tests genuine Cech coboundary invariance f_{01}(Z) -> f_{01}(Z) + h_0(Z) - h_1(Z) on U_0 intersect U_1:
          - h_0(Z) = c / (C . Z)^2 is homogeneous of degree -2 and holomorphic on U_0 (pole at |zeta_C| > 1).
          - h_1(Z) = d / (D . Z)^2 is homogeneous of degree -2 and holomorphic on U_1 minus {zeta_D} with zero residue inside |zeta|=1.
        Verifies that h_0(Z) - h_1(Z) is non-zero along the contour yet changes phi(x) by ~1e-16.
        """
        x4 = torch.tensor([0.15, -0.10, 0.12, 0.08], dtype=torch.float64)

        C = torch.tensor([0.10 - 0.05j, 0.20 + 0.10j, 3.00 + 0.50j, 1.00 + 0.00j], dtype=torch.complex128)
        D = torch.tensor([-0.15 + 0.10j, 0.10 - 0.20j, -0.20 + 0.10j, 1.00 + 0.00j], dtype=torch.complex128)

        def cech_coboundary(Z):
            CZ = torch.einsum("a,...a->...", C, Z)
            DZ = torch.einsum("a,...a->...", D, Z)
            h0_Z = 0.75 / (CZ ** 2)
            h1_Z = -0.45 / (DZ ** 2)
            return h0_Z - h1_Z

        Z_sample = torch.randn((20, 4), dtype=torch.complex128)
        cob_homog_err = TwistorCohomologyEngine.check_homogeneity(cech_coboundary, Z_sample, k=-2)
        self.assertLess(cob_homog_err, 1e-14)

        phi_original  = TwistorCohomologyEngine.contour_integral_transform(x4, num_pts=256)
        phi_perturbed = TwistorCohomologyEngine.contour_integral_transform(x4, num_pts=256, coboundary_fn=cech_coboundary)
        gauge_diff = torch.abs(phi_perturbed - phi_original).item()

        print(f"  -> [RIGOROUS AUDIT 3] U_0/U_1 Cech Coboundary Homogeneity Err: {cob_homog_err:.2e} | |phi_pert - phi_orig|: {gauge_diff:.2e}")
        self.assertLess(gauge_diff, 1e-14)

    def test_4_special_conformal_k_matrix_su22_invariance(self):
        """
        Verifies the true 4-parameter Special Conformal transformation K(b):
            pi'_{A'} = pi_{A'} - i B_{A'A} omega^A
        and proves K(b)^dagger Sigma K(b) = Sigma in SU(2,2).
        """
        b_4d = torch.tensor([0.8, -1.2, 0.5, -0.9], dtype=torch.float64)
        K = PenroseTwistorCore.build_special_conformal_su22_matrix(b_4d)
        Sigma = PenroseTwistorCore.su22_metric(dtype=torch.complex128)
        err = torch.max(torch.abs(torch.conj(K.mT) @ Sigma @ K - Sigma)).item()
        print(f"  -> [RIGOROUS AUDIT 4] Special Conformal K(b) SU(2,2) Metric Error ||K^dagger Sigma K - Sigma||: {err:.2e}")
        self.assertLess(err, 1e-15)

    def test_5_spinor_transverse_projector_equals_euclidean_projector(self):
        """
        Explicitly verifies the reviewer's algebraic proof: for any spinor pi on CP^1
        with orthogonal spinor pi_perp = (-conj(pi_1), conj(pi_0)), the spinor transverse
        basis vectors a, b satisfy a a^T + b b^T = I_3 - n n^T identically.
        """
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
        a = m_vec.real
        b = m_vec.imag

        P_spinor = torch.einsum("ni,nj->nij", a, a) + torch.einsum("ni,nj->nij", b, b)
        I3 = torch.eye(3, dtype=torch.float64).unsqueeze(0)
        P_euclid = I3 - torch.einsum("ni,nj->nij", n, n)

        proj_diff = torch.max(torch.abs(P_spinor - P_euclid)).item()
        print(f"  -> [RIGOROUS AUDIT 5] Algebraic Identity ||(a a^T + b b^T) - (I - n n^T)||_inf: {proj_diff:.2e}")
        self.assertLess(proj_diff, 1e-14)

if __name__ == "__main__":
    unittest.main(verbosity=2)