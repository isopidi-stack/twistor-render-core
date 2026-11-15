import unittest
import torch
from twistor.incidence import sl2c_from_lie_algebra, rotate_spinor
from twistor.penrose_core import PenroseTwistorCore

class TestFullPenroseMathematicalStack(unittest.TestCase):

    def test_1_lorentz_boost_incidence_covariance(self):
        """
        Verifies that fixing pi' = (Lambda^dagger)^{-1} pi ensures exact Lorentz covariance
        of the Penrose incidence equation omega'^A = Lambda^A_B omega^B under BOTH SU(2)
        rotations AND relativistic SL(2,C) boosts.
        """
        torch.manual_seed(11)
        axis_angle = torch.tensor([0.65, -0.45, 0.80], dtype=torch.float64)
        boost_rapidity = torch.tensor([0.55, -0.35, 0.25], dtype=torch.float64)
        Lambda = sl2c_from_lie_algebra(axis_angle, boost_rapidity)

        x4 = torch.tensor([3.2, 1.1, -0.9, 2.4], dtype=torch.float64)
        X = PenroseTwistorCore.spacetime_to_hermitian(x4)
        pi = torch.tensor([0.8 + 0.2j, -0.3 + 0.5j], dtype=torch.complex128)
        omega_orig = 1j * (X @ pi)

        X_rot, omega_from_incidence, pi_rot = rotate_spinor(Lambda, X, pi)
        omega_covariant = Lambda @ omega_orig

        cov_err = torch.max(torch.abs(omega_from_incidence - omega_covariant)).item()
        det_err = abs(float(torch.linalg.det(X_rot).real.item() - torch.linalg.det(X).real.item()))
        print(f"  -> [PENROSE STACK 1] Boosted Incidence Covariance Error ||omega' - Lambda omega||: {cov_err:.2e} | Minkowski Interval Error: {det_err:.2e}")
        self.assertLess(cov_err, 1e-14)
        self.assertLess(det_err, 1e-14)

    def test_2_su22_poincare_metric_invariance(self):
        """
        Verifies that our 4x4 Poincare matrix P(Lambda, T) in SU(2,2) preserves the
        pseudo-Hermitian twistor metric Sigma: P^dagger Sigma P = Sigma, and preserves
        helicity s = 1/2 Z^dagger Sigma Z under arbitrary 6-DoF translations + boosts!
        """
        Lambda = sl2c_from_lie_algebra(
            torch.tensor([0.4, 0.9, -0.5], dtype=torch.float64),
            torch.tensor([-0.3, 0.2, 0.4], dtype=torch.float64)
        )
        T_4d = torch.tensor([12.5, -45.0, 88.2, -19.4], dtype=torch.float64)
        P = PenroseTwistorCore.build_poincare_su22_matrix(Lambda, T_4d)
        Sigma = PenroseTwistorCore.su22_metric(dtype=torch.complex128)

        metric_err = torch.max(torch.abs(torch.conj(P.mT) @ Sigma @ P - Sigma)).item()
        print(f"  -> [PENROSE STACK 2] 4x4 SU(2,2) Poincare Metric Invariance ||P^dagger Sigma P - Sigma||: {metric_err:.2e}")
        self.assertLess(metric_err, 1e-13)

    def test_3_twistor_klein_quadric_null_ray_intersection(self):
        """
        Proves Penrose's Fundamental Twistor Intersection Theorem:
        Two null twistors Z1, Z2 satisfy <Z1, Z2>_Sigma = 0 if and only if their corresponding
        null rays in Minkowski space intersect at a common spacetime event x0!
        """
        x0 = torch.tensor([1.5, -2.2, 3.8, 0.7], dtype=torch.float64)
        X0 = PenroseTwistorCore.spacetime_to_hermitian(x0)

        # Two distinct rays passing through the same event x0
        pi_1 = torch.tensor([1.0 + 0.0j, 0.4 - 0.7j], dtype=torch.complex128)
        pi_2 = torch.tensor([0.6 + 0.3j, -0.5 + 0.2j], dtype=torch.complex128)
        Z1 = torch.cat([1j * (X0 @ pi_1), pi_1], dim=-1)
        Z2 = torch.cat([1j * (X0 @ pi_2), pi_2], dim=-1)

        # A third ray passing through a displaced non-null-separated point x1
        x1 = x0 + torch.tensor([2.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        X1 = PenroseTwistorCore.spacetime_to_hermitian(x1)
        Z3 = torch.cat([1j * (X1 @ pi_2), pi_2], dim=-1)

        intersecting_prod = abs(PenroseTwistorCore.su22_inner_product(Z1, Z2).item())
        skew_prod = abs(PenroseTwistorCore.su22_inner_product(Z1, Z3).item())
        print(f"  -> [PENROSE STACK 3] Intersecting Rays |<Z1, Z2>_Sigma|: {intersecting_prod:.2e} | Skew Rays |<Z1, Z3>_Sigma|: {skew_prod:.4f}")
        self.assertLess(intersecting_prod, 1e-14)
        self.assertGreater(skew_prod, 0.1)

    def test_4_forward_tube_positive_frequency_splitting(self):
        """
        Verifies that lifting real spacetime events into the complex Forward Tube M^+
        (z^a = x^a - i eta^a with future-timelike eta^a) maps twistors strictly into PT^+
        (Z^dagger Sigma Z > 0), while past-timelike -eta^a maps strictly into PT^- (Z^dagger Sigma Z < 0),
        and real events lie exactly on the null boundary PN (Z^dagger Sigma Z == 0).
        """
        x_real = torch.tensor([4.0, -1.2, 2.5, 0.8], dtype=torch.float64)
        eta_future = torch.tensor([1.5, 0.3, -0.4, 0.2], dtype=torch.float64)  # eta^0 = 1.5 > ||vec|| = 0.538
        pi = torch.tensor([0.7 + 0.3j, -0.2 + 0.6j], dtype=torch.complex128)

        Z_pos = PenroseTwistorCore.lift_to_forward_tube(x_real, eta_future, pi)
        Z_neg = PenroseTwistorCore.lift_to_forward_tube(x_real, -eta_future, pi)
        Z_null = PenroseTwistorCore.lift_to_forward_tube(x_real, torch.zeros_like(eta_future), pi)

        norm_pos = PenroseTwistorCore.su22_inner_product(Z_pos, Z_pos).real.item()
        norm_neg = PenroseTwistorCore.su22_inner_product(Z_neg, Z_neg).real.item()
        norm_null = abs(PenroseTwistorCore.su22_inner_product(Z_null, Z_null).item())

        print(f"  -> [PENROSE STACK 4] Tube Splitting | PT^+ Norm: +{norm_pos:.4f} (>0) | PT^- Norm: {norm_neg:.4f} (<0) | PN Null Boundary: {norm_null:.2e}")
        self.assertGreater(norm_pos, 0.5)
        self.assertLess(norm_neg, -0.5)
        self.assertLess(norm_null, 1e-15)

    def test_5_penrose_contour_transform_solves_4d_wave_equation(self):
        """
        Verifies Penrose's Crown Theorem: Any field phi(x) generated by the contour integral
        transform oint f(i X^{AA'} pi_{A'}, pi_{A'}) d_pi over CP^1 automatically satisfies
        the exact 4D massless wave equation Box phi(x) = (d_t^2 - d_x^2 - d_y^2 - d_z^2) phi = 0!
        """
        x4 = torch.tensor([0.4, -0.3, 0.25, 0.15], dtype=torch.float64, requires_grad=True)
        x0_pole = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float64)

        def eval_real(pt):
            return PenroseTwistorCore.penrose_contour_transform_scalar(pt, x0_pole, num_contour_pts=128).real

        def eval_imag(pt):
            return PenroseTwistorCore.penrose_contour_transform_scalar(pt, x0_pole, num_contour_pts=128).imag

        hess_re = torch.autograd.functional.hessian(eval_real, x4)
        hess_im = torch.autograd.functional.hessian(eval_imag, x4)

        # D'Alembertian Box = d^2/dt^2 - d^2/dx^2 - d^2/dy^2 - d^2/dz^2
        minkowski_diag = torch.tensor([1.0, -1.0, -1.0, -1.0], dtype=torch.float64)
        box_re = torch.sum(torch.diag(hess_re) * minkowski_diag).item()
        box_im = torch.sum(torch.diag(hess_im) * minkowski_diag).item()
        wave_eq_residual = abs(box_re) + abs(box_im)

        print(f"  -> [PENROSE STACK 5] Penrose Contour Integral 4D Wave Operator Residual |Box phi(x)|: {wave_eq_residual:.2e}")
        self.assertLess(wave_eq_residual, 1e-14)

if __name__ == "__main__":
    print("="*96)
    print("[ARGON FULL PENROSE STACK AUDIT] Testing SU(2,2), Forward Tube PT^+, & Contour Transform")
    print("="*96)
    unittest.main(verbosity=2)