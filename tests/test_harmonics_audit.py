import math
import unittest
import numpy as np
import torch
from scipy.special import sph_harm_y
from twistor.harmonics import PenroseTwistorHarmonicEngine
from twistor.incidence import sl2c_from_lie_algebra

class TestGenuineConformalHarmonics(unittest.TestCase):

    def setUp(self):
        self.engine = PenroseTwistorHarmonicEngine()

    def test_1_agreement_with_scipy_spherical_harmonics(self):
        """
        Verifies that our spinor bilinear eigenmodes Y_{l,m}(pi) match SciPy's 
        reference spherical harmonics sph_harm_y(l, m, theta, phi) to machine precision.
        """
        torch.manual_seed(101)
        # Generate random angles theta in (0, pi), phi in (0, 2*pi)
        theta = torch.DoubleTensor(200).uniform_(0.05, math.pi - 0.05)
        phi = torch.DoubleTensor(200).uniform_(0.0, 2.0 * math.pi)

        nx = torch.sin(theta) * torch.cos(phi)
        ny = torch.sin(theta) * torch.sin(phi)
        nz = torch.cos(theta)
        pts = torch.stack([nx, ny, nz], dim=-1)

        pi = self.engine.points_to_celestial_spinors(pts)
        modes = self.engine.evaluate_complex_eigenmodes(pi)

        theta_np = theta.numpy()
        phi_np = phi.numpy()

        max_err = 0.0
        for (l, m), val_spinor in modes.items():
            ref_scipy = sph_harm_y(l, m, theta_np, phi_np)
            err = np.max(np.abs(val_spinor.numpy() - ref_scipy))
            max_err = max(max_err, float(err))

        print(f"  -> [HARMONIC AUDIT 1] Spinor Y_{{l,m}}(pi) vs SciPy sph_harm_y Max Error: {max_err:.2e}")
        self.assertLess(max_err, 1e-14)

    def test_2_exact_laplace_beltrami_eigenvalues(self):
        """
        Directly applies the spherical Laplace-Beltrami differential operator
        -Delta_{S^2} f = -[ (1/sin(theta)) d/dtheta(sin(theta) df/dtheta) + (1/sin^2(theta)) d^2f/dphi^2 ]
        via exact torch.autograd differential calculus and verifies:
          - Monopole (l=0): -Delta Y_{0,0} = 0 * Y_{0,0}  (lambda_0 = 0, Lowest Energy Storage)
          - Dipole   (l=1): -Delta Y_{1,m} = 2 * Y_{1,m}  (lambda_1 = 2)
          - Quadrupole(l=2): -Delta Y_{2,m} = 6 * Y_{2,m} (lambda_2 = 6)
        """
        torch.manual_seed(202)
        theta = torch.DoubleTensor(64).uniform_(0.2, math.pi - 0.2).requires_grad_(True)
        phi = torch.DoubleTensor(64).uniform_(0.2, 2.0 * math.pi - 0.2).requires_grad_(True)

        nx = torch.sin(theta) * torch.cos(phi)
        ny = torch.sin(theta) * torch.sin(phi)
        nz = torch.cos(theta)
        pts = torch.stack([nx, ny, nz], dim=-1)
        pi = self.engine.points_to_celestial_spinors(pts)
        basis = self.engine.evaluate_real_basis(pi)  # (N, 9)

        expected_lambdas = [0.0, 2.0, 2.0, 2.0, 6.0, 6.0, 6.0, 6.0, 6.0]
        max_pde_residual = 0.0

        for idx, lam in enumerate(expected_lambdas):
            f = basis[:, idx]
            df_dtheta, df_dphi = torch.autograd.grad(f.sum(), (theta, phi), create_graph=True)
            sin_t_df = torch.sin(theta) * df_dtheta
            d2f_dtheta2_term = torch.autograd.grad(sin_t_df.sum(), theta, retain_graph=True)[0] / torch.sin(theta)
            d2f_dphi2 = torch.autograd.grad(df_dphi.sum(), phi, retain_graph=True)[0]

            neg_laplacian_f = -(d2f_dtheta2_term + d2f_dphi2 / (torch.sin(theta)**2))
            pde_residual = torch.max(torch.abs(neg_laplacian_f - lam * f)).item()
            max_pde_residual = max(max_pde_residual, pde_residual)

        print(f"  -> [HARMONIC AUDIT 2] Autograd PDE Eigenvalue Residual ||(-Delta - l(l+1))Y_lm||_inf: {max_pde_residual:.2e}")
        self.assertLess(max_pde_residual, 1e-13)

    def test_3_hilbert_space_orthonormality(self):
        """
        Verifies L^2(CP^1, sin(theta) dtheta dphi) Hilbert space orthonormality
        <Y_a | Y_b> = delta_{ab} using Gauss-Legendre quadrature on S^2.
        """
        n_quad = 32
        nodes, weights_z = np.polynomial.legendre.leggauss(n_quad)
        phi_nodes = np.linspace(0.0, 2.0 * math.pi, 2 * n_quad, endpoint=False)
        dphi = 2.0 * math.pi / (2 * n_quad)

        zz, pp = np.meshgrid(nodes, phi_nodes, indexing="ij")
        ww = np.broadcast_to(weights_z[:, None] * dphi, zz.shape).reshape(-1)

        theta_q = np.arccos(zz.reshape(-1))
        phi_q = pp.reshape(-1)

        pts = torch.tensor(np.stack([
            np.sin(theta_q) * np.cos(phi_q),
            np.sin(theta_q) * np.sin(phi_q),
            np.cos(theta_q)
        ], axis=-1), dtype=torch.float64)

        w_tensor = torch.tensor(ww, dtype=torch.float64)
        pi = self.engine.points_to_celestial_spinors(pts)
        B = self.engine.evaluate_real_basis(pi)  # (M, 9)

        gram = torch.einsum("ma,m,mb->ab", B, w_tensor, B)
        ortho_err = torch.max(torch.abs(gram - torch.eye(9, dtype=torch.float64))).item()
        print(f"  -> [HARMONIC AUDIT 3] L^2(CP^1) Hilbert Space Orthonormality Error ||G - I||_inf: {ortho_err:.2e}")
        self.assertLess(ortho_err, 1e-14)

    def test_4_robot_translation_and_su2_rotation_invariance(self):
        """
        Proves that our Conformal Casimir Energy Functional is 100% invariant under:
          1. Arbitrary 3D robot camera translations (e.g., moving 500 meters away)
          2. Arbitrary SU(2) spinor rotations of the camera frame
        Whereas the old heuristic (Tr(X)^2 - 4 det(X) = 4 r^2) blew up under translation!
        """
        torch.manual_seed(303)
        C_cam = torch.tensor([2.5, -1.8, 0.9], dtype=torch.float64)
        pts_local = torch.randn((300, 3), dtype=torch.float64)
        pts_world = pts_local + C_cam

        pi_orig = self.engine.points_to_celestial_spinors(pts_world, camera_center=C_cam)
        B_orig = self.engine.evaluate_real_basis(pi_orig)
        # Synthesize a pure dipole + quadrupole physical field on the sphere
        psi_field = 1.5 * B_orig[:, 0] + 2.0 * B_orig[:, 2] - 0.8 * B_orig[:, 6]
        res_orig = self.engine.compute_conformal_energy_functional(pi_orig, psi_field)

        # 1. Translate robot & world by T = [+500.0, -320.0, +1200.0] meters and scale depths by 3.7x
        T_shift = torch.tensor([500.0, -320.0, 1200.0], dtype=torch.float64)
        pts_translated = C_cam + T_shift + 3.7 * (pts_world - C_cam)
        C_translated = C_cam + T_shift
        pi_trans = self.engine.points_to_celestial_spinors(pts_translated, camera_center=C_translated)
        res_trans = self.engine.compute_conformal_energy_functional(pi_trans, psi_field)

        # 2. Rotate spinors by an arbitrary SU(2) operator U
        U = sl2c_from_lie_algebra(torch.tensor([0.75, -1.10, 0.45], dtype=torch.float64))
        pi_rot = torch.einsum("ab,nb->na", U, pi_orig)
        res_rot = self.engine.compute_conformal_energy_functional(pi_rot, psi_field)

        trans_err = abs(float(res_orig["conformal_energy"].item() - res_trans["conformal_energy"].item()))
        rot_err   = abs(float(res_orig["conformal_energy"].item() - res_rot["conformal_energy"].item()))
        print(f"  -> [HARMONIC AUDIT 4] Robot Translation/Scale Energy Error: {trans_err:.2e} | SU(2) Rotation Energy Error: {rot_err:.2e}")
        self.assertLess(trans_err, 1e-13)
        self.assertLess(rot_err, 1e-12)

if __name__ == "__main__":
    print("="*94)
    print("[ARGON MATHEMATICAL AUDIT] Testing Genuine L^2(CP^1) Conformal Casimir Harmonics vs References")
    print("="*94)
    unittest.main(verbosity=2)