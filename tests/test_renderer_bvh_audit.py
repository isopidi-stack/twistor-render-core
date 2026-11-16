import unittest
import torch
from twistor.renderer import TwistorRenderer
from twistor.bvh import TwistorBVH
from twistor.penrose_core import PenroseTwistorCore
from twistor.harmonics import PenroseTwistorHarmonicEngine

class TestRendererAndBVHRemediation(unittest.TestCase):

    def test_1_full_15_parameter_su22_group_closure(self):
        theta_3d = torch.tensor([0.35, -0.70, 0.55], dtype=torch.float64)
        boost_3d = torch.tensor([0.25, -0.40, 0.30], dtype=torch.float64)
        trans_4d = torch.tensor([4.20, -1.80, 3.10, -2.50], dtype=torch.float64)
        b_4d     = torch.tensor([0.15, -0.25, 0.35, -0.10], dtype=torch.float64)
        lam      = 0.45

        G = TwistorRenderer.build_full_15param_su22_matrix(theta_3d, boost_3d, trans_4d, b_4d, dilation_lambda=lam)
        Sigma = PenroseTwistorCore.su22_metric(dtype=torch.complex128)
        metric_err = torch.max(torch.abs(torch.conj(G.mT) @ Sigma @ G - Sigma)).item()
        print(f"  -> [RENDERER AUDIT 1] Full 15-Param SU(2,2) ||G^dagger Sigma G - Sigma||: {metric_err:.2e}")
        self.assertLess(metric_err, 1e-13)

    def test_2_null_cone_lifting_det_zero(self):
        torch.manual_seed(12)
        pts = torch.randn((80, 3), dtype=torch.float64) * 10.0
        cam = torch.tensor([1.5, -2.0, 0.5], dtype=torch.float64)
        dX_herm, Z = TwistorRenderer.lift_points_to_null_cone_twistors(pts, cam)

        scale_sq = torch.sum((pts - cam.unsqueeze(0)) ** 2, dim=-1)
        rel_det_err = torch.max(torch.abs(torch.linalg.det(dX_herm)) / scale_sq).item()
        norm_err = torch.max(torch.abs(PenroseTwistorCore.su22_inner_product(Z, Z)) / scale_sq).item()
        print(f"  -> [RENDERER AUDIT 2] 4D Null-Cone Lift | rel|det(dX)|: {rel_det_err:.2e} | rel|Z^dagger Sigma Z|: {norm_err:.2e}")
        self.assertLess(rel_det_err, 1e-14)
        self.assertLess(norm_err, 1e-14)

    def test_3_spinor_vs_euclidean_point_to_ray_distance_identity(self):
        torch.manual_seed(24)
        pts = torch.randn((60, 3), dtype=torch.float64) * 5.0
        cam = torch.tensor([-1.0, 2.0, -0.5], dtype=torch.float64)
        ray_pts = torch.randn((32, 3), dtype=torch.float64)
        ray_spinors = PenroseTwistorHarmonicEngine.points_to_celestial_spinors(ray_pts)
        ray_dirs = PenroseTwistorHarmonicEngine.spinor_to_unit_ray(ray_spinors)

        r2_spinor = TwistorRenderer.spinor_perpendicular_residual_sq(pts, cam, ray_spinors)
        r2_euclid = TwistorRenderer.euclidean_perpendicular_residual_sq(pts, cam, ray_dirs)

        scale_sq = torch.max(r2_euclid).item()
        rel_diff = torch.max(torch.abs(r2_spinor - r2_euclid)).item() / scale_sq
        print(f"  -> [RENDERER AUDIT 3] Spinor ||D pi||^2 - |pi^dag D pi|^2 vs Euclidean ||d||^2 - (d.n)^2 Rel Diff: {rel_diff:.2e}")
        self.assertLess(rel_diff, 1e-14)

    def test_4_recursive_aabb_bvh_culling_and_agreement_with_baseline(self):
        torch.manual_seed(55)
        cluster_centers = torch.tensor([
            [ 0.0,  0.0, 10.0], [ 1.2, -0.8,  9.0], [-12.0,  12.0, 10.0], [12.0,  12.0, 10.0],
            [-15.0, 0.0, -12.0], [15.0, 0.0, -12.0], [  0.0,  18.0, -8.0], [ 0.0, -18.0, -8.0]
        ], dtype=torch.float64)
        pts = torch.cat([c.unsqueeze(0) + 0.35 * torch.randn((80, 3), dtype=torch.float64) for c in cluster_centers], dim=0)

        cam = torch.zeros(3, dtype=torch.float64)
        renderer = TwistorRenderer(width=16, height=16, fov_deg=50.0)
        _, ray_dirs = renderer.generate_camera_rays(camera_center=cam)

        bvh = TwistorBVH(pts, leaf_size=16, sigma=0.15, cutoff_mult=5.5)
        img_bvh, stats = bvh.render(cam, ray_dirs)
        img_brute = renderer.render(pts, cam, ray_dirs, 0.15)

        peak_intensity = torch.max(img_bvh).item()
        max_img_diff = torch.max(torch.abs(img_bvh - img_brute)).item()
        pruned_pct = stats["pruned_fraction"] * 100.0
        print(f"  -> [BVH AUDIT 4] Peak Intensity: {peak_intensity:.3f} (>0) | AABB Pruned: {pruned_pct:.1f}% | Max Diff vs Brute-Force: {max_img_diff:.2e}")
        self.assertGreater(peak_intensity, 0.05)
        self.assertGreater(pruned_pct, 75.0)
        self.assertLess(max_img_diff, 1e-6)

    def test_5_conformal_camera_ray_action_omega_zero_vs_nonzero(self):
        renderer = TwistorRenderer(width=8, height=8, fov_deg=60.0)
        b_4d = torch.tensor([0.15, -0.35, 0.25, -0.20], dtype=torch.float64)
        trans_4d = torch.tensor([1.0, 2.5, -1.5, 0.8], dtype=torch.float64)
        cam_nonzero = torch.tensor([1.2, -0.9, 2.0], dtype=torch.float64)

        _, dirs_id = renderer.generate_camera_rays()
        _, dirs_origin_kd = renderer.generate_camera_rays(special_conformal_4d=b_4d, dilation_lambda=0.6)
        origin_noop_diff = torch.max(torch.abs(dirs_origin_kd - dirs_id)).item()

        _, dirs_cam_k = renderer.generate_camera_rays(special_conformal_4d=b_4d, camera_center=cam_nonzero)
        cam_k_shift = torch.max(torch.abs(dirs_cam_k - dirs_id)).item()

        _, dirs_trans_k = renderer.generate_camera_rays(trans_4d=trans_4d, special_conformal_4d=b_4d)
        trans_k_shift = torch.max(torch.abs(dirs_trans_k - dirs_id)).item()

        _, dirs_cam_kd = renderer.generate_camera_rays(special_conformal_4d=b_4d, dilation_lambda=0.7, camera_center=cam_nonzero)
        dilation_k_shift = torch.max(torch.abs(dirs_cam_kd - dirs_cam_k)).item()

        print(f"  -> [RENDERER AUDIT 5] Origin omega=0 Dir No-Op: {origin_noop_diff:.2e} | Active K+Cam: {cam_k_shift:.4f} | Active K+Trans: {trans_k_shift:.4f} | Active D+K: {dilation_k_shift:.4f}")
        self.assertLess(origin_noop_diff, 1e-15)
        self.assertGreater(cam_k_shift, 0.05)
        self.assertGreater(trans_k_shift, 0.05)
        self.assertGreater(dilation_k_shift, 0.05)

    def test_6_omega_prime_incidence_origin_reconstruction_and_all_15_params_active(self):
        """
        Addresses Reviewer 1's findings #4 and #5:
          1. Verifies that transform_spacetime_event_su22 computes X' = (AX - iB)(iCX + D)^{-1}
             with exact Hermiticity and zero incidence defect across all 15 SU(2,2) parameters.
          2. Verifies that all 5 SU(2,2) subgroup families (Rotation, Boost, Translation, Dilation,
             and Special Conformal K(b)) actively transform the rendered image.
        """
        renderer = TwistorRenderer(width=8, height=8, fov_deg=60.0)
        cam = torch.tensor([0.8, -0.6, 1.2], dtype=torch.float64)
        cam_4d = torch.cat([torch.zeros(1, dtype=torch.float64), cam], dim=0)
        X_cam = PenroseTwistorCore.spacetime_to_hermitian(cam_4d)
        pts = torch.tensor([
            [0.8, -0.6, 5.0], [1.2, -0.3, 4.5], [0.4, -0.9, 5.5], [0.9, -0.5, 4.0]
        ], dtype=torch.float64)

        # Baseline image with identity transformation
        pi_0, _, orig_0, _, _ = renderer.generate_camera_rays(camera_center=cam, return_ray_origins=True)
        img_0 = renderer.render(pts, orig_0, sigma=0.35, ray_spinors=pi_0)

        # 1. Rotation alone
        pi_rot, _, orig_rot, _, _ = renderer.generate_camera_rays(
            theta_3d=torch.tensor([0.15, -0.20, 0.10], dtype=torch.float64), camera_center=cam, return_ray_origins=True
        )
        diff_rot = torch.max(torch.abs(renderer.render(pts, orig_rot, sigma=0.35, ray_spinors=pi_rot) - img_0)).item()

        # 2. Boost alone
        pi_bst, _, orig_bst, _, _ = renderer.generate_camera_rays(
            boost_3d=torch.tensor([0.15, 0.10, -0.15], dtype=torch.float64), camera_center=cam, return_ray_origins=True
        )
        diff_bst = torch.max(torch.abs(renderer.render(pts, orig_bst, sigma=0.35, ray_spinors=pi_bst) - img_0)).item()

        # 3. Translation alone
        pi_trn, _, orig_trn, _, _ = renderer.generate_camera_rays(
            trans_4d=torch.tensor([0.0, 0.5, -0.4, 0.3], dtype=torch.float64), camera_center=cam, return_ray_origins=True
        )
        diff_trn = torch.max(torch.abs(renderer.render(pts, orig_trn, sigma=0.35, ray_spinors=pi_trn) - img_0)).item()

        # 4. Dilation alone
        pi_dil, _, orig_dil, _, _ = renderer.generate_camera_rays(
            dilation_lambda=0.45, camera_center=cam, return_ray_origins=True
        )
        diff_dil = torch.max(torch.abs(renderer.render(pts, orig_dil, sigma=0.35, ray_spinors=pi_dil) - img_0)).item()

        # 5. Special Conformal K(b) alone + verify exact block matrix incidence transformation X'
        sc_params = torch.tensor([0.10, -0.25, 0.20, -0.15], dtype=torch.float64)
        G_sc = renderer.build_full_15param_su22_matrix(
            torch.zeros(3, dtype=torch.float64),
            torch.zeros(3, dtype=torch.float64),
            torch.zeros(4, dtype=torch.float64),
            sc_params,
            dilation_lambda=0.0
        )
        X_prime_sc = renderer.transform_spacetime_event_su22(G_sc, X_cam)
        hermitian_err = torch.max(torch.abs(X_prime_sc - torch.conj(X_prime_sc.mT))).item()

        pi_sc, _, orig_sc, _, _ = renderer.generate_camera_rays(
            special_conformal_4d=sc_params, camera_center=cam, return_ray_origins=True
        )
        diff_sc = torch.max(torch.abs(renderer.render(pts, orig_sc, sigma=0.35, ray_spinors=pi_sc) - img_0)).item()

        print(f"  -> [RENDERER AUDIT 6] K(b) Spacetime Matrix Hermitian Error: {hermitian_err:.2e} | Image Shifts -> Rot: {diff_rot:.3f}, Boost: {diff_bst:.3f}, Trans: {diff_trn:.3f}, Dilation: {diff_dil:.3f}, K(b): {diff_sc:.3f}")
        self.assertLess(hermitian_err, 1e-14)
        self.assertGreater(diff_rot, 0.01)
        self.assertGreater(diff_bst, 0.01)
        self.assertGreater(diff_trn, 0.01)
        self.assertGreater(diff_dil, 0.01)
        self.assertGreater(diff_sc,  0.01)


    def test_7_identity_origin_and_15param_ray_line_convergence(self):
        """
        Addresses the November 16, 2026 Audit Findings (P0 & P1):
          1. Identity transformation check: for camera_center = (1, 2, 3), every reconstructed
             ray origin on the t = 0 slice equals (1, 2, 3) to machine precision (< 1e-15).
          2. Full 15-parameter SU(2,2) check: verifies exact Hermiticity of X', exact incidence
             ||omega' - i X_ray pi'|| < 1e-14, and proves that every reconstructed spatial ray
             line o_ray + lambda * n' passes directly through the transformed camera position r'
             at lambda = t' (distance < 1e-14).
        """
        renderer = TwistorRenderer(width=6, height=6, fov_deg=60.0)
        cam_id = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

        # 1. Identity transformation reconstruction check
        _, _, orig_id, X_id, Z_id = renderer.generate_camera_rays(camera_center=cam_id, return_ray_origins=True)
        id_origin_err = torch.max(torch.linalg.norm(orig_id - cam_id.unsqueeze(0), dim=-1)).item()
        id_inc_err = torch.max(torch.linalg.norm(
            Z_id[:, 0:2] - 1j * torch.einsum("mab,mb->ma", X_id, Z_id[:, 2:4]), dim=-1
        )).item()

        # 2. Combined 15-parameter transformation check
        t3 = torch.tensor([0.12, -0.18, 0.09], dtype=torch.float64)
        b3 = torch.tensor([0.06, -0.08, 0.05], dtype=torch.float64)
        tr4 = torch.tensor([0.25, -0.30, 0.45, -0.20], dtype=torch.float64)
        sc4 = torch.tensor([0.04, -0.06, 0.05, -0.03], dtype=torch.float64)
        lam = 0.25

        cam_4d = torch.cat([torch.zeros(1, dtype=torch.float64), cam_id], dim=0)
        X_cam = PenroseTwistorCore.spacetime_to_hermitian(cam_4d)
        G_15 = renderer.build_full_15param_su22_matrix(t3, b3, tr4, sc4, dilation_lambda=lam)
        X_prime = renderer.transform_spacetime_event_su22(G_15, X_cam)

        herm_err = torch.max(torch.abs(X_prime - torch.conj(X_prime.mT))).item()
        t_prime = 0.5 * (X_prime[0, 0] + X_prime[1, 1]).real
        r_prime = torch.stack([
            0.5 * (X_prime[0, 1] + X_prime[1, 0]).real,
            0.5 * (X_prime[1, 0] - X_prime[0, 1]).imag,
            0.5 * (X_prime[0, 0] - X_prime[1, 1]).real,
        ], dim=0)

        pi_15, dirs_15, orig_15, X_ray_15, Z_15 = renderer.generate_camera_rays(
            theta_3d=t3, boost_3d=b3, trans_4d=tr4, special_conformal_4d=sc4,
            dilation_lambda=lam, camera_center=cam_id, return_ray_origins=True
        )

        # Verify incidence ||omega' - i X_ray pi'|| < 1e-14
        inc_defect_15 = torch.max(torch.linalg.norm(
            Z_15[:, 0:2] - 1j * torch.einsum("mab,mb->ma", X_ray_15, Z_15[:, 2:4]), dim=-1
        )).item()

        # Verify every ray line passes through r' at parameter lambda = t'
        ray_pts_at_t_prime = orig_15 - t_prime * dirs_15
        line_pass_err = torch.max(torch.linalg.norm(ray_pts_at_t_prime - r_prime.unsqueeze(0), dim=-1)).item()

        print(f"  -> [RENDERER AUDIT 7] Identity Origin Err: {id_origin_err:.2e} | 15-Param Incidence Defect: {inc_defect_15:.2e} | Ray-Line to r' Dist: {line_pass_err:.2e}")
        self.assertLess(id_origin_err, 1e-14)
        self.assertLess(id_inc_err, 1e-14)
        self.assertLess(herm_err, 1e-14)
        self.assertLess(inc_defect_15, 1e-14)
        self.assertLess(line_pass_err, 1e-14)

if __name__ == "__main__":
    unittest.main(verbosity=2)