import unittest
import torch
from twistor.renderer import TwistorRenderer
from twistor.bvh import TwistorBVH
from twistor.penrose_core import PenroseTwistorCore
from twistor.harmonics import PenroseTwistorHarmonicEngine

class TestRendererAndBVHRemediation(unittest.TestCase):

    def test_1_full_15_parameter_su22_group_closure(self):
        """
        Verifies that the full 15-parameter SU(2,2) transformation G = K(b) @ D(lambda) @ P(Lambda, T)
        (3 rotations + 3 boosts + 4 translations + 4 special conformal K(b) + 1 dilation lambda)
        preserves the SU(2,2) metric G^dagger Sigma G = Sigma to machine precision.
        """
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
        """
        Verifies that lift_points_to_null_cone_twistors places all points strictly on the
        future light cone relative to the camera (relative |det(dX)| / ||d||^2 < 1e-15).
        """
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
        """
        Verifies the mathematical review's point-to-line identity:
            ||D pi||^2 - |<pi | D | pi>|^2 == ||d||^2 - (d . n)^2
        where D = d_k sigma^k is the traceless Hermitian spatial displacement matrix.
        """
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
        """
        Verifies that TwistorBVH builds a genuine recursive AABB tree, renders non-zero
        in-frustum clusters accurately against the brute-force TwistorRenderer baseline,
        and prunes >75% of ray-point evaluations via Kay-Kajiya slab intersection tests.
        """
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
        img_brute = renderer.render(pts, cam, ray_dirs, sigma=0.15)

        peak_intensity = torch.max(img_bvh).item()
        max_img_diff = torch.max(torch.abs(img_bvh - img_brute)).item()
        pruned_pct = stats["pruned_fraction"] * 100.0
        print(f"  -> [BVH AUDIT 4] Peak Intensity: {peak_intensity:.3f} (>0) | AABB Pruned: {pruned_pct:.1f}% | Max Diff vs Brute-Force: {max_img_diff:.2e}")
        self.assertGreater(peak_intensity, 0.05, "In-frustum clusters at z=10 (1/(1+z^2) ~ 0.01/pt) must produce non-zero rendered intensity!")
        self.assertGreater(pruned_pct, 75.0, "BVH must prune at least 75% of ray-point evaluations!")
        self.assertLess(max_img_diff, 1e-6)

    def test_5_conformal_camera_ray_action_omega_zero_vs_nonzero(self):
        """
        Explicitly tests the algebraic properties of generate_camera_rays under SU(2,2) subgroups:
          1. When omega = 0 (camera_center=None, trans_4d=0), D(lambda) and K(b) act via projective
             rescaling pi' = e^{-lambda/2} pi on CP^1, leaving unit rays n unchanged (diff == 0).
          2. When omega != 0 (via camera_center != 0 or trans_4d != 0), P(I, T) shifts omega^A = i T^{AA'} pi_{A'},
             so subsequent K(b) (pi' = e^{-lambda/2} pi - i e^{lambda/2} B omega) and D(lambda) both
             induce non-trivial projective ray deformations on CP^1.
        """
        renderer = TwistorRenderer(width=8, height=8, fov_deg=60.0)
        b_4d = torch.tensor([0.15, -0.35, 0.25, -0.20], dtype=torch.float64)
        trans_4d = torch.tensor([1.0, 2.5, -1.5, 0.8], dtype=torch.float64)
        cam_nonzero = torch.tensor([1.2, -0.9, 2.0], dtype=torch.float64)

        # Case 1: omega = 0 (origin light cone) -> K(b) and D(lambda) are projective no-ops on CP^1
        _, dirs_id = renderer.generate_camera_rays()
        _, dirs_origin_kd = renderer.generate_camera_rays(special_conformal_4d=b_4d, dilation_lambda=0.6)
        origin_noop_diff = torch.max(torch.abs(dirs_origin_kd - dirs_id)).item()

        # Case 2: omega != 0 via camera_center != 0 and K(b) != 0 -> non-trivial ray warp
        _, dirs_cam_k = renderer.generate_camera_rays(special_conformal_4d=b_4d, camera_center=cam_nonzero)
        cam_k_shift = torch.max(torch.abs(dirs_cam_k - dirs_id)).item()

        # Case 3: omega != 0 via spacetime translation T != 0 followed by K(b) -> non-trivial ray warp
        _, dirs_trans_k = renderer.generate_camera_rays(trans_4d=trans_4d, special_conformal_4d=b_4d)
        trans_k_shift = torch.max(torch.abs(dirs_trans_k - dirs_id)).item()

        # Case 4: Dilation D(lambda) coupled with K(b) at omega != 0 changes the relative weight e^{lambda} B omega
        _, dirs_cam_kd = renderer.generate_camera_rays(special_conformal_4d=b_4d, dilation_lambda=0.7, camera_center=cam_nonzero)
        dilation_k_shift = torch.max(torch.abs(dirs_cam_kd - dirs_cam_k)).item()

        print(f"  -> [RENDERER AUDIT 5] Origin omega=0 No-Op Diff: {origin_noop_diff:.2e} | Active K(b)+Cam Shift: {cam_k_shift:.4f} | Active K(b)+Trans Shift: {trans_k_shift:.4f} | Active D(lambda)+K(b) Shift: {dilation_k_shift:.4f}")
        self.assertLess(origin_noop_diff, 1e-15)
        self.assertGreater(cam_k_shift, 0.05)
        self.assertGreater(trans_k_shift, 0.05)
        self.assertGreater(dilation_k_shift, 0.05)

if __name__ == "__main__":
    unittest.main(verbosity=2)