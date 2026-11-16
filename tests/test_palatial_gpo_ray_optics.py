import math
import unittest
import torch
from palatial_gpo_ray_optics import PalatialGPORayOptics


class TestPalatialGPORayOptics(unittest.TestCase):
    """Formal unit tests for Project Argon CP^4 Ray Bundles and GPO Robot Vision."""

    def test_1_sigma5_strictly_real_float64_and_hermitian_symmetry(self):
        """Verifies SIGMA_5 is strictly torch.float64 and equals its transpose."""
        self.assertEqual(PalatialGPORayOptics.SIGMA_5.dtype, torch.float64)
        self.assertFalse(PalatialGPORayOptics.SIGMA_5.is_complex())
        sym_err = float(torch.max(torch.abs(PalatialGPORayOptics.SIGMA_5 - PalatialGPORayOptics.SIGMA_5.T)).item())
        self.assertEqual(sym_err, 0.0)
        print(f"  -> [ARGON RAY 1] SIGMA_5 dtype: {PalatialGPORayOptics.SIGMA_5.dtype} | Symmetry Err: {sym_err:.2e}")

    def test_2_cp4_ray_bundle_triadic_partition_of_unity_and_u1_gauge(self):
        """Verifies c_G + c_P + c_O == 1.0 to 1e-15 and exact U(1) ray phase invariance."""
        gen = torch.Generator(device="cpu").manual_seed(2026)
        origins = torch.randn((512, 3), dtype=torch.float64, generator=gen)
        dirs = torch.randn((512, 3), dtype=torch.float64, generator=gen)
        horizon = torch.rand((512, 1), dtype=torch.float64, generator=gen) * 0.5

        Z5 = PalatialGPORayOptics.lift_rays_to_cp4(origins, dirs, curvature_horizon=horizon)
        out1 = PalatialGPORayOptics.evaluate_robot_vision_gpo_rgb(Z5)
        triad_err = float(torch.max(torch.abs(out1["triad_sum"] - 1.0)).item())
        self.assertLess(triad_err, 1e-15)

        phases = torch.exp(1j * (2.0 * math.pi * torch.rand((512, 1), dtype=torch.float64, generator=gen)))
        out2 = PalatialGPORayOptics.evaluate_robot_vision_gpo_rgb(Z5 * phases)
        rgb_diff = float(torch.max(torch.abs(out1["rgb_channels"] - out2["rgb_channels"])).item())
        self.assertLess(rgb_diff, 1e-15)
        print(f"  -> [ARGON RAY 2] Ray Triad Sum Err: {triad_err:.2e} | U(1) RGB Gauge Diff: {rgb_diff:.2e}")

    def test_3_equatorial_null_origin_maximizes_green_channel(self):
        """Verifies unperturbed origin rays (omega aligned with pi, horizon=0) lock onto the Green band."""
        origins = torch.zeros((64, 3), dtype=torch.float64)
        dirs = torch.randn((64, 3), dtype=torch.float64)
        Z5 = PalatialGPORayOptics.lift_rays_to_cp4(origins, dirs, curvature_horizon=None)
        out = PalatialGPORayOptics.evaluate_robot_vision_gpo_rgb(Z5)

        green_err = float(torch.max(torch.abs(out["c_green"] - 1.0)).item())
        purple_err = float(torch.max(torch.abs(out["c_purple"])).item())
        orange_err = float(torch.max(torch.abs(out["c_orange"])).item())
        self.assertLess(green_err, 1e-15)
        self.assertLess(purple_err, 1e-15)
        self.assertLess(orange_err, 1e-15)
        print(f"  -> [ARGON RAY 3] Equatorial Ground Truth | Green: {1.0 - green_err:.6f} | Purple: {purple_err:.2e} | Orange: {orange_err:.2e}")


if __name__ == "__main__":
    unittest.main()
