from __future__ import annotations

import hashlib
import math
import pathlib
import sys
import unittest
import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from twistor import congruence as cg
from twistor.congruence_adapter import evaluate_congruence_along_ray

EXPECTED_LEGACY_SHA256 = "8809F3AF1D5BBB43202128C740306402D9BFF264911146CB5DB974EBF9716217"


class TestCongruenceAdapterMigration(unittest.TestCase):
    def test_1_frozen_legacy_hash_unchanged(self):
        gpo_path = REPO_ROOT / "palatial_gpo_ray_optics.py"
        digest = hashlib.sha256(gpo_path.read_bytes()).hexdigest().upper()
        self.assertEqual(digest, EXPECTED_LEGACY_SHA256)

    def test_2_continued_vs_principal_across_lifted_disk(self):
        b = 1.25
        x0 = torch.tensor([0.3, -0.4, 0.5], dtype=torch.float64)
        w0 = cg.principal_w(x0, b, sheet_sign=-1.0)
        k0 = cg.tangent_field(x0, w0, b)
        s_star = float((-w0.real).item())
        s_vals = torch.linspace(0.0, 2.2 * s_star, 221, dtype=torch.float64)

        cont = evaluate_congruence_along_ray(x0, k0, s_vals, b=b, sheet_mode="continued", initial_sheet_sign=-1.0)
        prin = evaluate_congruence_along_ray(x0, k0, s_vals, b=b, sheet_mode="principal", initial_sheet_sign=-1.0)

        self.assertIsNotNone(cont.min_step_margin)
        self.assertGreater(cont.min_step_margin, 0.0)
        self.assertIsNone(prin.min_step_margin)

        # Continued path stays smooth across the disk cut; principal path has a branch jump
        cont_max_dk = float(torch.max(torch.linalg.norm(cont.k[1:] - cont.k[:-1], dim=-1)).item())
        prin_max_dk = float(torch.max(torch.linalg.norm(prin.k[1:] - prin.k[:-1], dim=-1)).item())
        self.assertLess(cont_max_dk, 0.05)
        self.assertGreater(prin_max_dk, 0.5)

    def test_3_equatorial_handoff_and_z_axis_regularity(self):
        b = 1.25
        # Equatorial sweep outside spanning disk: r^2 = 1.6^2 + (-1.2)^2 = 4.0 > 1.25^2
        eq = evaluate_congruence_along_ray(
            origin=torch.tensor([1.6, -1.2, -2.5], dtype=torch.float64),
            direction=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
            s_values=torch.linspace(0.0, 5.0, 501, dtype=torch.float64),
            b=b,
            sheet_mode="continued",
            initial_sheet_sign=1.0,
        )
        switches = torch.where(eq.use_south[:-1] != eq.use_south[1:])[0]
        self.assertEqual(len(switches), 1)
        self.assertLess(float(torch.max(torch.abs(torch.linalg.norm(eq.k, dim=-1) - 1.0)).item()), 1e-14)

        # z-axis sweep remains finite everywhere
        z_ray = evaluate_congruence_along_ray(
            origin=torch.tensor([0.0, 0.0, -1.0], dtype=torch.float64),
            direction=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
            s_values=torch.linspace(0.0, 2.0, 201, dtype=torch.float64),
            b=b,
            sheet_mode="continued",
            initial_sheet_sign=1.0,
        )
        self.assertTrue(torch.isfinite(z_ray.k).all().item())
        self.assertTrue(torch.isfinite(z_ray.rho).all().item())


if __name__ == "__main__":
    unittest.main(verbosity=2)