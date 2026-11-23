"""
Tests for Bridge Lemma 5 (stereographic screen-gauge covariance, inverse transition,
spin-weight scaling, Hermitian norm invariance, equatorial handoffs, and polar chart selection).
"""

from __future__ import annotations

import hashlib
import pathlib
import sys
import unittest
import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from twistor import congruence as cg

EXPECTED_LEGACY_SHA256 = "8809F3AF1D5BBB43202128C740306402D9BFF264911146CB5DB974EBF9716217"


class TestBridgeLemma5ScreenGauge(unittest.TestCase):
    def test_1_frozen_legacy_hash_intact(self):
        digest = hashlib.sha256((REPO_ROOT / "palatial_gpo_ray_optics.py").read_bytes()).hexdigest().upper()
        self.assertEqual(digest, EXPECTED_LEGACY_SHA256)

    def test_2_transition_inverse_and_spin_weight_covariance(self):
        b = 1.25
        pts = torch.tensor([
            [1.6, -1.2, -0.4],
            [-0.9, 1.1, 0.7],
            [0.5, 0.6, -1.3],
        ], dtype=torch.float64)
        w = cg.principal_w(pts, b, sheet_sign=+1.0)
        k = cg.tangent_field(pts, w, b)
        kx, ky, kz = k[:, 0], k[:, 1], k[:, 2]

        zeta = torch.complex(kx, ky) / (1.0 + kz)
        eta = torch.complex(kx, -ky) / (1.0 - kz)
        self.assertLess(float(torch.max(torch.abs(zeta * eta - 1.0)).item()), 1e-13)

        phi_sn = torch.conj(zeta) / zeta
        phi_ns_at_eta = torch.conj(eta) / eta
        inv_err = float(torch.max(torch.abs(phi_ns_at_eta - torch.conj(phi_sn))).item())
        self.assertLess(inv_err, 1e-14)
        self.assertLess(float(torch.max(torch.abs(torch.abs(phi_sn) - 1.0)).item()), 1e-14)

        m_s = cg.screen_vector_south(zeta)
        m_n = cg.screen_vector_north(eta)
        handoff_res = float(torch.max(torch.linalg.norm(m_n - phi_sn.unsqueeze(-1) * m_s, dim=-1)).item())
        self.assertLess(handoff_res, 1e-14)

        u_vec = torch.tensor([0.6 + 0.2j, -0.3 + 0.5j, 0.4 - 0.1j], dtype=torch.complex128)
        z_s = torch.sum(m_s * u_vec.unsqueeze(0), dim=-1)
        z_n = torch.sum(m_n * u_vec.unsqueeze(0), dim=-1)

        self.assertLess(float(torch.max(torch.abs(z_n - phi_sn * z_s)).item()), 1e-14)
        self.assertLess(float(torch.max(torch.abs(torch.conj(z_n) - torch.conj(phi_sn) * torch.conj(z_s))).item()), 1e-14)
        self.assertLess(float(torch.max(torch.abs((z_n ** 2) - (phi_sn ** 2) * (z_s ** 2))).item()), 1e-14)
        self.assertLess(float(torch.max(torch.abs((z_n * torch.conj(z_n)) - (z_s * torch.conj(z_s)))).item()), 1e-14)

    def test_3_hermitian_norm_invariance_and_fixed_sheet_distinction(self):
        b = 1.25
        pos = torch.tensor([0.3, -0.4, 0.5], dtype=torch.float64)
        w_plus = cg.principal_w(pos, b, sheet_sign=+1.0)
        w_minus = cg.principal_w(pos, b, sheet_sign=-1.0)

        k_plus = cg.tangent_field(pos, w_plus, b)
        k_minus = cg.tangent_field(pos, w_minus, b)
        self.assertGreater(float(torch.linalg.norm(k_plus - k_minus).item()), 0.5)

        zeta = torch.complex(k_plus[0], k_plus[1]) / (1.0 + k_plus[2])
        eta = torch.complex(k_plus[0], -k_plus[1]) / (1.0 - k_plus[2])
        phi = torch.conj(zeta) / zeta

        m_s = cg.screen_vector_south(zeta)
        m_n = cg.screen_vector_north(eta)
        self.assertGreater(float(torch.linalg.norm(m_n - m_s).item()), 0.1)
        herm_s = complex(torch.sum(torch.conj(m_s) * m_s).item())
        herm_n = complex(torch.sum(torch.conj(m_n) * m_n).item())
        self.assertLess(abs(herm_n - herm_s), 1e-14)
        self.assertLess(abs(herm_n - (abs(phi) ** 2) * herm_s), 1e-14)

    def test_4_equatorial_handoff_and_polar_chart_selection(self):
        b = 1.25
        zs = torch.linspace(-2.5, 2.5, 501, dtype=torch.float64)
        pts = torch.stack([torch.full_like(zs, 1.6), torch.full_like(zs, -1.2), zs], dim=-1)
        w_list = [cg.principal_w(pts[0], b, sheet_sign=+1.0)]
        for i in range(1, len(pts)):
            w_list.append(cg.continue_sheet_step(pts[i], w_list[-1], b))
        w_cont = torch.stack(w_list, dim=0)

        atlas = cg.bounded_chart_and_screen(pts, w_cont, b)
        switches = torch.where(atlas["use_south"][:-1] != atlas["use_south"][1:])[0]
        self.assertEqual(len(switches), 1)
        idx = int(switches[0].item())

        max_selected_abs = float(torch.max(torch.abs(atlas["coord"])).item())
        self.assertLessEqual(max_selected_abs, 1.0 + 1e-14)

        k_sw = atlas["k"][idx]
        zeta_sw = torch.complex(k_sw[0], k_sw[1]) / (1.0 + k_sw[2])
        eta_sw = torch.complex(k_sw[0], -k_sw[1]) / (1.0 - k_sw[2])
        phi_sw = torch.conj(zeta_sw) / zeta_sw
        m_s_sw = cg.screen_vector_south(zeta_sw)
        m_n_sw = cg.screen_vector_north(eta_sw)
        self.assertLess(float(torch.linalg.norm(m_n_sw - phi_sw * m_s_sw).item()), 1e-14)

        polar_positions = torch.tensor([
            [1e-7, 0.0, 1.0],
            [1e-7, 0.0, -1.0],
        ], dtype=torch.float64)
        for pos in polar_positions:
            for sign in (+1.0, -1.0):
                w = cg.principal_w(pos, b, sheet_sign=sign)
                atlas_p = cg.bounded_chart_and_screen(pos.unsqueeze(0), w.unsqueeze(0), b)
                kz = float(atlas_p["k"][0, 2].item())
                selected_south = bool(atlas_p["use_south"][0].item())
                self.assertEqual(selected_south, kz >= 0.0)
                self.assertLessEqual(float(torch.abs(atlas_p["coord"][0]).item()), 1.0 + 1e-14)
                self.assertTrue(bool(torch.isfinite(atlas_p["m"]).all().item()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
