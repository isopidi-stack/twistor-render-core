"""
Verification suite for Bridge Lemma 6 conditional interval-certified ray
continuation (exact integer-verified directed-rounding Decimal square-bound
checks, direct _interval_dist_point_to_segment zero-crossing check,
approximately consistent Robinson-ray rejection under step mismatch,
exact stored-float64 Fraction norm/discriminant checks, outward-rounded
complex distance upper bounds, large-guard rejection, and strict dual-sign
certification).
"""

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction
import hashlib
import math
import pathlib
import random
import sys
import unittest
from unittest.mock import patch
import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from twistor import congruence as cg
from twistor.certified_ray import _complex_distance_upper, _vec3_norm_interval, propagate_ray_step_enclosure
from twistor.certified_selector import (
    _certified_phase_inner_product_sign,
    _interval_dist_point_to_segment,
    evaluate_certified_step,
)
from twistor.interval_bounds import (
    ComplexRectInterval,
    RealInterval,
    square_geq_decimal_exact,
    square_leq_decimal_exact,
    sqrt_ceil_decimal,
    sqrt_floor_decimal,
)

EXPECTED_LEGACY_SHA256 = "8809F3AF1D5BBB43202128C740306402D9BFF264911146CB5DB974EBF9716217"


def _exact_fraction_delta_parts(
    x_f: float, y_f: float, z_f: float, b_f: float
) -> tuple[Fraction, Fraction]:
    x = Fraction(Decimal.from_float(float(x_f)))
    y = Fraction(Decimal.from_float(float(y_f)))
    z = Fraction(Decimal.from_float(float(z_f)))
    b = Fraction(Decimal.from_float(float(b_f)))
    return x * x + y * y + z * z - b * b, Fraction(2) * b * z


class TestBridgeLemma6CertifiedRay(unittest.TestCase):
    def test_1_frozen_legacy_hash_intact(self):
        digest = hashlib.sha256((REPO_ROOT / "palatial_gpo_ray_optics.py").read_bytes()).hexdigest().upper()
        self.assertEqual(digest, EXPECTED_LEGACY_SHA256)

    def test_2_exact_directed_sqrt_segment_distance_and_consistent_robinson_step(self):
        r_lo_3 = sqrt_floor_decimal(Decimal(2), prec=3)
        r_hi_3 = sqrt_ceil_decimal(Decimal(2), prec=3)
        self.assertEqual(r_lo_3, Decimal("1.41"))
        self.assertEqual(r_hi_3, Decimal("1.42"))
        self.assertTrue(square_leq_decimal_exact(r_lo_3, Decimal(2)))
        self.assertTrue(square_geq_decimal_exact(r_hi_3, Decimal(2)))

        huge_iv = RealInterval.point_decimal(Decimal("1e999999999999999999"))
        self.assertFalse(huge_iv.sqr(prec=30).is_valid_finite)

        z_re = Decimal("-1.00000000000000000000000000000000000000000000000000000000000000000000000000000000000001")
        z_im = Decimal(0)
        rect = ComplexRectInterval(
            re=RealInterval.from_bounds_decimal(Decimal(0), Decimal("0.5")),
            im=RealInterval.from_bounds_decimal(Decimal(0), Decimal(0)),
        )
        dist_iv = _complex_distance_upper(z_re, z_im, rect, prec=30)
        self.assertTrue(dist_iv.is_valid_finite)
        assert dist_iv.hi is not None
        self.assertGreater(dist_iv.hi, Decimal("1.5"))

        neg_alpha_iv = RealInterval.point_decimal(Decimal("1.0"), "neg_alpha")
        d_short = _interval_dist_point_to_segment(neg_alpha_iv, Decimal(0), Decimal("0.5"), prec=80)
        d_long = _interval_dist_point_to_segment(neg_alpha_iv, Decimal(0), Decimal("1.5"), prec=80)
        assert d_short.lo is not None and d_long.lo is not None
        self.assertGreater(d_short.lo, Decimal(0))
        self.assertEqual(d_long.lo, Decimal(0))

        k_rat = torch.tensor([0.6, 0.8, 0.0], dtype=torch.float64)
        kx_q = Fraction.from_float(float(k_rat[0].item()))
        ky_q = Fraction.from_float(float(k_rat[1].item()))
        kz_q = Fraction.from_float(float(k_rat[2].item()))
        norm_sq_stored = kx_q * kx_q + ky_q * ky_q + kz_q * kz_q
        norm_iv = _vec3_norm_interval(k_rat, prec=80)
        assert norm_iv.lo is not None and norm_iv.hi is not None
        self.assertLessEqual(Fraction(norm_iv.lo) ** 2, norm_sq_stored)
        self.assertLessEqual(norm_sq_stored, Fraction(norm_iv.hi) ** 2)

        # Approximately geometrically consistent Robinson ray up to float64 rounding at b=1.0
        b = 1.0
        pos_approx = torch.tensor([math.sqrt(2.0), 0.0, 0.0], dtype=torch.float64)
        w0_real_neg = torch.tensor(-1.0 + 0.0j, dtype=torch.complex128)
        k_rob = cg.tangent_field(pos_approx, w0_real_neg, b)

        dec_mismatch = evaluate_certified_step(
            pos_hat=pos_approx, k_hat=k_rob, prev_w_hat=w0_real_neg,
            h_hat=0.5, h_exact=1.5, b=b,
            eps_x_curr=0.0, eps_k_curr=0.0, eps_w_curr=1e-9,
            fl_advance_err=1e-15, fl_delta_err=1e-12, fl_sqrt_err=1e-15, fl_pred_add_err=1e-15,
            ip_roundoff=1e-14, cmp_roundoff=1e-14, guard=1e-8,
            min_clearance=1e-6, max_ray_drift=10.0,
        )
        self.assertFalse(dec_mismatch.certified_accept)
        self.assertIn(
            dec_mismatch.reason,
            ("discriminant_enclosure_contains_or_touches_zero", "segment_clearance_not_certified"),
        )

    def test_3_exact_next_delta_distance_and_fraction_oracle(self):
        rng = random.Random(20261124)
        for _ in range(25):
            a_f = rng.uniform(-5.0, 5.0)
            b_f = rng.uniform(-5.0, 5.0)
            r_a = rng.uniform(0.0, 0.25)
            r_b = rng.uniform(0.0, 0.25)
            iv_a = RealInterval.expand_by_radius_decimal(RealInterval.point_float(a_f), Decimal.from_float(r_a), prec=80)
            iv_b = RealInterval.expand_by_radius_decimal(RealInterval.point_float(b_f), Decimal.from_float(r_b), prec=80)
            iv_prod = iv_a.mul(iv_b, prec=80)
            self.assertTrue(iv_prod.is_valid_finite)
            assert iv_prod.lo is not None and iv_prod.hi is not None

            fa_lo, fa_hi = Fraction(iv_a.lo), Fraction(iv_a.hi)
            fb_lo, fb_hi = Fraction(iv_b.lo), Fraction(iv_b.hi)
            exact_prods = (fa_lo * fb_lo, fa_lo * fb_hi, fa_hi * fb_lo, fa_hi * fb_hi)
            self.assertLessEqual(Fraction(iv_prod.lo), min(exact_prods))
            self.assertGreaterEqual(Fraction(iv_prod.hi), max(exact_prods))

        b = 1.25
        x0 = torch.tensor([0.3, -0.4, 0.5], dtype=torch.float64)
        w0 = cg.principal_w(x0, b, sheet_sign=-1.0)
        k0 = cg.tangent_field(x0, w0, b)

        enc_tiny_del = propagate_ray_step_enclosure(
            pos_hat=x0, k_hat=k0, prev_w_hat=w0, h_hat=0.25, b=b,
            eps_x_curr=1e-12, eps_k_curr=1e-12, eps_w_curr=1e-12,
            fl_advance_err=1e-15, fl_delta_err=0.0, fl_sqrt_err=1e-15, fl_pred_add_err=1e-15,
        )
        self.assertFalse(enc_tiny_del.valid)
        self.assertEqual(enc_tiny_del.reason, "computed_delta_exceeds_fl_delta_budget")

        dec_large_guard = evaluate_certified_step(
            pos_hat=x0, k_hat=k0, prev_w_hat=w0, h_hat=0.25, b=b,
            eps_x_curr=1e-12, eps_k_curr=1e-12, eps_w_curr=1e-12,
            fl_advance_err=1e-15, fl_delta_err=1e-12, fl_sqrt_err=1e-15, fl_pred_add_err=1e-15,
            ip_roundoff=1e-14, cmp_roundoff=1e-14, guard=100.0,
            min_clearance=1e-4, max_ray_drift=1e-10,
        )
        self.assertFalse(dec_large_guard.certified_accept)
        self.assertIsNone(dec_large_guard.selected_root)

        dec_ok = evaluate_certified_step(
            pos_hat=x0, k_hat=k0, prev_w_hat=w0, h_hat=0.25, b=b,
            eps_x_curr=1e-12, eps_k_curr=1e-12, eps_w_curr=1e-12,
            fl_advance_err=1e-15, fl_delta_err=1e-12, fl_sqrt_err=1e-15, fl_pred_add_err=1e-15,
            ip_roundoff=1e-14, cmp_roundoff=1e-14, guard=1e-8,
            min_clearance=1e-4, max_ray_drift=1e-10,
        )
        self.assertTrue(dec_ok.certified_accept)
        assert dec_ok.selected_root is not None
        self.assertLess(float(torch.abs(dec_ok.selected_root - (w0 + 0.25)).item()), 1e-13)
        p_next = dec_ok.enclosure.pos_next_hat
        exact_re_q, exact_im_q = _exact_fraction_delta_parts(
            float(p_next[0].item()), float(p_next[1].item()), float(p_next[2].item()), b
        )
        assert dec_ok.enclosure.delta_exact_next_rect.re.lo is not None and dec_ok.enclosure.delta_exact_next_rect.re.hi is not None
        assert dec_ok.enclosure.delta_exact_next_rect.im.lo is not None and dec_ok.enclosure.delta_exact_next_rect.im.hi is not None
        self.assertLessEqual(Fraction(dec_ok.enclosure.delta_exact_next_rect.re.lo), exact_re_q)
        self.assertLessEqual(exact_re_q, Fraction(dec_ok.enclosure.delta_exact_next_rect.re.hi))
        self.assertLessEqual(Fraction(dec_ok.enclosure.delta_exact_next_rect.im.lo), exact_im_q)
        self.assertLessEqual(exact_im_q, Fraction(dec_ok.enclosure.delta_exact_next_rect.im.hi))

    def test_4_near_ring_drift_and_dual_sign_strictness(self):
        b = 1.0
        pos_near = torch.tensor([1.0 + 1e-10, 0.0, 0.0], dtype=torch.float64)
        w_near = cg.principal_w(pos_near, b, sheet_sign=-1.0)
        k_near = cg.tangent_field(pos_near, w_near, b)
        dec_near = evaluate_certified_step(
            pos_hat=pos_near, k_hat=k_near, prev_w_hat=w_near, h_hat=1e-4, b=b,
            eps_x_curr=1e-6, eps_k_curr=1e-6, eps_w_curr=1e-6,
            fl_advance_err=1e-12, fl_delta_err=1e-8, fl_sqrt_err=1e-8, fl_pred_add_err=1e-12,
            ip_roundoff=1e-12, cmp_roundoff=1e-12, guard=1e-8, min_clearance=0.0, max_ray_drift=1e-5,
        )
        self.assertFalse(dec_near.certified_accept)

        x0 = torch.tensor([0.3, -0.4, 0.5], dtype=torch.float64)
        w0 = cg.principal_w(x0, 1.25, sheet_sign=-1.0)
        k0 = cg.tangent_field(x0, w0, 1.25)
        dec_drift = evaluate_certified_step(
            pos_hat=x0, k_hat=k0, prev_w_hat=w0, h_hat=0.2, b=1.25,
            eps_x_curr=1e-10, eps_k_curr=5e-2, eps_w_curr=1e-10,
            fl_advance_err=1e-14, fl_delta_err=1e-12, fl_sqrt_err=1e-14, fl_pred_add_err=1e-14,
            ip_roundoff=1e-14, cmp_roundoff=1e-14, guard=1e-8, min_clearance=1e-4, max_ray_drift=1e-4,
        )
        self.assertFalse(dec_drift.certified_accept)
        self.assertEqual(dec_drift.reason, "ray_endpoint_drift_exceeds_budget")

        sgn_zero, ok_zero = _certified_phase_inner_product_sign(
            complex(1.0, 0.0), complex(0.0, 1.0),
            RealInterval.point_float(0.0), RealInterval.point_float(0.0), RealInterval.point_float(0.0),
        )
        self.assertIsNone(sgn_zero)
        self.assertFalse(ok_zero)

        # Deliberately force phase sign to be indeterminate (None, False) while predictor is satisfied
        with patch("twistor.certified_selector._certified_phase_inner_product_sign", return_value=(None, False)):
            dec_single = evaluate_certified_step(
                pos_hat=x0, k_hat=k0, prev_w_hat=w0, h_hat=0.25, b=1.25,
                eps_x_curr=1e-12, eps_k_curr=1e-12, eps_w_curr=1e-12,
                fl_advance_err=1e-15, fl_delta_err=1e-12, fl_sqrt_err=1e-15, fl_pred_add_err=1e-15,
                ip_roundoff=1e-14, cmp_roundoff=1e-14, guard=1e-8, min_clearance=1e-4, max_ray_drift=1e-10,
            )
        self.assertFalse(dec_single.certified_accept)
        self.assertEqual(dec_single.reason, "dual_certificates_not_both_satisfied")
        self.assertFalse(dec_single.phase_certified)
        self.assertTrue(dec_single.predictor_certified)

        a_iv = RealInterval.point_float(1.0)
        b_iv = RealInterval.point_float(1.0)
        self.assertFalse(a_iv.strictly_greater_than(b_iv))


if __name__ == "__main__":
    unittest.main(verbosity=2)
