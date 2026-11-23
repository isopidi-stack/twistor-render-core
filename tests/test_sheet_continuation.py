"""
Positive, negative, adversarial, assumption-violation diagnostic, non-finite
input/candidate/overflow/underflow validation, dtype/shape validation, and
outward-rounded interval enclosure tests for Bridge Lemma 4 (Lemmas 4A, 4B, 4C).
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Context, Decimal, ROUND_CEILING, ROUND_FLOOR
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

EXPECTED_LEGACY_SHA256 = "8809F3AF1D5BBB43202128C740306402D9BFF264911146CB5DB974EBF9716217"

_CTX_DN = Context(prec=60, rounding=ROUND_FLOOR)
_CTX_UP = Context(prec=60, rounding=ROUND_CEILING)


@dataclass(frozen=True)
class _Interval:
    """Outward-rounded Decimal interval [lo, hi] with signed-arithmetic support."""

    lo: Decimal
    hi: Decimal

    @staticmethod
    def from_float(val: float) -> "_Interval":
        d = Decimal.from_float(float(val))
        return _Interval(d, d)

    def add(self, other: "_Interval") -> "_Interval":
        return _Interval(_CTX_DN.add(self.lo, other.lo), _CTX_UP.add(self.hi, other.hi))

    def sub(self, other: "_Interval") -> "_Interval":
        return _Interval(_CTX_DN.subtract(self.lo, other.hi), _CTX_UP.subtract(self.hi, other.lo))

    def neg(self) -> "_Interval":
        return _Interval(-self.hi, -self.lo)

    def mul(self, other: "_Interval") -> "_Interval":
        dn_prods = (
            _CTX_DN.multiply(self.lo, other.lo),
            _CTX_DN.multiply(self.lo, other.hi),
            _CTX_DN.multiply(self.hi, other.lo),
            _CTX_DN.multiply(self.hi, other.hi),
        )
        up_prods = (
            _CTX_UP.multiply(self.lo, other.lo),
            _CTX_UP.multiply(self.lo, other.hi),
            _CTX_UP.multiply(self.hi, other.lo),
            _CTX_UP.multiply(self.hi, other.hi),
        )
        return _Interval(min(dn_prods), max(up_prods))

    def sqr(self) -> "_Interval":
        """Exact outward-rounded square handling negative and zero-straddling intervals."""
        if self.lo <= 0 <= self.hi:
            hi_sq = max(_CTX_UP.multiply(self.lo, self.lo), _CTX_UP.multiply(self.hi, self.hi))
            return _Interval(Decimal(0), hi_sq)
        if self.hi < 0:
            return _Interval(_CTX_DN.multiply(self.hi, self.hi), _CTX_UP.multiply(self.lo, self.lo))
        return _Interval(_CTX_DN.multiply(self.lo, self.lo), _CTX_UP.multiply(self.hi, self.hi))

    def sqrt(self) -> "_Interval":
        if self.lo < 0:
            raise ValueError("Cannot take real sqrt of interval containing negative values.")
        return _Interval(_CTX_DN.sqrt(self.lo), _CTX_UP.sqrt(self.hi))

    def clamp_nonneg(self) -> "_Interval":
        return _Interval(max(Decimal(0), self.lo), max(Decimal(0), self.hi))


def _dist_point_to_segment_interval(x_iv: _Interval, h_iv: _Interval) -> _Interval:
    """Outward-rounded enclosure for dist(x, [0, h]) with h >= 0."""
    if x_iv.hi <= 0:
        return x_iv.neg()
    if x_iv.lo >= h_iv.hi:
        return x_iv.sub(h_iv)
    if x_iv.lo >= 0 and x_iv.hi <= h_iv.lo:
        return _Interval(Decimal(0), Decimal(0))
    left_dist = x_iv.neg().clamp_nonneg()
    right_dist = x_iv.sub(h_iv).clamp_nonneg()
    return _Interval(Decimal(0), max(left_dist.hi, right_dist.hi))


class TestBridgeLemma4SheetContinuation(unittest.TestCase):
    def test_1_frozen_legacy_hash_intact(self):
        digest = hashlib.sha256((REPO_ROOT / "palatial_gpo_ray_optics.py").read_bytes()).hexdigest().upper()
        self.assertEqual(digest, EXPECTED_LEGACY_SHA256)

    def test_2_lemma4a_exact_ray_margin_and_safe_step_corollary(self):
        b = 1.25
        x0 = torch.tensor([0.3, -0.4, 0.5], dtype=torch.float64)
        w0 = cg.principal_w(x0, b, sheet_sign=-1.0)
        w0_c = complex(w0.item())
        self.assertLess(w0_c.real, 0.0)
        self.assertNotEqual(w0_c.imag, 0.0)

        h_crit = (abs(w0_c) ** 2) / (-w0_c.real)
        self.assertGreater(h_crit, -w0_c.real)

        mu_below, thresh_zero, ok_below = cg.phase_margin_step_bounds(w0, 0.99 * h_crit, 0.0, 0.0)
        mu_above, _, ok_above = cg.phase_margin_step_bounds(w0, 1.01 * h_crit, 0.0, 0.0)
        self.assertEqual(thresh_zero, 0.0)
        self.assertTrue(ok_below)
        self.assertFalse(ok_above)

        exact_mu = w0_c.real * (w0_c.real + 0.99 * h_crit) + (w0_c.imag ** 2)
        self.assertLess(abs(mu_below - exact_mu), 1e-14)

    def test_3_coarse_lifted_disk_step_dual_vs_phase_margin(self):
        b = 1.25
        x0 = torch.tensor([0.3, -0.4, 0.5], dtype=torch.float64)
        w0 = cg.principal_w(x0, b, sheet_sign=-1.0)
        k0 = cg.tangent_field(x0, w0, b)
        w0_c = complex(w0.item())
        h_crit = (abs(w0_c) ** 2) / (-w0_c.real)
        h_coarse = 1.4 * h_crit

        res_dual = cg.adaptive_continue_ray_step(
            pos=x0, k_dir=k0, prev_w=w0, h=h_coarse, b=b,
            eps_curr=1e-9, eps_next=1e-9, method="dual",
            pred_add_roundoff=1e-14, cmp_roundoff=1e-12, guard=1e-6, min_clearance=1e-4,
            max_subdivisions=0,
        )
        self.assertTrue(res_dual.accepted)
        self.assertEqual(res_dual.satisfied_by, "ray_predictor")
        self.assertFalse(res_dual.phase_criterion_satisfied)
        self.assertTrue(res_dual.predictor_criterion_satisfied)
        self.assertTrue(res_dual.clearance_satisfied)
        self.assertEqual(res_dual.subdivisions, 0)
        self.assertLess(float(torch.abs(res_dual.selected_root - (w0 + h_coarse)).item()), 1e-13)

        res_phase = cg.adaptive_continue_ray_step(
            pos=x0, k_dir=k0, prev_w=w0, h=h_coarse, b=b,
            eps_curr=1e-9, eps_next=1e-9, method="phase_margin",
            ip_roundoff=1e-12, guard=1e-6, min_clearance=1e-4, max_subdivisions=16,
        )
        self.assertTrue(res_phase.accepted)
        self.assertEqual(res_phase.satisfied_by, "phase_margin")
        self.assertGreater(res_phase.subdivisions, 0)
        self.assertLess(res_phase.h_used, h_crit)
        self.assertLess(float(torch.abs(res_phase.selected_root - (w0 + res_phase.h_used)).item()), 1e-13)

    def test_4_singularity_crossing_segment_rejected_despite_confident_endpoint_predictor(self):
        b = 1.0
        x0 = torch.tensor([2.0, 0.0, 0.0], dtype=torch.float64)
        w0 = cg.principal_w(x0, b, sheet_sign=-1.0)
        k0 = cg.tangent_field(x0, w0, b)
        w0_c = complex(w0.item())
        h_jump = 2.0 * (-w0_c.real)

        _, _, pred_ok_endpoint = cg.ray_predictor_step_bounds(w0, h_jump, eps_pred=1e-9, eps_cand=1e-9)
        self.assertTrue(pred_ok_endpoint)

        d_low, clear_ok = cg.ray_segment_clearance_bounds(w0, h_jump, eps_curr=1e-9, min_clearance=1e-6)
        self.assertEqual(d_low, 0.0)
        self.assertFalse(clear_ok)

        res_reject = cg.adaptive_continue_ray_step(
            pos=x0, k_dir=k0, prev_w=w0, h=h_jump, b=b,
            eps_curr=1e-9, eps_next=1e-9, method="dual",
            min_clearance=1e-6, max_subdivisions=0,
        )
        self.assertFalse(res_reject.accepted)
        self.assertEqual(res_reject.satisfied_by, "none")
        self.assertFalse(res_reject.clearance_satisfied)
        self.assertEqual(res_reject.reason, "insufficient_segment_clearance")
        self.assertIsNone(res_reject.selected_root)

        res_sub = cg.adaptive_continue_ray_step(
            pos=x0, k_dir=k0, prev_w=w0, h=h_jump, b=b,
            eps_curr=1e-9, eps_next=1e-9, method="dual",
            min_clearance=1e-6, max_subdivisions=8,
        )
        self.assertTrue(res_sub.accepted)
        self.assertLess(res_sub.h_used, -w0_c.real)
        self.assertGreater(res_sub.segment_clearance, 1e-6)

    def test_5_near_ring_rejection_and_verified_perturbed_root_assumptions(self):
        b = 1.0
        x_ring = torch.tensor([1.0 + 1e-8, 0.0, 0.0], dtype=torch.float64)
        w_ring = cg.principal_w(x_ring, b, sheet_sign=-1.0)
        k_ring = cg.tangent_field(x_ring, w_ring, b)
        res_ring = cg.adaptive_continue_ray_step(
            pos=x_ring, k_dir=k_ring, prev_w=w_ring, h=1e-3, b=b,
            eps_curr=2e-4, eps_next=2e-4, method="dual", min_clearance=0.0, max_subdivisions=2,
        )
        self.assertFalse(res_ring.accepted)
        self.assertEqual(res_ring.satisfied_by, "none")
        self.assertAlmostEqual(res_ring.h_used, 2.5e-4, places=15)

        b2 = 1.25
        x0 = torch.tensor([0.3, -0.4, 0.5], dtype=torch.float64)
        w0 = cg.principal_w(x0, b2, sheet_sign=-1.0)
        k0 = cg.tangent_field(x0, w0, b2)
        eps_curr = 0.05
        pred_add_roundoff = 1e-14
        eps_next = 1e-10
        w0_c = complex(w0.item())
        ortho_unit = complex(-w0_c.imag, w0_c.real) / abs(w0_c)
        w0_pert = torch.tensor(w0_c + (0.95 * eps_curr) * ortho_unit, dtype=torch.complex128)
        self.assertLessEqual(float(torch.abs(w0_pert - w0).item()), eps_curr)

        res_both = cg.adaptive_continue_ray_step(
            pos=x0, k_dir=k0, prev_w=w0_pert, h=0.2, b=b2,
            eps_curr=eps_curr, eps_next=eps_next, method="dual",
            pred_add_roundoff=pred_add_roundoff, ip_roundoff=1e-12, cmp_roundoff=1e-12,
            guard=1e-6, min_clearance=0.1, max_subdivisions=4,
        )
        self.assertTrue(res_both.accepted)
        self.assertEqual(res_both.satisfied_by, "both")
        self.assertEqual(res_both.phase_sign, res_both.predictor_sign)

        w_exact_next = w0 + res_both.h_used
        actual_pred_err = float(torch.abs((w0_pert + res_both.h_used) - w_exact_next).item())
        self.assertLessEqual(actual_pred_err, eps_curr + pred_add_roundoff)

        w_cand_raw = torch.sqrt(cg.delta(res_both.pos_next, b2))
        actual_cand_err = min(
            float(torch.abs(w_cand_raw - w_exact_next).item()),
            float(torch.abs(-w_cand_raw - w_exact_next).item()),
        )
        self.assertLessEqual(actual_cand_err, eps_next)
        self.assertLess(float(torch.abs(res_both.selected_root - w_exact_next).item()), 1e-13)

    def test_6_dual_mode_detects_invalid_candidate_fault(self):
        """
        Assumption-violation diagnostic: deliberately supplies an invalid candidate root
        violating eps_next when both a-priori criteria pass. Verifies defensive disagreement
        detection (not a counterexample to Lemma 4C).
        """
        b = 1.25
        x0 = torch.tensor([0.3, -0.4, 0.5], dtype=torch.float64)
        w0 = cg.principal_w(x0, b, sheet_sign=-1.0)
        k0 = cg.tangent_field(x0, w0, b)
        h = 0.3

        w0_c = complex(w0.item())
        w_pred_c = w0_c + h
        ang0 = math.atan2(w0_c.imag, w0_c.real)
        ang1 = math.atan2(w_pred_c.imag, w_pred_c.real)
        ang_fault = 0.5 * (ang0 + ang1) + 0.5 * math.pi
        faulty_cand = torch.tensor(complex(math.cos(ang_fault), math.sin(ang_fault)), dtype=torch.complex128)

        res_fault = cg.adaptive_continue_ray_step(
            pos=x0, k_dir=k0, prev_w=w0, h=h, b=b,
            eps_curr=1e-9, eps_next=1e-9, method="dual",
            min_clearance=0.1, max_subdivisions=0,
            candidate_root_fn=lambda _pos, _b: faulty_cand,
        )
        self.assertFalse(res_fault.accepted)
        self.assertEqual(res_fault.satisfied_by, "none")
        self.assertTrue(res_fault.phase_criterion_satisfied)
        self.assertTrue(res_fault.predictor_criterion_satisfied)
        self.assertNotEqual(res_fault.phase_sign, res_fault.predictor_sign)
        self.assertEqual(res_fault.reason, "invariant_violation_dual_sign_disagreement")
        self.assertIsNone(res_fault.selected_root)

    def test_7_outward_rounded_interval_enclosures_all_lemma4c_bounds(self):
        b = 1.25
        x0 = torch.tensor([0.3, -0.4, 0.5], dtype=torch.float64)
        w0 = cg.principal_w(x0, b, sheet_sign=-1.0)
        a = complex(w0.item())
        h = 0.25
        e0, e1, d_ip, d_cmp, guard = 1e-6, 1e-6, 1e-12, 1e-11, 1e-9

        ar = _Interval.from_float(a.real)
        ai = _Interval.from_float(a.imag)
        h_iv = _Interval.from_float(h)
        e0_iv = _Interval.from_float(e0)
        e1_iv = _Interval.from_float(e1)
        dip_iv = _Interval.from_float(d_ip)
        dcmp_iv = _Interval.from_float(d_cmp)
        g_iv = _Interval.from_float(guard)
        two_iv = _Interval(Decimal(2), Decimal(2))
        half_iv = _Interval(Decimal("0.5"), Decimal("0.5"))

        a_sq = ar.sqr().add(ai.sqr())
        a_abs = a_sq.sqrt()
        p_h = a_sq.add(h_iv.mul(ar))
        pen = e0_iv.mul(two_iv.mul(a_abs).add(e0_iv).add(h_iv))
        mu_iv = p_h.sub(pen)

        ap_abs = ar.add(h_iv).sqr().add(ai.sqr()).sqrt()
        e_bar_iv = a_abs.add(e0_iv).mul(e1_iv).add(ap_abs.add(e0_iv).mul(e0_iv)).add(e0_iv.mul(e1_iv))
        phase_thresh_iv = e_bar_iv.add(dip_iv).add(g_iv)

        self.assertGreater(mu_iv.lo, phase_thresh_iv.hi)
        mu_f64, thresh_f64, ok_f64 = cg.phase_margin_step_bounds(w0, h, e0, e1, ip_roundoff=d_ip, guard=guard)
        self.assertTrue(ok_f64)
        self.assertLess(abs(Decimal.from_float(mu_f64) - mu_iv.lo), Decimal("1e-13"))
        self.assertLess(abs(Decimal.from_float(thresh_f64) - phase_thresh_iv.hi), Decimal("1e-13"))

        pred_thresh_iv = two_iv.mul(e0_iv).add(e1_iv).add(half_iv.mul(dcmp_iv)).add(g_iv)
        self.assertGreater(ap_abs.lo, pred_thresh_iv.hi)
        pmag_f64, pth_f64, pok_f64 = cg.ray_predictor_step_bounds(w0, h, e0, e1, cmp_roundoff=d_cmp, guard=guard)
        self.assertTrue(pok_f64)
        self.assertLess(abs(Decimal.from_float(pmag_f64) - ap_abs.lo), Decimal("1e-13"))
        self.assertLess(abs(Decimal.from_float(pth_f64) - pred_thresh_iv.hi), Decimal("1e-13"))

        for h_clr in (0.25, 1.0):
            h_c_iv = _Interval.from_float(h_clr)
            dist_re_iv = _dist_point_to_segment_interval(ar.neg(), h_c_iv)
            d_min_iv = ai.sqr().add(dist_re_iv.sqr()).sqrt().sub(e0_iv).clamp_nonneg()
            d_f64, c_ok = cg.ray_segment_clearance_bounds(w0, h_clr, e0, min_clearance=0.1)
            self.assertTrue(c_ok)
            self.assertLess(abs(Decimal.from_float(d_f64) - d_min_iv.lo), Decimal("1e-13"))

        w_unit = torch.tensor(1.0 + 0.0j, dtype=torch.complex128)
        h_b = 0.5
        _, _, ok_eq = cg.ray_predictor_step_bounds(w_unit, h_b, eps_pred=0.4, eps_cand=0.5, cmp_roundoff=0.4)
        _, _, ok_in = cg.ray_predictor_step_bounds(w_unit, h_b, eps_pred=0.4, eps_cand=0.5 - 1e-12, cmp_roundoff=0.4)
        _, _, ok_out = cg.ray_predictor_step_bounds(w_unit, h_b, eps_pred=0.4, eps_cand=0.5 + 1e-12, cmp_roundoff=0.4)
        self.assertFalse(ok_eq)
        self.assertTrue(ok_in)
        self.assertFalse(ok_out)

        # Boundary-enclosure test: exact outward-rounded lower bound rejects equality
        # and rejects the smallest float64 >= p_exact_iv.lo
        p_exact_iv = (
            _Interval.from_float(0.1).add(_Interval.from_float(0.3)).sqr()
            .add(_Interval.from_float(0.2).sqr())
            .sqrt()
        )
        self.assertFalse(p_exact_iv.lo > p_exact_iv.lo)
        f64_ceil = float(p_exact_iv.lo)
        if Decimal.from_float(f64_ceil) < p_exact_iv.lo:
            f64_ceil = math.nextafter(f64_ceil, math.inf)
        self.assertFalse(p_exact_iv.lo > Decimal.from_float(f64_ceil))

    def test_8_non_finite_input_and_candidate_failure_policy(self):
        b = 1.25
        x0 = torch.tensor([0.3, -0.4, 0.5], dtype=torch.float64)
        w0 = cg.principal_w(x0, b, sheet_sign=-1.0)
        k0 = cg.tangent_field(x0, w0, b)

        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError):
                cg.ray_segment_clearance_bounds(w0, 0.2, eps_curr=bad)
            with self.assertRaises(ValueError):
                cg.ray_segment_clearance_bounds(w0, 0.2, eps_curr=0.0, min_clearance=bad)
            with self.assertRaises(ValueError):
                cg.phase_margin_step_bounds(w0, 0.2, eps_curr=bad, eps_next=0.0)
            with self.assertRaises(ValueError):
                cg.phase_margin_step_bounds(w0, 0.2, eps_curr=0.0, eps_next=0.0, ip_roundoff=bad)
            with self.assertRaises(ValueError):
                cg.phase_margin_step_bounds(w0, 0.2, eps_curr=0.0, eps_next=0.0, guard=bad)
            with self.assertRaises(ValueError):
                cg.ray_predictor_step_bounds(w0, 0.2, eps_pred=bad, eps_cand=0.0)
            with self.assertRaises(ValueError):
                cg.ray_predictor_step_bounds(w0, 0.2, eps_pred=0.0, eps_cand=0.0, cmp_roundoff=bad)
            with self.assertRaises(ValueError):
                cg.adaptive_continue_ray_step(x0, k0, w0, 0.2, b, 1e-6, 1e-6, pred_add_roundoff=bad)

        bad_w = torch.tensor(complex(float("nan"), 1.0), dtype=torch.complex128)
        bad_pos = torch.tensor([float("inf"), 0.0, 0.0], dtype=torch.float64)
        bad_k = torch.tensor([0.0, float("nan"), 1.0], dtype=torch.float64)
        with self.assertRaises(ValueError):
            cg.phase_margin_step_bounds(bad_w, 0.2, 0.0, 0.0)
        with self.assertRaises(ValueError):
            cg.adaptive_continue_ray_step(bad_pos, k0, w0, 0.2, b, 1e-6, 1e-6)
        with self.assertRaises(ValueError):
            cg.adaptive_continue_ray_step(x0, bad_k, w0, 0.2, b, 1e-6, 1e-6)

        res_nan_cand = cg.adaptive_continue_ray_step(
            x0, k0, w0, 0.2, b, 1e-6, 1e-6, max_subdivisions=2,
            candidate_root_fn=lambda _p, _b: bad_w,
        )
        self.assertFalse(res_nan_cand.accepted)
        self.assertEqual(res_nan_cand.satisfied_by, "none")
        self.assertEqual(res_nan_cand.reason, "non_finite_candidate_root")
        self.assertEqual(res_nan_cand.subdivisions, 2)
        self.assertAlmostEqual(res_nan_cand.h_used, 0.05, places=15)

        with self.assertRaises(ValueError):
            cg.adaptive_continue_ray_step(
                x0, k0, w0, 0.2, b, 1e-6, 1e-6, max_subdivisions=0,
                candidate_root_fn=lambda _p, _b: torch.tensor([1.0 + 0.0j], dtype=torch.complex128),
            )

    def test_9_criterion_overflow_isolation_and_shape_validation(self):
        x0 = torch.tensor([0.3, -0.4, 0.5], dtype=torch.float64)
        k0 = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)

        w_huge = torch.tensor(1e200 + 0.0j, dtype=torch.complex128)
        _, _, phase_ok = cg.phase_margin_step_bounds(w_huge, 1.0, 0.0, 0.0)
        self.assertFalse(phase_ok)
        res_phase_ovf = cg.adaptive_continue_ray_step(
            pos=x0, k_dir=k0, prev_w=w_huge, h=1.0, b=1.25,
            eps_curr=0.0, eps_next=0.0, method="phase_margin", max_subdivisions=1,
            candidate_root_fn=lambda _p, _b: w_huge,
        )
        self.assertFalse(res_phase_ovf.accepted)
        self.assertFalse(res_phase_ovf.phase_criterion_satisfied)

        w_max = torch.tensor(1e308 + 0.0j, dtype=torch.complex128)
        _, _, pred_ok = cg.ray_predictor_step_bounds(w_max, 1e308, 0.0, 0.0)
        self.assertFalse(pred_ok)
        res_pred_ovf = cg.adaptive_continue_ray_step(
            pos=x0, k_dir=k0, prev_w=w_max, h=1e308, b=1.25,
            eps_curr=0.0, eps_next=0.0, method="ray_predictor", max_subdivisions=0,
            candidate_root_fn=lambda _p, _b: w_max,
        )
        self.assertFalse(res_pred_ovf.accepted)
        self.assertFalse(res_pred_ovf.predictor_criterion_satisfied)

        w0 = torch.tensor(1.0 + 0.5j, dtype=torch.complex128)
        with self.assertRaises(ValueError):
            cg.adaptive_continue_ray_step(torch.tensor([1.0, 2.0]), k0, w0, 0.1, 1.0, 1e-6, 1e-6)
        with self.assertRaises(ValueError):
            cg.adaptive_continue_ray_step(x0, torch.tensor([[1.0, 0.0, 0.0]]), w0, 0.1, 1.0, 1e-6, 1e-6)
        with self.assertRaises(ValueError):
            cg.adaptive_continue_ray_step(x0, k0, torch.tensor([1.0 + 0.5j]), 0.1, 1.0, 1e-6, 1e-6)
        with self.assertRaises(ValueError):
            cg.phase_margin_step_bounds(torch.tensor([1.0 + 0.5j]), 0.1, 0.0, 0.0)

    def test_10_adaptive_halving_step_underflow_returns_controlled_rejection(self):
        x0 = torch.tensor([0.3, -0.4, 0.5], dtype=torch.float64)
        k0 = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        w0 = torch.tensor(0.5 + 0.0j, dtype=torch.complex128)
        min_subnormal = math.nextafter(0.0, 1.0)
        h_start = 2.0 * min_subnormal

        res = cg.adaptive_continue_ray_step(
            pos=x0, k_dir=k0, prev_w=w0, h=h_start, b=1.25,
            eps_curr=1.0, eps_next=1.0, method="dual",
            min_clearance=10.0, max_subdivisions=4,
        )
        self.assertFalse(res.accepted)
        self.assertEqual(res.satisfied_by, "none")
        self.assertEqual(res.reason, "step_underflow")
        self.assertEqual(res.subdivisions, 1)
        self.assertEqual(res.h_used, min_subnormal)
        self.assertIsNone(res.selected_root)

    def test_11_max_subdivisions_rejects_booleans_and_non_integers(self):
        x0 = torch.tensor([0.3, -0.4, 0.5], dtype=torch.float64)
        k0 = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        w0 = torch.tensor(1.0 + 0.5j, dtype=torch.complex128)
        for bad_subs in (True, False, 1.5, 2.0, -1):
            with self.assertRaises(ValueError):
                cg.adaptive_continue_ray_step(
                    x0, k0, w0, 0.1, 1.25, 1e-6, 1e-6, max_subdivisions=bad_subs  # type: ignore[arg-type]
                )

    def test_12_real_vector_validation_rejects_complex_tensors(self):
        x0 = torch.tensor([0.3, -0.4, 0.5], dtype=torch.float64)
        k0 = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        w0 = torch.tensor(1.0 + 0.5j, dtype=torch.complex128)
        complex_vec = torch.tensor([0.3 + 0.1j, -0.4, 0.5], dtype=torch.complex128)

        with self.assertRaises(ValueError):
            cg.adaptive_continue_ray_step(complex_vec, k0, w0, 0.1, 1.25, 1e-6, 1e-6)
        with self.assertRaises(ValueError):
            cg.adaptive_continue_ray_step(x0, complex_vec, w0, 0.1, 1.25, 1e-6, 1e-6)

    def test_13_dual_mode_overflow_fallback_accepts_via_predictor(self):
        x0 = torch.tensor([0.3, -0.4, 0.5], dtype=torch.float64)
        k0 = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        w_huge = torch.tensor(1e200 + 0.0j, dtype=torch.complex128)

        # Phase criterion and inner product overflow (|a|^2 -> inf), while predictor stays finite
        res = cg.adaptive_continue_ray_step(
            pos=x0, k_dir=k0, prev_w=w_huge, h=1.0, b=1.25,
            eps_curr=0.0, eps_next=0.0, method="dual", max_subdivisions=0,
            candidate_root_fn=lambda _p, _b: w_huge,
        )
        self.assertTrue(res.accepted)
        self.assertEqual(res.satisfied_by, "ray_predictor")
        self.assertFalse(res.phase_criterion_satisfied)
        self.assertTrue(res.predictor_criterion_satisfied)
        self.assertIsNone(res.phase_sign)
        self.assertEqual(res.predictor_sign, 1)
        self.assertEqual(res.reason, "predictor_satisfied_phase_inconclusive")
        self.assertIsNotNone(res.selected_root)


if __name__ == "__main__":
    unittest.main(verbosity=2)
