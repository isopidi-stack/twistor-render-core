"""
Bridge Lemma 6D: Strict dual-certificate evaluation of segment clearance,
continuation criteria, and candidate-sign selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import torch

from twistor.certified_ray import CertifiedStepEnclosure, propagate_ray_step_enclosure
from twistor.interval_bounds import (
    RealInterval,
    dec_from_float_checked,
    dec_nonneg_from_float_checked,
)


@dataclass(frozen=True)
class CertifiedContinuationDecision:
    """Result of evaluating Lemma 6D strict interval continuation & sign checks."""

    certified_accept: bool
    reason: str
    clearance_certified: bool
    phase_certified: bool
    predictor_certified: bool
    phase_sign_certified: bool
    predictor_sign_certified: bool
    selected_root: torch.Tensor | None
    enclosure: CertifiedStepEnclosure


def _interval_dist_point_to_segment(
    pt_iv: RealInterval,
    seg_lo: Decimal,
    seg_hi: Decimal,
    prec: int = 80,
) -> RealInterval:
    if not pt_iv.is_valid_finite or pt_iv.lo is None or pt_iv.hi is None:
        return RealInterval.invalid("invalid_pt_interval")
    seg_lo_iv = RealInterval.point_decimal(seg_lo, "seg_lo")
    seg_hi_iv = RealInterval.point_decimal(seg_hi, "seg_hi")
    zero_iv = RealInterval.point_decimal(Decimal(0), "zero")

    if pt_iv.hi <= seg_lo:
        d_iv = seg_lo_iv.sub(pt_iv, prec=prec)
        if not d_iv.is_valid_finite or d_iv.lo is None or d_iv.hi is None:
            return RealInterval.invalid("dist_left_overflow")
        return RealInterval.from_bounds_decimal(max(Decimal(0), d_iv.lo), max(Decimal(0), d_iv.hi), note="dist_left")

    if pt_iv.lo >= seg_hi:
        d_iv = pt_iv.sub(seg_hi_iv, prec=prec)
        if not d_iv.is_valid_finite or d_iv.lo is None or d_iv.hi is None:
            return RealInterval.invalid("dist_right_overflow")
        return RealInterval.from_bounds_decimal(max(Decimal(0), d_iv.lo), max(Decimal(0), d_iv.hi), note="dist_right")

    left_ex = seg_lo_iv.sub(pt_iv, prec=prec) if pt_iv.lo < seg_lo else zero_iv
    right_ex = pt_iv.sub(seg_hi_iv, prec=prec) if pt_iv.hi > seg_hi else zero_iv
    if not (left_ex.is_valid_finite and right_ex.is_valid_finite and left_ex.hi is not None and right_ex.hi is not None):
        return RealInterval.invalid("dist_straddle_overflow")
    return RealInterval.from_bounds_decimal(
        Decimal(0),
        max(Decimal(0), left_ex.hi, right_ex.hi),
        note="dist_straddles",
    )


def _certified_phase_inner_product_sign(
    w_cand_c: complex,
    w_prev_c: complex,
    eps_cand_iv: RealInterval,
    eps_prev_iv: RealInterval,
    ip_roundoff_iv: RealInterval,
    prec: int = 80,
) -> tuple[int | None, bool]:
    if not (eps_cand_iv.is_valid_finite and eps_prev_iv.is_valid_finite and ip_roundoff_iv.is_valid_finite):
        return None, False

    cr = RealInterval.point_float(w_cand_c.real, "w_cand.real")
    ci = RealInterval.point_float(w_cand_c.imag, "w_cand.imag")
    pr = RealInterval.point_float(w_prev_c.real, "w_prev.real")
    pi = RealInterval.point_float(w_prev_c.imag, "w_prev.imag")

    ip_nom_iv = cr.mul(pr, prec=prec).add(ci.mul(pi, prec=prec), prec=prec)
    c_abs_iv = cr.sqr(prec=prec).add(ci.sqr(prec=prec), prec=prec).sqrt(prec=prec)
    p_abs_iv = pr.sqr(prec=prec).add(pi.sqr(prec=prec), prec=prec).sqrt(prec=prec)

    rad_iv = (
        p_abs_iv.mul(eps_cand_iv, prec=prec)
        .add(c_abs_iv.mul(eps_prev_iv, prec=prec), prec=prec)
        .add(eps_cand_iv.mul(eps_prev_iv, prec=prec), prec=prec)
        .add(ip_roundoff_iv, prec=prec)
    )
    if not rad_iv.is_valid_finite or rad_iv.hi is None:
        return None, False

    ip_enc_iv = RealInterval.expand_by_radius_decimal(ip_nom_iv, rad_iv.hi, prec=prec)
    if not ip_enc_iv.is_valid_finite or ip_enc_iv.lo is None or ip_enc_iv.hi is None:
        return None, False

    if ip_enc_iv.lo > Decimal(0):
        return 1, True
    if ip_enc_iv.hi < Decimal(0):
        return -1, True
    return None, False


def _complex_dist_interval(
    z_re_iv: RealInterval,
    z_im_iv: RealInterval,
    p_re_iv: RealInterval,
    p_im_iv: RealInterval,
    prec: int = 80,
) -> RealInterval:
    d_re = z_re_iv.sub(p_re_iv, prec=prec)
    d_im = z_im_iv.sub(p_im_iv, prec=prec)
    return d_re.sqr(prec=prec).add(d_im.sqr(prec=prec), prec=prec).sqrt(prec=prec)


def _certified_predictor_candidate_sign(
    w_cand_c: complex,
    w_pred_c: complex,
    eps_cand_iv: RealInterval,
    eps_pred_iv: RealInterval,
    cmp_roundoff_iv: RealInterval,
    prec: int = 80,
) -> tuple[int | None, bool]:
    if not (eps_cand_iv.is_valid_finite and eps_pred_iv.is_valid_finite and cmp_roundoff_iv.is_valid_finite):
        return None, False

    cr_iv = RealInterval.point_float(w_cand_c.real, "w_cand.real")
    ci_iv = RealInterval.point_float(w_cand_c.imag, "w_cand.imag")
    mcr_iv = RealInterval.point_decimal(-dec_from_float_checked(w_cand_c.real, "w_cand.real"), "-w_cand.real")
    mci_iv = RealInterval.point_decimal(-dec_from_float_checked(w_cand_c.imag, "w_cand.imag"), "-w_cand.imag")
    wr_iv = RealInterval.point_float(w_pred_c.real, "w_pred.real")
    wi_iv = RealInterval.point_float(w_pred_c.imag, "w_pred.imag")

    dp_iv = _complex_dist_interval(cr_iv, ci_iv, wr_iv, wi_iv, prec=prec)
    dm_iv = _complex_dist_interval(mcr_iv, mci_iv, wr_iv, wi_iv, prec=prec)

    two_iv = RealInterval.point_decimal(Decimal(2), "const_2")
    half_iv = RealInterval.point_decimal(Decimal("0.5"), "const_half")
    rad_sum_iv = eps_cand_iv.add(eps_pred_iv, prec=prec)
    sep_rad_iv = two_iv.mul(rad_sum_iv, prec=prec).add(half_iv.mul(cmp_roundoff_iv, prec=prec), prec=prec)
    if not sep_rad_iv.is_valid_finite or sep_rad_iv.hi is None:
        return None, False

    diff_iv = dm_iv.sub(dp_iv, prec=prec)
    cert_diff_iv = RealInterval.expand_by_radius_decimal(diff_iv, sep_rad_iv.hi, prec=prec)
    if not cert_diff_iv.is_valid_finite or cert_diff_iv.lo is None or cert_diff_iv.hi is None:
        return None, False

    if cert_diff_iv.lo > Decimal(0):
        return 1, True
    if cert_diff_iv.hi < Decimal(0):
        return -1, True
    return None, False


def evaluate_certified_step(
    pos_hat: torch.Tensor,
    k_hat: torch.Tensor,
    prev_w_hat: torch.Tensor,
    h_hat: float,
    b: float,
    eps_x_curr: float,
    eps_k_curr: float,
    eps_w_curr: float,
    fl_advance_err: float,
    fl_delta_err: float,
    fl_sqrt_err: float,
    fl_pred_add_err: float,
    ip_roundoff: float,
    cmp_roundoff: float,
    guard: float,
    min_clearance: float,
    max_ray_drift: float,
    h_exact: float | None = None,
    ray_drift_curr: float = 0.0,
    prec: int = 80,
) -> CertifiedContinuationDecision:
    """
    Evaluate Bridge Lemma 6D strict interval conditions under the dual-certificate contract.
    """
    max_drift_d = dec_nonneg_from_float_checked(max_ray_drift, "max_ray_drift")
    min_clear_d = dec_nonneg_from_float_checked(min_clearance, "min_clearance")
    d_ip_d = dec_nonneg_from_float_checked(ip_roundoff, "ip_roundoff")
    d_cmp_d = dec_nonneg_from_float_checked(cmp_roundoff, "cmp_roundoff")
    guard_d = dec_nonneg_from_float_checked(guard, "guard")
    ew0_d = dec_nonneg_from_float_checked(eps_w_curr, "eps_w_curr")

    h_hat_d = dec_from_float_checked(h_hat, "h_hat")
    h_ref_d = h_hat_d if h_exact is None else dec_from_float_checked(h_exact, "h_exact")
    h_ref_iv = RealInterval.point_decimal(h_ref_d, "h_exact")

    min_clear_iv = RealInterval.point_decimal(min_clear_d, "min_clearance")
    d_ip_iv = RealInterval.point_decimal(d_ip_d, "ip_roundoff")
    d_cmp_iv = RealInterval.point_decimal(d_cmp_d, "cmp_roundoff")
    guard_iv = RealInterval.point_decimal(guard_d, "guard")
    e0_iv = RealInterval.point_decimal(ew0_d, "eps_w_curr")

    enc = propagate_ray_step_enclosure(
        pos_hat=pos_hat,
        k_hat=k_hat,
        prev_w_hat=prev_w_hat,
        h_hat=h_hat,
        b=b,
        eps_x_curr=eps_x_curr,
        eps_k_curr=eps_k_curr,
        eps_w_curr=eps_w_curr,
        fl_advance_err=fl_advance_err,
        fl_delta_err=fl_delta_err,
        fl_sqrt_err=fl_sqrt_err,
        fl_pred_add_err=fl_pred_add_err,
        h_exact=h_exact,
        ray_drift_curr=ray_drift_curr,
        prec=prec,
    )
    if not enc.valid:
        return CertifiedContinuationDecision(
            certified_accept=False,
            reason=enc.reason,
            clearance_certified=False,
            phase_certified=False,
            predictor_certified=False,
            phase_sign_certified=False,
            predictor_sign_certified=False,
            selected_root=None,
            enclosure=enc,
        )

    if not enc.eps_endpoint.is_valid_finite or enc.eps_endpoint.hi is None or enc.eps_endpoint.hi > max_drift_d:
        return CertifiedContinuationDecision(
            certified_accept=False,
            reason="ray_endpoint_drift_exceeds_budget",
            clearance_certified=False,
            phase_certified=False,
            predictor_certified=False,
            phase_sign_certified=False,
            predictor_sign_certified=False,
            selected_root=None,
            enclosure=enc,
        )

    w_prev_c = complex(prev_w_hat.to(dtype=torch.complex128).item())
    a_re_d = dec_from_float_checked(w_prev_c.real, "a_re")
    a_re = RealInterval.point_decimal(a_re_d, "a_re")
    a_im = RealInterval.point_float(w_prev_c.imag, "a_im")
    ew_iv = enc.eps_pred
    ec_iv = enc.eps_cand
    two_iv = RealInterval.point_decimal(Decimal(2), "const_2")
    half_iv = RealInterval.point_decimal(Decimal("0.5"), "const_half")

    neg_alpha_iv = RealInterval.point_decimal(-a_re_d, "neg_alpha")
    dist_re_iv = _interval_dist_point_to_segment(neg_alpha_iv, Decimal(0), h_ref_d, prec=prec)
    d_nom_iv = a_im.sqr(prec=prec).add(dist_re_iv.sqr(prec=prec), prec=prec).sqrt(prec=prec)
    d_min_lower_iv = d_nom_iv.sub(e0_iv, prec=prec)
    clearance_ok = d_min_lower_iv.strictly_greater_than(min_clear_iv)

    if not clearance_ok:
        return CertifiedContinuationDecision(
            certified_accept=False,
            reason="segment_clearance_not_certified",
            clearance_certified=False,
            phase_certified=False,
            predictor_certified=False,
            phase_sign_certified=False,
            predictor_sign_certified=False,
            selected_root=None,
            enclosure=enc,
        )

    a_sq_iv = a_re.sqr(prec=prec).add(a_im.sqr(prec=prec), prec=prec)
    a_abs_iv = a_sq_iv.sqrt(prec=prec)
    p_h_iv = a_sq_iv.add(h_ref_iv.mul(a_re, prec=prec), prec=prec)
    pen_iv = e0_iv.mul(
        two_iv.mul(a_abs_iv, prec=prec).add(e0_iv, prec=prec).add(h_ref_iv, prec=prec),
        prec=prec,
    )
    mu_lower_iv = p_h_iv.sub(pen_iv, prec=prec)

    ap_abs_iv = a_re.add(h_ref_iv, prec=prec).sqr(prec=prec).add(a_im.sqr(prec=prec), prec=prec).sqrt(prec=prec)
    e_bar_iv = (
        a_abs_iv.add(e0_iv, prec=prec).mul(ec_iv, prec=prec)
        .add(ap_abs_iv.add(e0_iv, prec=prec).mul(e0_iv, prec=prec), prec=prec)
        .add(e0_iv.mul(ec_iv, prec=prec), prec=prec)
    )
    phase_rhs_iv = e_bar_iv.add(d_ip_iv, prec=prec).add(guard_iv, prec=prec)
    phase_margin_ok = mu_lower_iv.strictly_greater_than(phase_rhs_iv)

    w_pred_c = complex(enc.w_pred_hat.item())
    p_re_iv = RealInterval.point_float(w_pred_c.real, "p_re")
    p_im_iv = RealInterval.point_float(w_pred_c.imag, "p_im")
    p_mag_iv = p_re_iv.sqr(prec=prec).add(p_im_iv.sqr(prec=prec), prec=prec).sqrt(prec=prec)
    pred_rhs_iv = (
        two_iv.mul(ew_iv, prec=prec)
        .add(ec_iv, prec=prec)
        .add(half_iv.mul(d_cmp_iv, prec=prec), prec=prec)
        .add(guard_iv, prec=prec)
    )
    pred_margin_ok = p_mag_iv.strictly_greater_than(pred_rhs_iv)

    w_cand_c = complex(enc.w_cand_hat.item())
    phase_sign, phase_sign_ok = _certified_phase_inner_product_sign(
        w_cand_c, w_prev_c, ec_iv, e0_iv, d_ip_iv, prec=prec
    )
    pred_sign, pred_sign_ok = _certified_predictor_candidate_sign(
        w_cand_c, w_pred_c, ec_iv, ew_iv, d_cmp_iv, prec=prec
    )

    phase_cert = phase_margin_ok and phase_sign_ok and (phase_sign is not None)
    pred_cert = pred_margin_ok and pred_sign_ok and (pred_sign is not None)

    if not (phase_cert and pred_cert):
        return CertifiedContinuationDecision(
            certified_accept=False,
            reason="dual_certificates_not_both_satisfied",
            clearance_certified=True,
            phase_certified=phase_cert,
            predictor_certified=pred_cert,
            phase_sign_certified=phase_sign_ok,
            predictor_sign_certified=pred_sign_ok,
            selected_root=None,
            enclosure=enc,
        )

    if phase_sign != pred_sign:
        return CertifiedContinuationDecision(
            certified_accept=False,
            reason="invariant_violation_dual_sign_disagreement",
            clearance_certified=True,
            phase_certified=True,
            predictor_certified=True,
            phase_sign_certified=True,
            predictor_sign_certified=True,
            selected_root=None,
            enclosure=enc,
        )

    selected = enc.w_cand_hat if phase_sign == 1 else -enc.w_cand_hat
    return CertifiedContinuationDecision(
        certified_accept=True,
        reason="certified_both_criteria",
        clearance_certified=True,
        phase_certified=True,
        predictor_certified=True,
        phase_sign_certified=True,
        predictor_sign_certified=True,
        selected_root=selected,
        enclosure=enc,
    )
