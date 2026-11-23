"""
Bridge Lemma 6A–6C: Conditional interval-certified ray-position, discriminant,
and candidate-root error propagation using directed-rounding Decimal intervals.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, Overflow
import torch

from twistor import congruence as cg
from twistor.interval_bounds import (
    ComplexRectInterval,
    RealInterval,
    dec_from_float_checked,
    dec_nonneg_from_float_checked,
    make_contexts,
    sqrt_ceil_decimal,
    sqrt_floor_decimal,
)

_ARITH_ERRORS = (InvalidOperation, Overflow)


@dataclass(frozen=True)
class CertifiedStepEnclosure:
    """Propagated enclosures for Lemmas 6A–6C along a candidate step."""

    valid: bool
    reason: str
    pos_hat: torch.Tensor
    pos_next_hat: torch.Tensor
    h_hat: float
    eps_x_next: RealInterval
    eps_endpoint: RealInterval
    delta_hat: complex
    delta_ideal_rect: ComplexRectInterval
    delta_computed_rect: ComplexRectInterval
    delta_exact_next_rect: ComplexRectInterval
    d_hat_to_exact_next: RealInterval
    eps_delta_abs: RealInterval
    computed_delta_certified: bool
    modulus_sq_ideal: RealInterval
    condition_factor: RealInterval | None
    w_cand_hat: torch.Tensor
    eps_cand: RealInterval
    w_pred_hat: torch.Tensor
    eps_pred: RealInterval


def _vec3_norm_interval(vec: torch.Tensor, prec: int = 80) -> RealInterval:
    dn, up = make_contexts(prec)
    sq_lo = Decimal(0)
    sq_hi = Decimal(0)
    try:
        for i in range(3):
            vi = dec_from_float_checked(float(vec[i].item()), f"vec[{i}]")
            sq_lo = dn.add(sq_lo, dn.multiply(vi, vi))
            sq_hi = up.add(sq_hi, up.multiply(vi, vi))
        return RealInterval.from_bounds_decimal(
            sqrt_floor_decimal(sq_lo, prec=prec),
            sqrt_ceil_decimal(sq_hi, prec=prec),
            note="vec3_norm",
        )
    except _ARITH_ERRORS:
        return RealInterval.invalid("decimal_overflow_vec3_norm")


def _eval_delta_rect_from_pos_intervals(
    x_iv: RealInterval,
    y_iv: RealInterval,
    z_iv: RealInterval,
    b_d: Decimal,
    prec: int = 80,
) -> ComplexRectInterval:
    two_iv = RealInterval.point_decimal(Decimal(2), "const_2")
    b_iv = RealInterval.point_decimal(b_d, "b_exact")
    b_sq_iv = b_iv.sqr(prec=prec)
    re_iv = (
        x_iv.sqr(prec=prec)
        .add(y_iv.sqr(prec=prec), prec=prec)
        .add(z_iv.sqr(prec=prec), prec=prec)
        .sub(b_sq_iv, prec=prec)
    )
    im_iv = two_iv.mul(b_iv, prec=prec).mul(z_iv, prec=prec)
    return ComplexRectInterval(re=re_iv, im=im_iv)


def _abs_diff_upper_decimal(a: Decimal, b: Decimal, prec: int = 80) -> Decimal:
    dn, up = make_contexts(prec)
    low = dn.subtract(a, b)
    high = up.subtract(a, b)
    return max(up.abs(low), up.abs(high))


def _complex_distance_upper(z_re: Decimal, z_im: Decimal, rect: ComplexRectInterval, prec: int = 80) -> RealInterval:
    if not rect.is_valid_finite or rect.re.lo is None or rect.re.hi is None or rect.im.lo is None or rect.im.hi is None:
        return RealInterval.invalid("invalid_rect_for_distance")
    _, up = make_contexts(prec)
    try:
        dx = max(
            _abs_diff_upper_decimal(z_re, rect.re.lo, prec=prec),
            _abs_diff_upper_decimal(z_re, rect.re.hi, prec=prec),
        )
        dy = max(
            _abs_diff_upper_decimal(z_im, rect.im.lo, prec=prec),
            _abs_diff_upper_decimal(z_im, rect.im.hi, prec=prec),
        )
        d_hi = sqrt_ceil_decimal(up.add(up.multiply(dx, dx), up.multiply(dy, dy)), prec=prec)
        return RealInterval.from_bounds_decimal(Decimal(0), d_hi, note="rect_dist_upper")
    except _ARITH_ERRORS:
        return RealInterval.invalid("rect_dist_overflow")


def propagate_ray_step_enclosure(
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
    h_exact: float | None = None,
    ray_drift_curr: float = 0.0,
    prec: int = 80,
) -> CertifiedStepEnclosure:
    """
    Propagate conditional interval enclosures for Lemmas 6A–6C using |h_exact| * eps_k_curr.
    """
    cg._validate_b(b)
    if pos_hat.is_complex() or pos_hat.shape != (3,):
        raise ValueError(
            f"pos_hat must be a real tensor of shape (3,), got dtype={pos_hat.dtype}, shape={tuple(pos_hat.shape)}."
        )
    if k_hat.is_complex() or k_hat.shape != (3,):
        raise ValueError(
            f"k_hat must be a real tensor of shape (3,), got dtype={k_hat.dtype}, shape={tuple(k_hat.shape)}."
        )
    w_prev_c128 = prev_w_hat.to(dtype=torch.complex128)
    if w_prev_c128.ndim != 0:
        raise ValueError(f"prev_w_hat must be a scalar (0-D) tensor, got shape {tuple(w_prev_c128.shape)}.")

    pos64 = pos_hat.to(dtype=torch.float64)
    k64 = k_hat.to(dtype=torch.float64)
    h_d = dec_from_float_checked(h_hat, "h_hat")
    if h_d <= Decimal(0):
        raise ValueError(f"h_hat must be positive, got {h_hat!r}.")
    h_ref_d = h_d if h_exact is None else dec_from_float_checked(h_exact, "h_exact")
    if h_ref_d <= Decimal(0):
        raise ValueError(f"h_exact must be positive, got {h_exact!r}.")

    ex0_d = dec_nonneg_from_float_checked(eps_x_curr, "eps_x_curr")
    ek0_d = dec_nonneg_from_float_checked(eps_k_curr, "eps_k_curr")
    ew0_d = dec_nonneg_from_float_checked(eps_w_curr, "eps_w_curr")
    drift0_d = dec_nonneg_from_float_checked(ray_drift_curr, "ray_drift_curr")
    fl_adv_d = dec_nonneg_from_float_checked(fl_advance_err, "fl_advance_err")
    fl_del_d = dec_nonneg_from_float_checked(fl_delta_err, "fl_delta_err")
    fl_sqr_d = dec_nonneg_from_float_checked(fl_sqrt_err, "fl_sqrt_err")
    fl_padd_d = dec_nonneg_from_float_checked(fl_pred_add_err, "fl_pred_add_err")

    pos_next64 = pos64 + float(h_hat) * k64
    w_pred64 = w_prev_c128 + float(h_hat)
    w_cand64 = torch.sqrt(cg.delta(pos_next64, b)).to(dtype=torch.complex128)
    delta_hat_c = complex(cg.delta(pos_next64, b).item())

    inv_r = RealInterval.invalid("non_finite_state")
    inv_c = ComplexRectInterval(inv_r, inv_r)

    if not (
        bool(torch.isfinite(pos64).all().item())
        and bool(torch.isfinite(k64).all().item())
        and bool(torch.isfinite(w_prev_c128).all().item())
        and bool(torch.isfinite(pos_next64).all().item())
        and bool(torch.isfinite(w_pred64).all().item())
        and bool(torch.isfinite(w_cand64).all().item())
        and bool(torch.isfinite(torch.tensor(delta_hat_c, dtype=torch.complex128)).all().item())
    ):
        return CertifiedStepEnclosure(
            valid=False,
            reason="non_finite_state_or_candidate",
            pos_hat=pos64,
            pos_next_hat=pos_next64,
            h_hat=float(h_hat),
            eps_x_next=inv_r,
            eps_endpoint=inv_r,
            delta_hat=delta_hat_c,
            delta_ideal_rect=inv_c,
            delta_computed_rect=inv_c,
            delta_exact_next_rect=inv_c,
            d_hat_to_exact_next=inv_r,
            eps_delta_abs=inv_r,
            computed_delta_certified=False,
            modulus_sq_ideal=inv_r,
            condition_factor=None,
            w_cand_hat=w_cand64,
            eps_cand=inv_r,
            w_pred_hat=w_pred64,
            eps_pred=inv_r,
        )

    dn, up = make_contexts(prec)
    two_iv = RealInterval.point_decimal(Decimal(2), "const_2")
    b_d = dec_from_float_checked(b, "b")

    ex0 = RealInterval.point_decimal(ex0_d, "eps_x_curr")
    ek0 = RealInterval.point_decimal(ek0_d, "eps_k_curr")
    ew0 = RealInterval.point_decimal(ew0_d, "eps_w_curr")
    drift0 = RealInterval.point_decimal(drift0_d, "ray_drift_curr")
    fl_adv = RealInterval.point_decimal(fl_adv_d, "fl_advance_err")
    fl_del = RealInterval.point_decimal(fl_del_d, "fl_delta_err")
    fl_sqr = RealInterval.point_decimal(fl_sqr_d, "fl_sqrt_err")
    fl_padd = RealInterval.point_decimal(fl_padd_d, "fl_pred_add_err")

    h_iv = RealInterval.point_decimal(h_d, "h_hat")
    h_ref_iv = RealInterval.point_decimal(h_ref_d, "h_exact")
    h_ref_abs_iv = h_ref_iv.abs_enclosure(prec=prec)
    h_diff_abs = h_iv.sub(h_ref_iv, prec=prec).abs_enclosure(prec=prec)
    k_norm_iv = _vec3_norm_interval(k64, prec=prec)

    eps_x_next = (
        ex0
        .add(h_ref_abs_iv.mul(ek0, prec=prec), prec=prec)
        .add(h_diff_abs.mul(k_norm_iv, prec=prec), prec=prec)
        .add(fl_adv, prec=prec)
    )

    eps_endpoint = (
        ex0
        .add(h_ref_abs_iv.mul(ek0, prec=prec), prec=prec)
        .add(h_diff_abs.mul(k_norm_iv, prec=prec), prec=prec)
        .add(drift0, prec=prec)
        .add(fl_adv, prec=prec)
    )

    x0_iv = RealInterval.point_float(float(pos64[0].item()), "x0")
    y0_iv = RealInterval.point_float(float(pos64[1].item()), "y0")
    z0_iv = RealInterval.point_float(float(pos64[2].item()), "z0")
    kx_iv = RealInterval.point_float(float(k64[0].item()), "kx")
    ky_iv = RealInterval.point_float(float(k64[1].item()), "ky")
    kz_iv = RealInterval.point_float(float(k64[2].item()), "kz")

    xideal_nom_iv = x0_iv.add(h_ref_iv.mul(kx_iv, prec=prec), prec=prec)
    yideal_nom_iv = y0_iv.add(h_ref_iv.mul(ky_iv, prec=prec), prec=prec)
    zideal_nom_iv = z0_iv.add(h_ref_iv.mul(kz_iv, prec=prec), prec=prec)

    xn_pt_iv = RealInterval.point_float(float(pos_next64[0].item()), "xn")
    yn_pt_iv = RealInterval.point_float(float(pos_next64[1].item()), "yn")
    zn_pt_iv = RealInterval.point_float(float(pos_next64[2].item()), "zn")
    delta_exact_next_rect = _eval_delta_rect_from_pos_intervals(xn_pt_iv, yn_pt_iv, zn_pt_iv, b_d, prec=prec)

    dhat_re_d = dec_from_float_checked(delta_hat_c.real, "delta_hat.real")
    dhat_im_d = dec_from_float_checked(delta_hat_c.imag, "delta_hat.imag")
    d_hat_to_exact_next = _complex_distance_upper(dhat_re_d, dhat_im_d, delta_exact_next_rect, prec=prec)

    if not (
        eps_endpoint.is_valid_finite and eps_endpoint.hi is not None
        and eps_x_next.is_valid_finite and eps_x_next.hi is not None
        and d_hat_to_exact_next.is_valid_finite and d_hat_to_exact_next.hi is not None
    ):
        return CertifiedStepEnclosure(
            valid=False,
            reason="endpoint_or_next_delta_invalid",
            pos_hat=pos64,
            pos_next_hat=pos_next64,
            h_hat=float(h_hat),
            eps_x_next=eps_x_next,
            eps_endpoint=eps_endpoint,
            delta_hat=delta_hat_c,
            delta_ideal_rect=inv_c,
            delta_computed_rect=inv_c,
            delta_exact_next_rect=delta_exact_next_rect,
            d_hat_to_exact_next=d_hat_to_exact_next,
            eps_delta_abs=inv_r,
            computed_delta_certified=False,
            modulus_sq_ideal=inv_r,
            condition_factor=None,
            w_cand_hat=w_cand64,
            eps_cand=inv_r,
            w_pred_hat=w_pred64,
            eps_pred=inv_r,
        )

    computed_delta_ok = d_hat_to_exact_next.hi <= fl_del_d

    xideal_iv = RealInterval.expand_by_radius_decimal(xideal_nom_iv, eps_endpoint.hi, prec=prec, note="xideal")
    yideal_iv = RealInterval.expand_by_radius_decimal(yideal_nom_iv, eps_endpoint.hi, prec=prec, note="yideal")
    zideal_iv = RealInterval.expand_by_radius_decimal(zideal_nom_iv, eps_endpoint.hi, prec=prec, note="zideal")
    delta_ideal_rect = _eval_delta_rect_from_pos_intervals(xideal_iv, yideal_iv, zideal_iv, b_d, prec=prec)

    xcomp_iv = RealInterval.expand_by_radius_decimal(xn_pt_iv, eps_x_next.hi, prec=prec, note="xcomp")
    ycomp_iv = RealInterval.expand_by_radius_decimal(yn_pt_iv, eps_x_next.hi, prec=prec, note="ycomp")
    zcomp_iv = RealInterval.expand_by_radius_decimal(zn_pt_iv, eps_x_next.hi, prec=prec, note="zcomp")
    delta_computed_rect = _eval_delta_rect_from_pos_intervals(xcomp_iv, ycomp_iv, zcomp_iv, b_d, prec=prec)

    b_iv = RealInterval.point_decimal(b_d, "b_exact")
    m_iv = (
        xn_pt_iv.sqr(prec=prec)
        .add(yn_pt_iv.sqr(prec=prec), prec=prec)
        .add(zn_pt_iv.sqr(prec=prec), prec=prec)
        .add(b_iv.sqr(prec=prec), prec=prec)
    )
    sqrt_m_iv = m_iv.sqrt(prec=prec)

    shift_delta_iv = (
        two_iv.mul(sqrt_m_iv, prec=prec)
        .mul(eps_endpoint, prec=prec)
        .add(eps_endpoint.sqr(prec=prec), prec=prec)
    )
    eps_delta_abs = shift_delta_iv.add(d_hat_to_exact_next, prec=prec)

    mod_sq_ideal = delta_ideal_rect.mod_sq(prec=prec)
    mod_delta_ideal_iv = delta_ideal_rect.modulus(prec=prec)

    if not (
        delta_ideal_rect.is_valid_finite
        and delta_computed_rect.is_valid_finite
        and eps_delta_abs.is_valid_finite
        and mod_delta_ideal_iv.is_valid_finite
        and mod_delta_ideal_iv.lo is not None
        and mod_delta_ideal_iv.hi is not None
        and eps_delta_abs.hi is not None
        and m_iv.lo is not None
        and m_iv.hi is not None
    ):
        return CertifiedStepEnclosure(
            valid=False,
            reason="enclosure_overflow",
            pos_hat=pos64,
            pos_next_hat=pos_next64,
            h_hat=float(h_hat),
            eps_x_next=eps_x_next,
            eps_endpoint=eps_endpoint,
            delta_hat=delta_hat_c,
            delta_ideal_rect=delta_ideal_rect,
            delta_computed_rect=delta_computed_rect,
            delta_exact_next_rect=delta_exact_next_rect,
            d_hat_to_exact_next=d_hat_to_exact_next,
            eps_delta_abs=eps_delta_abs,
            computed_delta_certified=computed_delta_ok,
            modulus_sq_ideal=mod_sq_ideal,
            condition_factor=None,
            w_cand_hat=w_cand64,
            eps_cand=inv_r,
            w_pred_hat=w_pred64,
            eps_pred=inv_r,
        )

    cond_iv: RealInterval | None = None
    if mod_delta_ideal_iv.lo > Decimal(0):
        try:
            c_lo = dn.divide(m_iv.lo, mod_delta_ideal_iv.hi)
            c_hi = up.divide(m_iv.hi, mod_delta_ideal_iv.lo)
            cond_iv = RealInterval.from_bounds_decimal(c_lo, c_hi, note="C_Delta")
        except _ARITH_ERRORS:
            cond_iv = RealInterval.invalid("C_Delta_overflow")

    if not computed_delta_ok:
        return CertifiedStepEnclosure(
            valid=False,
            reason="computed_delta_exceeds_fl_delta_budget",
            pos_hat=pos64,
            pos_next_hat=pos_next64,
            h_hat=float(h_hat),
            eps_x_next=eps_x_next,
            eps_endpoint=eps_endpoint,
            delta_hat=delta_hat_c,
            delta_ideal_rect=delta_ideal_rect,
            delta_computed_rect=delta_computed_rect,
            delta_exact_next_rect=delta_exact_next_rect,
            d_hat_to_exact_next=d_hat_to_exact_next,
            eps_delta_abs=eps_delta_abs,
            computed_delta_certified=False,
            modulus_sq_ideal=mod_sq_ideal,
            condition_factor=cond_iv,
            w_cand_hat=w_cand64,
            eps_cand=inv_r,
            w_pred_hat=w_pred64,
            eps_pred=inv_r,
        )

    if mod_delta_ideal_iv.lo <= eps_delta_abs.hi:
        return CertifiedStepEnclosure(
            valid=False,
            reason="discriminant_enclosure_contains_or_touches_zero",
            pos_hat=pos64,
            pos_next_hat=pos_next64,
            h_hat=float(h_hat),
            eps_x_next=eps_x_next,
            eps_endpoint=eps_endpoint,
            delta_hat=delta_hat_c,
            delta_ideal_rect=delta_ideal_rect,
            delta_computed_rect=delta_computed_rect,
            delta_exact_next_rect=delta_exact_next_rect,
            d_hat_to_exact_next=d_hat_to_exact_next,
            eps_delta_abs=eps_delta_abs,
            computed_delta_certified=True,
            modulus_sq_ideal=mod_sq_ideal,
            condition_factor=cond_iv,
            w_cand_hat=w_cand64,
            eps_cand=inv_r,
            w_pred_hat=w_pred64,
            eps_pred=inv_r,
        )

    try:
        denom_lo = sqrt_floor_decimal(dn.subtract(mod_delta_ideal_iv.lo, eps_delta_abs.hi), prec=prec)
        if denom_lo <= Decimal(0):
            return CertifiedStepEnclosure(
                valid=False,
                reason="non_positive_root_denom",
                pos_hat=pos64,
                pos_next_hat=pos_next64,
                h_hat=float(h_hat),
                eps_x_next=eps_x_next,
                eps_endpoint=eps_endpoint,
                delta_hat=delta_hat_c,
                delta_ideal_rect=delta_ideal_rect,
                delta_computed_rect=delta_computed_rect,
                delta_exact_next_rect=delta_exact_next_rect,
                d_hat_to_exact_next=d_hat_to_exact_next,
                eps_delta_abs=eps_delta_abs,
                computed_delta_certified=True,
                modulus_sq_ideal=mod_sq_ideal,
                condition_factor=cond_iv,
                w_cand_hat=w_cand64,
                eps_cand=inv_r,
                w_pred_hat=w_pred64,
                eps_pred=inv_r,
            )
        root_pert_hi = up.divide(eps_delta_abs.hi, denom_lo)
        assert fl_sqr.hi is not None and ew0.hi is not None and h_diff_abs.hi is not None and fl_padd.hi is not None
        cand_err_hi = up.add(root_pert_hi, fl_sqr.hi)
        pred_err_hi = up.add(up.add(ew0.hi, h_diff_abs.hi), fl_padd.hi)
    except _ARITH_ERRORS:
        return CertifiedStepEnclosure(
            valid=False,
            reason="root_or_predictor_overflow",
            pos_hat=pos64,
            pos_next_hat=pos_next64,
            h_hat=float(h_hat),
            eps_x_next=eps_x_next,
            eps_endpoint=eps_endpoint,
            delta_hat=delta_hat_c,
            delta_ideal_rect=delta_ideal_rect,
            delta_computed_rect=delta_computed_rect,
            delta_exact_next_rect=delta_exact_next_rect,
            d_hat_to_exact_next=d_hat_to_exact_next,
            eps_delta_abs=eps_delta_abs,
            computed_delta_certified=True,
            modulus_sq_ideal=mod_sq_ideal,
            condition_factor=cond_iv,
            w_cand_hat=w_cand64,
            eps_cand=inv_r,
            w_pred_hat=w_pred64,
            eps_pred=inv_r,
        )

    eps_cand_iv = RealInterval.from_bounds_decimal(Decimal(0), cand_err_hi, note="eps_cand_existential_sign")
    eps_pred_iv = RealInterval.from_bounds_decimal(Decimal(0), pred_err_hi, note="eps_pred")

    if not (eps_cand_iv.is_valid_finite and eps_pred_iv.is_valid_finite):
        return CertifiedStepEnclosure(
            valid=False,
            reason="root_or_predictor_enclosure_invalid",
            pos_hat=pos64,
            pos_next_hat=pos_next64,
            h_hat=float(h_hat),
            eps_x_next=eps_x_next,
            eps_endpoint=eps_endpoint,
            delta_hat=delta_hat_c,
            delta_ideal_rect=delta_ideal_rect,
            delta_computed_rect=delta_computed_rect,
            delta_exact_next_rect=delta_exact_next_rect,
            d_hat_to_exact_next=d_hat_to_exact_next,
            eps_delta_abs=eps_delta_abs,
            computed_delta_certified=True,
            modulus_sq_ideal=mod_sq_ideal,
            condition_factor=cond_iv,
            w_cand_hat=w_cand64,
            eps_cand=eps_cand_iv,
            w_pred_hat=w_pred64,
            eps_pred=eps_pred_iv,
        )

    return CertifiedStepEnclosure(
        valid=True,
        reason="propagated_enclosure_valid",
        pos_hat=pos64,
        pos_next_hat=pos_next64,
        h_hat=float(h_hat),
        eps_x_next=eps_x_next,
        eps_endpoint=eps_endpoint,
        delta_hat=delta_hat_c,
        delta_ideal_rect=delta_ideal_rect,
        delta_computed_rect=delta_computed_rect,
        delta_exact_next_rect=delta_exact_next_rect,
        d_hat_to_exact_next=d_hat_to_exact_next,
        eps_delta_abs=eps_delta_abs,
        computed_delta_certified=True,
        modulus_sq_ideal=mod_sq_ideal,
        condition_factor=cond_iv,
        w_cand_hat=w_cand64,
        eps_cand=eps_cand_iv,
        w_pred_hat=w_pred64,
        eps_pred=eps_pred_iv,
    )
