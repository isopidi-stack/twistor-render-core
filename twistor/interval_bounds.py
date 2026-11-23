"""
Directed-rounding Decimal interval arithmetic for Bridge Lemma 6.

Enforces exact integer-coefficient square-bound inequalities (r_lo^2 <= x
and r_hi^2 >= x) for finite nonnegative Decimal inputs within supported
exponent limits, and catches Decimal Overflow/InvalidOperation cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Context, Decimal, InvalidOperation, Overflow, ROUND_CEILING, ROUND_FLOOR
import math

_ARITH_ERRORS = (InvalidOperation, Overflow)


def make_contexts(prec: int = 80) -> tuple[Context, Context]:
    if isinstance(prec, bool) or not isinstance(prec, int) or prec < 24:
        raise ValueError(f"prec must be an integer >= 24 (not bool), got {prec!r}.")
    return (
        Context(prec=prec, rounding=ROUND_FLOOR),
        Context(prec=prec, rounding=ROUND_CEILING),
    )


def dec_from_float_checked(val: float, name: str) -> Decimal:
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise ValueError(f"{name} must be a finite real number (not bool), got {val!r}.")
    f = float(val)
    if not math.isfinite(f):
        raise ValueError(f"{name} must be finite, got {val!r}.")
    return Decimal.from_float(f)


def dec_nonneg_from_float_checked(val: float, name: str) -> Decimal:
    d = dec_from_float_checked(val, name)
    if d < Decimal(0):
        raise ValueError(f"{name} must be nonnegative, got {val!r}.")
    return d


def _decimal_to_int_exp_nonneg(d: Decimal) -> tuple[int, int]:
    if not d.is_finite() or d < Decimal(0):
        raise InvalidOperation("Expected finite nonnegative Decimal.")
    tup = d.as_tuple()
    if tup.sign != 0 or not isinstance(tup.exponent, int):
        raise InvalidOperation("Invalid Decimal tuple for nonneg conversion.")
    mantissa = 0
    for digit in tup.digits:
        mantissa = mantissa * 10 + int(digit)
    return mantissa, int(tup.exponent)


def _compare_scaled_ints(m_a: int, e_a: int, m_b: int, e_b: int) -> int:
    if m_a == 0 and m_b == 0:
        return 0
    if m_a == 0:
        return -1 if m_b > 0 else 0
    if m_b == 0:
        return 1 if m_a > 0 else 0

    if abs(e_a - e_b) > 20000:
        raise InvalidOperation("Exponent difference exceeds supported exact-scaling budget.")

    common_exp = min(e_a, e_b)
    sa = m_a * (10 ** (e_a - common_exp))
    sb = m_b * (10 ** (e_b - common_exp))
    if sa < sb:
        return -1
    if sa > sb:
        return 1
    return 0


def square_leq_decimal_exact(r: Decimal, x: Decimal) -> bool:
    """Exact check whether r^2 <= x for finite nonneg Decimals r and x."""
    mr, er = _decimal_to_int_exp_nonneg(r)
    mx, ex = _decimal_to_int_exp_nonneg(x)
    return _compare_scaled_ints(mr * mr, 2 * er, mx, ex) <= 0


def square_geq_decimal_exact(r: Decimal, x: Decimal) -> bool:
    """Exact check whether r^2 >= x for finite nonneg Decimals r and x."""
    mr, er = _decimal_to_int_exp_nonneg(r)
    mx, ex = _decimal_to_int_exp_nonneg(x)
    return _compare_scaled_ints(mr * mr, 2 * er, mx, ex) >= 0


def _step_quantum_at_prec(r: Decimal, prec: int, rounding_mode: str) -> Decimal:
    if r == Decimal(0):
        exp = -prec
    else:
        exp = r.adjusted() - (prec - 1)
    ctx_step = Context(prec=prec, rounding=rounding_mode)
    return ctx_step.multiply(Decimal(1), Decimal(1).scaleb(exp))


def sqrt_floor_decimal(x: Decimal, prec: int = 80) -> Decimal:
    """Return target-precision (`prec` significant digits) r_lo >= 0 with exact r_lo^2 <= x."""
    if isinstance(prec, bool) or not isinstance(prec, int) or prec < 1:
        raise ValueError(f"prec must be a positive integer (not bool), got {prec!r}.")
    if not x.is_finite() or x < Decimal(0):
        raise InvalidOperation("sqrt_floor_decimal requires finite x >= 0")
    if x == Decimal(0):
        return Context(prec=prec, rounding=ROUND_FLOOR).plus(Decimal(0))

    init_prec = max(prec * 2 + 32, 128)
    init_ctx = Context(prec=init_prec, rounding=ROUND_FLOOR)
    target_ctx = Context(prec=prec, rounding=ROUND_FLOOR)

    r = target_ctx.plus(init_ctx.sqrt(x))
    step = _step_quantum_at_prec(r, prec, ROUND_FLOOR)
    if step <= Decimal(0):
        step = Decimal(1).scaleb(-prec)

    while not square_leq_decimal_exact(r, x):
        r_next = target_ctx.subtract(r, step)
        if r_next == r or r_next < Decimal(0):
            r = Decimal(0)
            break
        r = r_next
        step = _step_quantum_at_prec(r, prec, ROUND_FLOOR)

    if not square_leq_decimal_exact(r, x):
        raise ArithmeticError("Failed to establish exact floor square-root bound.")
    return r


def sqrt_ceil_decimal(x: Decimal, prec: int = 80) -> Decimal:
    """Return target-precision (`prec` significant digits) r_hi >= 0 with exact r_hi^2 >= x."""
    if isinstance(prec, bool) or not isinstance(prec, int) or prec < 1:
        raise ValueError(f"prec must be a positive integer (not bool), got {prec!r}.")
    if not x.is_finite() or x < Decimal(0):
        raise InvalidOperation("sqrt_ceil_decimal requires finite x >= 0")
    if x == Decimal(0):
        return Context(prec=prec, rounding=ROUND_CEILING).plus(Decimal(0))

    init_prec = max(prec * 2 + 32, 128)
    init_ctx = Context(prec=init_prec, rounding=ROUND_CEILING)
    target_ctx = Context(prec=prec, rounding=ROUND_CEILING)

    r = target_ctx.plus(init_ctx.sqrt(x))
    step = _step_quantum_at_prec(r, prec, ROUND_CEILING)
    if step <= Decimal(0):
        step = Decimal(1).scaleb(-prec)

    while not square_geq_decimal_exact(r, x):
        r_next = target_ctx.add(r, step)
        if r_next == r:
            step = _step_quantum_at_prec(r, max(prec + 8, 32), ROUND_CEILING)
            r_next = target_ctx.add(r, step)
        r = r_next
        step = _step_quantum_at_prec(r, prec, ROUND_CEILING)

    if not square_geq_decimal_exact(r, x):
        raise ArithmeticError("Failed to establish exact ceiling square-root bound.")
    return r


@dataclass(frozen=True)
class RealInterval:
    """Directed-rounding enclosure [lo, hi] in Decimal."""

    lo: Decimal | None
    hi: Decimal | None
    valid: bool = True
    note: str = "ok"

    @staticmethod
    def point_decimal(d: Decimal, note: str = "exact_decimal") -> "RealInterval":
        if not d.is_finite():
            return RealInterval(None, None, valid=False, note="non_finite_decimal")
        return RealInterval(d, d, valid=True, note=note)

    @staticmethod
    def point_float(val: float, name: str = "x") -> "RealInterval":
        d = dec_from_float_checked(val, name)
        return RealInterval(d, d, valid=True, note="exact_float64")

    @staticmethod
    def from_bounds_decimal(lo: Decimal, hi: Decimal, note: str = "bounds") -> "RealInterval":
        if not (lo.is_finite() and hi.is_finite()) or lo > hi:
            return RealInterval(None, None, valid=False, note="invalid_bounds")
        return RealInterval(lo, hi, valid=True, note=note)

    @staticmethod
    def expand_by_radius_decimal(
        iv: "RealInterval",
        radius: Decimal,
        prec: int = 80,
        note: str = "expanded",
    ) -> "RealInterval":
        if not iv.is_valid_finite or iv.lo is None or iv.hi is None:
            return RealInterval.invalid("invalid_base_interval_expand")
        if not radius.is_finite() or radius < Decimal(0):
            return RealInterval.invalid("invalid_radius_expand")
        dn, up = make_contexts(prec)
        try:
            lo = dn.subtract(iv.lo, radius)
            hi = up.add(iv.hi, radius)
        except _ARITH_ERRORS:
            return RealInterval.invalid("decimal_overflow_expand")
        if not (lo.is_finite() and hi.is_finite() and lo <= hi):
            return RealInterval.invalid("non_finite_or_inverted_expand")
        return RealInterval(lo, hi, valid=True, note=note)

    @staticmethod
    def invalid(reason: str) -> "RealInterval":
        return RealInterval(None, None, valid=False, note=reason)

    @property
    def is_valid_finite(self) -> bool:
        return (
            self.valid
            and self.lo is not None
            and self.hi is not None
            and self.lo.is_finite()
            and self.hi.is_finite()
            and self.lo <= self.hi
        )

    def strictly_greater_than(self, other: "RealInterval") -> bool:
        """Certified strict inequality self.lo > other.hi when both enclosures are valid."""
        if not (self.is_valid_finite and other.is_valid_finite):
            return False
        assert self.lo is not None and other.hi is not None
        return self.lo > other.hi

    def add(self, other: "RealInterval", prec: int = 80) -> "RealInterval":
        if not (self.is_valid_finite and other.is_valid_finite):
            return RealInterval.invalid("invalid_operand_add")
        assert self.lo is not None and self.hi is not None and other.lo is not None and other.hi is not None
        dn, up = make_contexts(prec)
        try:
            lo = dn.add(self.lo, other.lo)
            hi = up.add(self.hi, other.hi)
        except _ARITH_ERRORS:
            return RealInterval.invalid("decimal_overflow_add")
        if not (lo.is_finite() and hi.is_finite()):
            return RealInterval.invalid("non_finite_add")
        return RealInterval(lo, hi, valid=True, note="add")

    def sub(self, other: "RealInterval", prec: int = 80) -> "RealInterval":
        if not (self.is_valid_finite and other.is_valid_finite):
            return RealInterval.invalid("invalid_operand_sub")
        assert self.lo is not None and self.hi is not None and other.lo is not None and other.hi is not None
        dn, up = make_contexts(prec)
        try:
            lo = dn.subtract(self.lo, other.hi)
            hi = up.subtract(self.hi, other.lo)
        except _ARITH_ERRORS:
            return RealInterval.invalid("decimal_overflow_sub")
        if not (lo.is_finite() and hi.is_finite()):
            return RealInterval.invalid("non_finite_sub")
        return RealInterval(lo, hi, valid=True, note="sub")

    def mul(self, other: "RealInterval", prec: int = 80) -> "RealInterval":
        if not (self.is_valid_finite and other.is_valid_finite):
            return RealInterval.invalid("invalid_operand_mul")
        assert self.lo is not None and self.hi is not None and other.lo is not None and other.hi is not None
        dn, up = make_contexts(prec)
        try:
            dn_vals = (
                dn.multiply(self.lo, other.lo),
                dn.multiply(self.lo, other.hi),
                dn.multiply(self.hi, other.lo),
                dn.multiply(self.hi, other.hi),
            )
            up_vals = (
                up.multiply(self.lo, other.lo),
                up.multiply(self.lo, other.hi),
                up.multiply(self.hi, other.lo),
                up.multiply(self.hi, other.hi),
            )
            lo = min(dn_vals)
            hi = max(up_vals)
        except _ARITH_ERRORS:
            return RealInterval.invalid("decimal_overflow_mul")
        if not (lo.is_finite() and hi.is_finite()):
            return RealInterval.invalid("non_finite_mul")
        return RealInterval(lo, hi, valid=True, note="mul")

    def sqr(self, prec: int = 80) -> "RealInterval":
        if not self.is_valid_finite:
            return RealInterval.invalid("invalid_operand_sqr")
        assert self.lo is not None and self.hi is not None
        dn, up = make_contexts(prec)
        try:
            if self.lo <= Decimal(0) <= self.hi:
                lo = Decimal(0)
                hi = max(up.multiply(self.lo, self.lo), up.multiply(self.hi, self.hi))
            elif self.hi < Decimal(0):
                lo = dn.multiply(self.hi, self.hi)
                hi = up.multiply(self.lo, self.lo)
            else:
                lo = dn.multiply(self.lo, self.lo)
                hi = up.multiply(self.hi, self.hi)
        except _ARITH_ERRORS:
            return RealInterval.invalid("decimal_overflow_sqr")
        if not (lo.is_finite() and hi.is_finite()):
            return RealInterval.invalid("non_finite_sqr")
        return RealInterval(lo, hi, valid=True, note="sqr")

    def sqrt(self, prec: int = 80) -> "RealInterval":
        if not self.is_valid_finite:
            return RealInterval.invalid("invalid_operand_sqrt")
        assert self.lo is not None and self.hi is not None
        if self.lo < Decimal(0):
            return RealInterval.invalid("negative_lower_bound_in_real_sqrt")
        try:
            lo = sqrt_floor_decimal(self.lo, prec=prec)
            hi = sqrt_ceil_decimal(self.hi, prec=prec)
        except _ARITH_ERRORS:
            return RealInterval.invalid("decimal_overflow_sqrt")
        if not (lo.is_finite() and hi.is_finite() and lo <= hi):
            return RealInterval.invalid("non_finite_or_inverted_sqrt")
        return RealInterval(lo, hi, valid=True, note="sqrt_directed_exact")

    def abs_enclosure(self, prec: int = 80) -> "RealInterval":
        if not self.is_valid_finite:
            return RealInterval.invalid("invalid_operand_abs")
        assert self.lo is not None and self.hi is not None
        dn, up = make_contexts(prec)
        try:
            if self.lo <= Decimal(0) <= self.hi:
                lo = Decimal(0)
            elif self.hi < Decimal(0):
                lo = dn.minus(self.hi)
            else:
                lo = dn.plus(self.lo)
            hi = max(up.minus(self.lo), up.plus(self.hi))
        except _ARITH_ERRORS:
            return RealInterval.invalid("decimal_overflow_abs")
        if not (lo.is_finite() and hi.is_finite()):
            return RealInterval.invalid("non_finite_abs")
        return RealInterval(max(Decimal(0), lo), hi, valid=True, note="abs")


@dataclass(frozen=True)
class ComplexRectInterval:
    """Complex rectangular enclosure [re_lo, re_hi] + i*[im_lo, im_hi]."""

    re: RealInterval
    im: RealInterval

    @property
    def is_valid_finite(self) -> bool:
        return self.re.is_valid_finite and self.im.is_valid_finite

    def mod_sq(self, prec: int = 80) -> RealInterval:
        if not self.is_valid_finite:
            return RealInterval.invalid("invalid_complex_mod_sq")
        return self.re.sqr(prec=prec).add(self.im.sqr(prec=prec), prec=prec)

    def modulus(self, prec: int = 80) -> RealInterval:
        return self.mod_sq(prec=prec).sqrt(prec=prec)
