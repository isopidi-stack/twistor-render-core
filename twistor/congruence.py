"""
Reference implementation of Theorems 1 and 2 from docs/THEOREMS_AND_PROOFS.md.

Theorem 1 (Two-Sheeted Covering & Congruence Optics):
  - Covering space: w^2 = Delta(x, y, z) = (z + i*b)^2 + x^2 + y^2, b != 0.
  - Global alpha-free unit tangent field on Omega_tilde_b:
        k(x, w) = ((alpha*x + b*y)/(alpha^2 + b^2), (alpha*y - b*x)/(alpha^2 + b^2), beta/b)
    where w = alpha + i*beta.
  - Optical conventions for right-handed (e1, e2, k) with B_AB = e_A^i e_B^j d_j k_i:
        sigma = 0,   rho = theta + i*omega = 1 / conj(w).
  - Bounded two-chart atlas: zeta*eta = 1, m_N(1/zeta) = (conj(zeta)/zeta) * m_S(zeta).

Theorem 2 (Restricted Robinson-Villarceau Correspondence):
  - Explicitly restricted to D_Rob = {(x0, k) : |k|=1, -1<kz<1, z0=0, x0!=0, x0_x*kx + x0_y*ky = 0}.
"""

from __future__ import annotations

import math
import torch


def _validate_b(b: float) -> float:
    b_val = float(b)
    if b_val == 0.0 or not math.isfinite(b_val):
        raise ValueError(f"Helicity parameter b must be finite and nonzero, got {b}.")
    return b_val


def delta(pos: torch.Tensor, b: float) -> torch.Tensor:
    """Compute Delta(x, y, z) = (z + i*b)^2 + x^2 + y^2 as complex128."""
    b_val = _validate_b(b)
    pos64 = pos.to(dtype=torch.float64)
    x, y, z = pos64[..., 0], pos64[..., 1], pos64[..., 2]
    v = torch.complex(z, torch.full_like(z, b_val))
    return v * v + (x * x + y * y).to(dtype=torch.complex128)


def principal_w(pos: torch.Tensor, b: float, sheet_sign: float = 1.0) -> torch.Tensor:
    """Evaluate w = sheet_sign * sqrt(Delta(pos, b)) using the principal branch."""
    if sheet_sign not in (1.0, -1.0, 1, -1):
        raise ValueError(f"sheet_sign must be +1 or -1, got {sheet_sign}.")
    return float(sheet_sign) * torch.sqrt(delta(pos, b))


def continue_sheet_step(pos: torch.Tensor, prev_w: torch.Tensor, b: float) -> torch.Tensor:
    """
    Incremental single-step branch continuation on Omega_tilde_b.
    Selects the square root of Delta(pos, b) having nonnegative real inner product with prev_w.
    """
    w_cand = torch.sqrt(delta(pos, b))
    prev_c = prev_w.to(dtype=torch.complex128, device=w_cand.device)
    inner = w_cand.real * prev_c.real + w_cand.imag * prev_c.imag
    return torch.where(inner < 0.0, -w_cand, w_cand)


def tangent_field(pos: torch.Tensor, w: torch.Tensor, b: float) -> torch.Tensor:
    """
    Global alpha-free unit tangent field k(x, w) on Omega_tilde_b:
        k = ((alpha*x + b*y)/(alpha^2 + b^2), (alpha*y - b*x)/(alpha^2 + b^2), beta/b).
    Nonsingular everywhere on Omega_tilde_b including the lifted spanning disk (alpha = 0)
    and the z-axis (x = y = 0).
    """
    b_val = _validate_b(b)
    pos64 = pos.to(dtype=torch.float64)
    w128 = w.to(dtype=torch.complex128, device=pos64.device)
    x, y = pos64[..., 0], pos64[..., 1]
    alpha, beta = w128.real, w128.imag
    denom = alpha * alpha + b_val * b_val
    kx = (alpha * x + b_val * y) / denom
    ky = (alpha * y - b_val * x) / denom
    kz = beta / b_val
    return torch.stack([kx, ky, kz], dim=-1)


def minkowski_null_lift(k: torch.Tensor) -> torch.Tensor:
    """Stationary 4D Minkowski null lift K^mu = (1, kx, ky, kz) for g = diag(1, -1, -1, -1)."""
    k64 = k.to(dtype=torch.float64)
    ones = torch.ones_like(k64[..., :1])
    return torch.cat([ones, k64], dim=-1)


def optical_scalar(w: torch.Tensor) -> torch.Tensor:
    """Complex optical scalar rho = theta + i*omega = 1 / conj(w) under Theorem 1 conventions."""
    w128 = w.to(dtype=torch.complex128)
    return 1.0 / torch.conj(w128)


def sheet_mismatch_norm(pos: torch.Tensor, w: torch.Tensor, b: float) -> torch.Tensor:
    """Exact global sheet-merge tangent mismatch D(x) = ||k(w) - k(-w)|| = 2|w|/sqrt(alpha^2 + b^2)."""
    b_val = _validate_b(b)
    w128 = w.to(dtype=torch.complex128)
    return 2.0 * torch.abs(w128) / torch.sqrt(w128.real * w128.real + b_val * b_val)


def screen_vector_south(zeta: torch.Tensor) -> torch.Tensor:
    """Southern chart complex screen vector m_S(zeta) = (1 - zeta^2, i(1 + zeta^2), -2*zeta) / (sqrt(2)(1 + |zeta|^2))."""
    z = zeta.to(dtype=torch.complex128)
    denom = math.sqrt(2.0) * (1.0 + (z.real * z.real + z.imag * z.imag))
    mx = (1.0 - z * z) / denom
    my = 1j * (1.0 + z * z) / denom
    mz = (-2.0 * z) / denom
    return torch.stack([mx, my, mz], dim=-1)


def screen_vector_north(eta: torch.Tensor) -> torch.Tensor:
    """Northern chart complex screen vector m_N(eta) = (eta^2 - 1, i(eta^2 + 1), -2*eta) / (sqrt(2)(1 + |eta|^2))."""
    e = eta.to(dtype=torch.complex128)
    denom = math.sqrt(2.0) * (1.0 + (e.real * e.real + e.imag * e.imag))
    mx = (e * e - 1.0) / denom
    my = 1j * (e * e + 1.0) / denom
    mz = (-2.0 * e) / denom
    return torch.stack([mx, my, mz], dim=-1)


def bounded_chart_and_screen(pos: torch.Tensor, w: torch.Tensor, b: float) -> dict[str, torch.Tensor]:
    """
    Bounded two-chart atlas on S^2:
      - Selects southern chart (zeta, m_S) when kz >= 0  (|zeta| <= 1)
      - Selects northern chart (eta,  m_N) when kz <  0  (|eta|  <  1)
    Avoids pole singularities and z-axis 0/0 cancellation.
    """
    k = tangent_field(pos, w, b)
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
    use_south = kz >= 0.0

    denom_s = torch.where(use_south, 1.0 + kz, torch.ones_like(kz))
    denom_n = torch.where(~use_south, 1.0 - kz, torch.ones_like(kz))
    zeta = torch.complex(kx, ky) / denom_s
    eta = torch.complex(kx, -ky) / denom_n

    coord = torch.where(use_south, zeta, eta)
    m_s = screen_vector_south(zeta)
    m_n = screen_vector_north(eta)
    m = torch.where(use_south.unsqueeze(-1), m_s, m_n)
    return {
        "k": k,
        "coord": coord,
        "use_south": use_south,
        "m": m,
    }


# ---------------------------------------------------------------------------
# Theorem 2 — Restricted Robinson-Villarceau Construction and Explicit Inverse
# ---------------------------------------------------------------------------

def validate_robinson_ray(x0: torch.Tensor, k: torch.Tensor, tol: float = 1e-9) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate that (x0, k) lies in the restricted domain D_Rob of Theorem 2."""
    x0_64 = x0.to(dtype=torch.float64)
    k_64 = k.to(dtype=torch.float64)
    if x0_64.shape != (3,) or k_64.shape != (3,):
        raise ValueError("x0 and k must have shape (3,).")
    k_norm = float(torch.linalg.norm(k_64).item())
    if abs(k_norm - 1.0) > tol:
        raise ValueError(f"|k| must equal 1, got {k_norm}.")
    kz = float(k_64[2].item())
    if not (-1.0 + tol < kz < 1.0 - tol):
        raise ValueError(f"k_z must lie strictly in (-1, 1), got {kz}.")
    if abs(float(x0_64[2].item())) > tol:
        raise ValueError(f"x0 must lie in z=0 plane, got z0={float(x0_64[2].item())}.")
    x0_norm = float(torch.linalg.norm(x0_64).item())
    if x0_norm <= tol:
        raise ValueError("x0 must be nonzero in D_Rob.")
    horiz_dot = float((x0_64[0] * k_64[0] + x0_64[1] * k_64[1]).item())
    if abs(horiz_dot) > tol * max(1.0, x0_norm):
        raise ValueError(f"x0 must be orthogonal to (kx, ky), got dot={horiz_dot}.")
    return x0_64, k_64


def robinson_ray_to_params(x0: torch.Tensor, k: torch.Tensor, tol: float = 1e-9) -> tuple[complex, float]:
    """Forward Theorem 2 map (x0, k) in D_Rob -> (zeta, b) in C* x R*."""
    x0_64, k_64 = validate_robinson_ray(x0, k, tol=tol)
    kx, ky, kz = float(k_64[0].item()), float(k_64[1].item()), float(k_64[2].item())
    x0x, x0y = float(x0_64[0].item()), float(x0_64[1].item())
    zeta = complex(kx, ky) / (1.0 + kz)
    b = -(x0x * ky - x0y * kx) / (1.0 - kz * kz)
    if abs(zeta) <= tol or abs(b) <= tol:
        raise ValueError("Degenerate parameter mapping: zeta and b must be nonzero.")
    return zeta, b


def robinson_ray_to_villarceau_circle(
    x0: torch.Tensor,
    k: torch.Tensor,
    num_points: int = 256,
    tol: float = 1e-9,
) -> dict[str, torch.Tensor | complex | float]:
    """
    Forward Theorem 2 map from (x0, k) in D_Rob to the Villarceau circle gamma_{zeta, b}(u),
    returning points (num_points, 3), exact 3D center C_{zeta,b}, radius R_{zeta,b}, and (zeta, b).
    """
    if num_points < 3:
        raise ValueError("num_points must be at least 3.")
    zeta, b = robinson_ray_to_params(x0, k, tol=tol)
    a, c = zeta.real, zeta.imag
    D = math.sqrt(1.0 + a * a + c * c)

    u = torch.linspace(0.0, 2.0 * math.pi, num_points + 1, dtype=torch.float64, device=x0.device)[:-1]
    eiu = torch.complex(torch.cos(u), torch.sin(u))
    Z0 = eiu / D
    Z1 = eiu * complex(a, c) / D
    scale = b / (1.0 - Z1.imag)
    pts = torch.stack([scale * Z0.real, scale * Z0.imag, scale * Z1.real], dim=-1)

    center = torch.tensor([b * c, b * a, 0.0], dtype=torch.float64, device=x0.device)
    radius = abs(b) * D
    return {
        "points": pts,
        "center": center,
        "radius": radius,
        "zeta": zeta,
        "b": b,
    }


def villarceau_circle_to_robinson_ray(gamma_u: torch.Tensor, tol: float = 1e-8) -> dict[str, torch.Tensor | complex | float]:
    """
    Explicit Theorem 2 inverse on the forward-defined Robinson-Villarceau circle family.
    Recovers (x0, k) in D_Rob, (zeta, b), center C, and radius R independently of point
    ordering or non-uniform angular spacing.
    """
    pts = gamma_u.to(dtype=torch.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < 3:
        raise ValueError("gamma_u must have shape (N, 3) with N >= 3.")

    # Origin-passing plane Z = aX - cY via SVD (no mean subtraction, since 0 lies on the plane)
    _, S, Vh = torch.linalg.svd(pts, full_matrices=False)
    if float(S[2].item()) > tol * max(1.0, float(S[0].item())):
        raise ValueError(f"Points do not lie on a plane through the origin (sigma_min={float(S[2].item()):.3e}).")

    normal = Vh[2]
    if abs(float(normal[2].item())) <= tol:
        raise ValueError("Containing plane is vertical (normal_z == 0); outside Theorem 2 image.")
    n_scaled = normal / normal[2]  # (-a, c, 1)
    a = float((-n_scaled[0]).item())
    c = float(n_scaled[1].item())
    ac_sq = a * a + c * c
    if ac_sq <= tol:
        raise ValueError("Horizontal plane (a = c = 0) is outside D_Rob (k_z = +/-1).")

    # Exact linear least-squares circle fit in orthonormal plane basis (u1, u2)
    u1, u2 = Vh[0], Vh[1]
    xi = pts @ u1
    eta = pts @ u2
    A_mat = torch.stack([2.0 * xi, 2.0 * eta, torch.ones_like(xi)], dim=-1)
    rhs = xi * xi + eta * eta
    sol = torch.linalg.lstsq(A_mat, rhs).solution
    xi_c, eta_c, c0 = sol[0], sol[1], sol[2]
    center = xi_c * u1 + eta_c * u2
    radius = float(torch.sqrt((c0 + xi_c * xi_c + eta_c * eta_c).clamp(min=0.0)).item())

    b = (c * float(center[0].item()) + a * float(center[1].item())) / ac_sq
    denom = 1.0 + ac_sq
    k = torch.tensor([2.0 * a / denom, 2.0 * c / denom, (1.0 - ac_sq) / denom], dtype=torch.float64, device=pts.device)
    x0 = torch.tensor([-b * float(k[1].item()), b * float(k[0].item()), 0.0], dtype=torch.float64, device=pts.device)

    validate_robinson_ray(x0, k, tol=1e-7)
    return {
        "x0": x0,
        "k": k,
        "zeta": complex(a, c),
        "b": b,
        "center": center,
        "radius": radius,
    }

# ---------------------------------------------------------------------------
# Bridge Lemma 4 — Conditional Sheet Continuation & Adaptive Step Controller
# ---------------------------------------------------------------------------

from dataclasses import dataclass as _dataclass
from typing import Callable as _Callable, Literal as _Literal

ContinuationMethod = _Literal["phase_margin", "ray_predictor", "dual"]
SatisfiedBy = _Literal["both", "phase_margin", "ray_predictor", "none"]


def _require_finite_positive(val: float, name: str) -> float:
    f = float(val)
    if not math.isfinite(f) or f <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {val}.")
    return f


def _require_finite_nonneg(val: float, name: str) -> float:
    f = float(val)
    if not math.isfinite(f) or f < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative, got {val}.")
    return f


def _require_nonneg_int(val: int, name: str) -> int:
    if isinstance(val, bool) or not isinstance(val, int) or val < 0:
        raise ValueError(f"{name} must be a nonnegative integer (not bool), got {val!r}.")
    return val


def _require_scalar_complex(t: torch.Tensor, name: str, require_finite: bool = True) -> torch.Tensor:
    c = t.to(dtype=torch.complex128)
    if c.ndim != 0:
        raise ValueError(f"{name} must be a scalar (0-D) complex tensor, got shape {tuple(c.shape)}.")
    if require_finite and not bool(torch.isfinite(c).all().item()):
        raise ValueError(f"{name} must contain a finite complex value.")
    return c


def _require_vec3_real(t: torch.Tensor, name: str) -> torch.Tensor:
    if t.is_complex():
        raise ValueError(f"{name} must be a real-valued tensor, got complex dtype {t.dtype}.")
    v = t.to(dtype=torch.float64)
    if v.shape != (3,):
        raise ValueError(f"{name} must be a 3-component vector with shape (3,), got {tuple(v.shape)}.")
    if not bool(torch.isfinite(v).all().item()):
        raise ValueError(f"{name} must contain only finite values.")
    return v


@_dataclass(frozen=True)
class SheetStepResult:
    """
    Structured diagnostics for a Bridge Lemma 4 adaptive ray-continuation step.

    Note: Criterion flags indicate whether the Lemma 4C inequalities held in finite float64
    under the caller-supplied error bounds; they do not by themselves establish
    backend-certified enclosures or verify exact ray-endpoint geometry.
    """

    accepted: bool
    satisfied_by: SatisfiedBy
    phase_criterion_satisfied: bool
    predictor_criterion_satisfied: bool
    clearance_satisfied: bool
    segment_clearance: float
    selected_root: torch.Tensor | None
    pos_next: torch.Tensor | None
    h_used: float
    subdivisions: int
    reason: str
    phase_sign: int | None = None
    predictor_sign: int | None = None


def ray_segment_clearance_bounds(
    w_hat: torch.Tensor,
    h: float,
    eps_curr: float,
    min_clearance: float = 0.0,
) -> tuple[float, bool]:
    """
    Mandatory Bridge Lemma 4C segment-clearance lower bound along w(s_n + t) = w_n + t, t in [0, h]:
        d_min_hat   = sqrt(Im(w_hat)^2 + dist(-Re(w_hat), [0, h])^2)
        d_min_lower = max(0, d_min_hat - eps_curr)
    An overflowing clearance evaluation cannot authorize acceptance (returns False).
    """
    h_val = _require_finite_positive(h, "h")
    e0 = _require_finite_nonneg(eps_curr, "eps_curr")
    c_min = _require_finite_nonneg(min_clearance, "min_clearance")
    w_c128 = _require_scalar_complex(w_hat, "w_hat")

    a = complex(w_c128.item())
    neg_alpha = -a.real
    if neg_alpha < 0.0:
        dist_re = -neg_alpha
    elif neg_alpha > h_val:
        dist_re = neg_alpha - h_val
    else:
        dist_re = 0.0
    d_min_hat = math.hypot(a.imag, dist_re)
    d_min_lower = max(0.0, d_min_hat - e0)
    if not (math.isfinite(dist_re) and math.isfinite(d_min_hat) and math.isfinite(d_min_lower)):
        return 0.0, False
    return d_min_lower, (d_min_lower > c_min)


def phase_margin_step_bounds(
    w_hat: torch.Tensor,
    h: float,
    eps_curr: float,
    eps_next: float,
    ip_roundoff: float = 0.0,
    guard: float = 0.0,
) -> tuple[float, float, bool]:
    """
    Lemma 4C-1 Robinson-ray phase-margin criterion (specific to w_{n+1} = w_n + h):
        P(h)      = |a|^2 + h * Re(a)
        mu_lower  = P(h) - eps_curr * (2*|a| + eps_curr + h)
        E_bar(h)  = (|a| + eps_curr)*eps_next + (|a + h| + eps_curr)*eps_curr + eps_curr*eps_next
    An overflowing evaluation cannot itself authorize acceptance (returns criterion_satisfied=False).
    """
    h_val = _require_finite_positive(h, "h")
    e0 = _require_finite_nonneg(eps_curr, "eps_curr")
    e1 = _require_finite_nonneg(eps_next, "eps_next")
    d_ip = _require_finite_nonneg(ip_roundoff, "ip_roundoff")
    g_val = _require_finite_nonneg(guard, "guard")
    w_c128 = _require_scalar_complex(w_hat, "w_hat")

    a = complex(w_c128.item())
    a_abs = abs(a)
    a_sq = a_abs * a_abs
    h_re = h_val * a.real
    p_h = a_sq + h_re
    pen = e0 * (2.0 * a_abs + e0 + h_val)
    mu_lower = p_h - pen
    ap_abs = abs(a + h_val)
    e_bar = ((a_abs + e0) * e1) + ((ap_abs + e0) * e0) + (e0 * e1)
    threshold = e_bar + d_ip + g_val

    intermediates = (a_abs, a_sq, h_re, p_h, pen, mu_lower, ap_abs, e_bar, threshold)
    if not all(math.isfinite(v) for v in intermediates):
        return mu_lower, threshold, False
    return mu_lower, threshold, (mu_lower > threshold)


def ray_predictor_step_bounds(
    w_hat: torch.Tensor,
    h: float,
    eps_pred: float,
    eps_cand: float,
    cmp_roundoff: float = 0.0,
    guard: float = 0.0,
) -> tuple[float, float, bool]:
    """
    Lemma 4C-2 exact Robinson-ray predictor criterion for p = w_hat + h:
        |p| > 2*eps_pred + eps_cand + 0.5*cmp_roundoff + guard
    An overflowing evaluation cannot itself authorize acceptance (returns criterion_satisfied=False).
    """
    h_val = _require_finite_positive(h, "h")
    ep = _require_finite_nonneg(eps_pred, "eps_pred")
    ec = _require_finite_nonneg(eps_cand, "eps_cand")
    d_cmp = _require_finite_nonneg(cmp_roundoff, "cmp_roundoff")
    g_val = _require_finite_nonneg(guard, "guard")
    w_c128 = _require_scalar_complex(w_hat, "w_hat")

    a = complex(w_c128.item())
    p_c = a + h_val
    pred_mag = abs(p_c)
    threshold = (2.0 * ep) + ec + (0.5 * d_cmp) + g_val
    if not (math.isfinite(p_c.real) and math.isfinite(p_c.imag) and math.isfinite(pred_mag) and math.isfinite(threshold)):
        return pred_mag, threshold, False
    return pred_mag, threshold, (pred_mag > threshold)


def adaptive_continue_ray_step(
    pos: torch.Tensor,
    k_dir: torch.Tensor,
    prev_w: torch.Tensor,
    h: float,
    b: float,
    eps_curr: float,
    eps_next: float,
    method: ContinuationMethod = "dual",
    pred_add_roundoff: float = 0.0,
    ip_roundoff: float = 0.0,
    cmp_roundoff: float = 0.0,
    guard: float = 0.0,
    min_clearance: float = 0.0,
    max_subdivisions: int = 16,
    candidate_root_fn: _Callable[[torch.Tensor, float], torch.Tensor] | None = None,
) -> SheetStepResult:
    """
    Adaptive Robinson-ray sheet-continuation controller implementing Lemma 4C
    with mandatory segment clearance and method in {"phase_margin", "ray_predictor", "dual"}.

    Failure policy:
      - Invalid input shapes/dtypes (including complex pos/k_dir or bool/non-int max_subdivisions),
        non-finite caller arguments, or a non-scalar candidate tensor raise ValueError.
      - Non-finite candidate-root values (NaN/inf), non-finite step intermediates, insufficient
        clearance, or step halving underflow (reason="step_underflow") return a controlled rejection.
      - An overflowing criterion (or overflowing selector arithmetic for that criterion) cannot
        itself authorize acceptance, without blocking a finite complementary selector in dual mode.

    Preconditions (caller-supplied, not derived by this controller):
      - eps_curr bounds |prev_w - w_n| on the intended sheet.
      - pred_add_roundoff bounds predictor addition rounding so eps_pred = eps_curr + pred_add_roundoff.
      - eps_next encloses candidate-root error arising from the computed endpoint pos + h*k_dir,
        Delta evaluation, and complex sqrt.
    """
    if method not in ("phase_margin", "ray_predictor", "dual"):
        raise ValueError(f"Unsupported method: {method}")
    max_subs = _require_nonneg_int(max_subdivisions, "max_subdivisions")

    _validate_b(b)
    h_try = _require_finite_positive(h, "h")
    e0 = _require_finite_nonneg(eps_curr, "eps_curr")
    e1 = _require_finite_nonneg(eps_next, "eps_next")
    d_add = _require_finite_nonneg(pred_add_roundoff, "pred_add_roundoff")
    d_ip = _require_finite_nonneg(ip_roundoff, "ip_roundoff")
    d_cmp = _require_finite_nonneg(cmp_roundoff, "cmp_roundoff")
    g_val = _require_finite_nonneg(guard, "guard")
    c_min = _require_finite_nonneg(min_clearance, "min_clearance")

    pos64 = _require_vec3_real(pos, "pos")
    k64 = _require_vec3_real(k_dir, "k_dir")
    prev_c = _require_scalar_complex(prev_w, "prev_w")

    eps_pred = e0 + d_add
    if not math.isfinite(eps_pred):
        raise ValueError("eps_curr + pred_add_roundoff overflowed to non-finite.")
    eval_cand = candidate_root_fn if candidate_root_fn is not None else (lambda p, bv: torch.sqrt(delta(p, bv)))

    last_h_tried = h_try
    last_clearance = 0.0
    last_clearance_ok = False
    last_phase_ok = False
    last_pred_ok = False
    last_reason = "uncertified_after_subdivision"

    for sub in range(max_subs + 1):
        last_h_tried = h_try
        clearance, clearance_ok = ray_segment_clearance_bounds(
            prev_c, h_try, eps_curr=e0, min_clearance=c_min
        )
        _, _, phase_ok = phase_margin_step_bounds(
            prev_c, h_try, eps_curr=e0, eps_next=e1, ip_roundoff=d_ip, guard=g_val
        )
        _, _, pred_ok = ray_predictor_step_bounds(
            prev_c, h_try, eps_pred=eps_pred, eps_cand=e1, cmp_roundoff=d_cmp, guard=g_val
        )
        last_clearance, last_clearance_ok, last_phase_ok, last_pred_ok = (
            clearance, clearance_ok, phase_ok, pred_ok
        )

        step_eval_ready = True
        if not clearance_ok:
            last_reason = "insufficient_segment_clearance"
            step_eval_ready = False

        if step_eval_ready:
            pos_next = pos64 + h_try * k64
            if not bool(torch.isfinite(pos_next).all().item()):
                last_reason = "non_finite_intermediate"
                step_eval_ready = False

        if step_eval_ready:
            w_cand = _require_scalar_complex(eval_cand(pos_next, b), "candidate_root", require_finite=False)
            if not bool(torch.isfinite(w_cand).all().item()):
                last_reason = "non_finite_candidate_root"
                step_eval_ready = False

        if step_eval_ready:
            # Isolate phase-margin and predictor selector arithmetic so overflow in one
            # selector does not poison the complementary selector in dual mode.
            ip_val = float((w_cand.real * prev_c.real + w_cand.imag * prev_c.imag).item())
            if math.isfinite(ip_val):
                phase_sign: int | None = 1 if ip_val >= 0.0 else -1
                phase_root: torch.Tensor | None = w_cand if phase_sign == 1 else -w_cand
            else:
                phase_sign = None
                phase_root = None
                phase_ok = False
                last_phase_ok = False

            w_pred = prev_c + h_try
            if bool(torch.isfinite(w_pred).all().item()):
                d_plus = float(torch.abs(w_cand - w_pred).item())
                d_minus = float(torch.abs(-w_cand - w_pred).item())
            else:
                d_plus = float("inf")
                d_minus = float("inf")

            if math.isfinite(d_plus) and math.isfinite(d_minus):
                pred_sign: int | None = 1 if d_plus <= d_minus else -1
                pred_root: torch.Tensor | None = w_cand if pred_sign == 1 else -w_cand
            else:
                pred_sign = None
                pred_root = None
                pred_ok = False
                last_pred_ok = False

            if method == "phase_margin":
                if phase_ok and phase_root is not None:
                    return SheetStepResult(
                        accepted=True,
                        satisfied_by="phase_margin",
                        phase_criterion_satisfied=phase_ok,
                        predictor_criterion_satisfied=pred_ok,
                        clearance_satisfied=clearance_ok,
                        segment_clearance=clearance,
                        selected_root=phase_root,
                        pos_next=pos_next,
                        h_used=h_try,
                        subdivisions=sub,
                        reason="phase_margin_criterion_satisfied",
                        phase_sign=phase_sign,
                        predictor_sign=pred_sign,
                    )
                last_reason = "phase_margin_criterion_not_satisfied"
            elif method == "ray_predictor":
                if pred_ok and pred_root is not None:
                    return SheetStepResult(
                        accepted=True,
                        satisfied_by="ray_predictor",
                        phase_criterion_satisfied=phase_ok,
                        predictor_criterion_satisfied=pred_ok,
                        clearance_satisfied=clearance_ok,
                        segment_clearance=clearance,
                        selected_root=pred_root,
                        pos_next=pos_next,
                        h_used=h_try,
                        subdivisions=sub,
                        reason="ray_predictor_criterion_satisfied",
                        phase_sign=phase_sign,
                        predictor_sign=pred_sign,
                    )
                last_reason = "ray_predictor_criterion_not_satisfied"
            else:  # method == "dual"
                if phase_ok and pred_ok and phase_root is not None and pred_root is not None:
                    if phase_sign != pred_sign:
                        return SheetStepResult(
                            accepted=False,
                            satisfied_by="none",
                            phase_criterion_satisfied=True,
                            predictor_criterion_satisfied=True,
                            clearance_satisfied=clearance_ok,
                            segment_clearance=clearance,
                            selected_root=None,
                            pos_next=None,
                            h_used=h_try,
                            subdivisions=sub,
                            reason="invariant_violation_dual_sign_disagreement",
                            phase_sign=phase_sign,
                            predictor_sign=pred_sign,
                        )
                    return SheetStepResult(
                        accepted=True,
                        satisfied_by="both",
                        phase_criterion_satisfied=True,
                        predictor_criterion_satisfied=True,
                        clearance_satisfied=clearance_ok,
                        segment_clearance=clearance,
                        selected_root=phase_root,
                        pos_next=pos_next,
                        h_used=h_try,
                        subdivisions=sub,
                        reason="dual_criteria_satisfied_with_agreement",
                        phase_sign=phase_sign,
                        predictor_sign=pred_sign,
                    )
                if pred_ok and pred_root is not None:
                    return SheetStepResult(
                        accepted=True,
                        satisfied_by="ray_predictor",
                        phase_criterion_satisfied=False,
                        predictor_criterion_satisfied=True,
                        clearance_satisfied=clearance_ok,
                        segment_clearance=clearance,
                        selected_root=pred_root,
                        pos_next=pos_next,
                        h_used=h_try,
                        subdivisions=sub,
                        reason="predictor_satisfied_phase_inconclusive",
                        phase_sign=phase_sign,
                        predictor_sign=pred_sign,
                    )
                if phase_ok and phase_root is not None:
                    return SheetStepResult(
                        accepted=True,
                        satisfied_by="phase_margin",
                        phase_criterion_satisfied=True,
                        predictor_criterion_satisfied=False,
                        clearance_satisfied=clearance_ok,
                        segment_clearance=clearance,
                        selected_root=phase_root,
                        pos_next=pos_next,
                        h_used=h_try,
                        subdivisions=sub,
                        reason="phase_satisfied_predictor_inconclusive",
                        phase_sign=phase_sign,
                        predictor_sign=pred_sign,
                    )
                last_reason = "dual_neither_criterion_satisfied"

        if sub == max_subs:
            break

        h_next = h_try * 0.5
        if h_next <= 0.0 or not math.isfinite(h_next):
            return SheetStepResult(
                accepted=False,
                satisfied_by="none",
                phase_criterion_satisfied=last_phase_ok,
                predictor_criterion_satisfied=last_pred_ok,
                clearance_satisfied=last_clearance_ok,
                segment_clearance=last_clearance,
                selected_root=None,
                pos_next=None,
                h_used=last_h_tried,
                subdivisions=sub,
                reason="step_underflow",
            )
        h_try = h_next

    return SheetStepResult(
        accepted=False,
        satisfied_by="none",
        phase_criterion_satisfied=last_phase_ok,
        predictor_criterion_satisfied=last_pred_ok,
        clearance_satisfied=last_clearance_ok,
        segment_clearance=last_clearance,
        selected_root=None,
        pos_next=None,
        h_used=last_h_tried,
        subdivisions=max_subs,
        reason=last_reason,
    )
