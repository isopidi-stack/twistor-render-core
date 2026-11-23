"""
Non-breaking adapter for evaluating twistor.congruence along rays or position sets
without mutating legacy renderers or frozen audit scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import torch

from twistor import congruence as cg

SheetMode = Literal["principal", "continued"]


@dataclass(frozen=True)
class CongruenceRaySample:
    """Structured output for evaluating Theorem 1 congruence quantities along positions."""

    positions: torch.Tensor      # (..., 3) float64
    w: torch.Tensor              # (...) complex128
    k: torch.Tensor              # (..., 3) float64
    null_lift: torch.Tensor      # (..., 4) float64
    rho: torch.Tensor            # (...) complex128
    coord: torch.Tensor          # (...) complex128 (zeta if use_south else eta)
    m: torch.Tensor              # (..., 3) complex128 (m_S if use_south else m_N)
    use_south: torch.Tensor      # (...) bool
    min_step_margin: float | None
    near_ring_count: int


def evaluate_congruence_along_ray(
    origin: torch.Tensor,
    direction: torch.Tensor,
    s_values: torch.Tensor,
    b: float,
    sheet_mode: SheetMode = "continued",
    initial_sheet_sign: float = 1.0,
    ring_w_tol: float = 1e-4,
) -> CongruenceRaySample:
    """
    Evaluate Theorem 1 congruence fields along x(s) = origin + s * direction.

    Scope note (Uncertified integration interface): Uses direct single-step
    continue_sheet_step along caller-supplied directions (not restricted to exact
    congruence rays where w(s) = w_0 + s) and does not perform Bridge Lemma 4
    adaptive subdivision, segment-clearance checks, or error-enclosure certification.
    without mutating legacy renderers or frozen audit scripts.
    """
    o64 = origin.to(dtype=torch.float64)
    d64 = direction.to(dtype=torch.float64)
    s64 = s_values.to(dtype=torch.float64)
    if s64.ndim != 1 or s64.numel() < 1:
        raise ValueError("s_values must be a non-empty 1D tensor.")

    positions = o64.unsqueeze(0) + s64.unsqueeze(-1) * d64.unsqueeze(0)

    if sheet_mode == "principal":
        w = cg.principal_w(positions, b, sheet_sign=initial_sheet_sign)
        min_margin: float | None = None
    elif sheet_mode == "continued":
        w_steps: list[torch.Tensor] = [cg.principal_w(positions[0], b, sheet_sign=initial_sheet_sign)]
        margins: list[float] = []
        for idx in range(1, positions.shape[0]):
            w_prev = w_steps[-1]
            w_next = cg.continue_sheet_step(positions[idx], w_prev, b)
            mu = float((w_next.real * w_prev.real + w_next.imag * w_prev.imag).item())
            margins.append(mu)
            w_steps.append(w_next)
        w = torch.stack(w_steps, dim=0)
        min_margin = min(margins) if margins else None
    else:
        raise ValueError(f"Unsupported sheet_mode: {sheet_mode}")

    atlas = cg.bounded_chart_and_screen(positions, w, b)
    k = atlas["k"]
    null_lift = cg.minkowski_null_lift(k)
    rho = cg.optical_scalar(w)
    near_ring_count = int((torch.abs(w) < float(ring_w_tol)).sum().item())

    return CongruenceRaySample(
        positions=positions,
        w=w,
        k=k,
        null_lift=null_lift,
        rho=rho,
        coord=atlas["coord"],
        m=atlas["m"],
        use_south=atlas["use_south"],
        min_step_margin=min_margin,
        near_ring_count=near_ring_count,
    )