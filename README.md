# Argon (`twistor-render-core` — Argon v4.4)

Argon (`twistor-render-core`) is a computational geometry and rendering engine exploring spinor/twistor parameterizations, Robinson congruences, and Hopf–Villarceau circle correspondences.

For formal theorem statements, proof status, and scope boundaries, see **[`docs/THEOREMS_AND_PROOFS.md`](docs/THEOREMS_AND_PROOFS.md)**. For rendering and numerical guarantees, scope, and evidence, see **[`docs/RENDERING_GUARANTEES.md`](docs/RENDERING_GUARANTEES.md)**.

## Research progress wall — 25 November 2026

> **Current checkpoint: Bridge 9I-C pinned evidence reconciled and auditor endorsed.** Restricted 9I-A R2 design review and 9I-B R3 initialization are owner-accepted and closed. Next action: explicit owner acceptance of 9I-C and authorization of 9I-D recursive verification. No recursive execution yet. This is a documentation-only record of separately reviewed research, not executable promotion to GitHub.

| Research milestone | Qualified status |
| --- | --- |
| Bridge 7B–7D | Conditional T1–T4 / a posteriori numerical bounds and exact selection |
| Bridge 8 / 8F | Rational-witness recursion and bounded adaptive continuation; 8F owner accepted |
| Bridge 9A–9F | Exact algebraic mathematics and degree-four bound — owner accepted |
| Bridge 9H | Standalone exact-algebraic reference — owner accepted; simulation only |
| Bridge 9I-A | R2 design review — owner accepted and closed; qualified native affine trace deferral (not instruction verification) |
| Bridge 9I-B | R3 authenticated initialization — owner accepted and closed |
| Bridge 9I-C | R1 pinned Windows single-step continuation — evidence reconciled, auditor accepts closure, **owner closure pending** |
| Bridge 9I-D | **Next: owner authorization pending**; six mandatory recursive verification requirements |
| Bridge 9I-E–G / 9J | Adaptive continuation, further verification and integration/promotion not authorized |
| Renderer migration / aggregate certification | Locked |

**Evidence:** 9H 14/14 standalone tests and 7,193 reported oracle assertions; 9I-B 6/6 pinned tests, 13 initializations, 19 rejections, 248 reconciled assertions; 9I-C 6/6 pinned tests, 7 verified transitions, 259 predicates, 25 control records, 3 radius-masking controls, both selection signs and zero recursive evaluator entries. One intentionally unreconciled publication is not counted as verified. All pre/post captured preservation records reconcile to the accepted 697-file, six-worktree baseline; 26 inherited source pins match. No continuous attestation, universal resource bound or aggregate certification claimed.

**[Full checkpoint, artifact fingerprints, limitations and exact resumption instructions](docs/BRIDGE_9_PROGRESS_CHECKPOINT_2026-11-25.md)**

### Publication and preservation boundary

- Pre-documentation GitHub main baseline: `d0db6f43827218ef8b0ea5447edc1a385fffd5e6` (Bridge 6); remote main checkpoint `f0f11adb8b08306b60fc103e3cf09de4b251d5be` remains unchanged by this documentation update.
- Accepted local Bridge 8F checkpoint: `76e6267a63d6e3ccde873959edecf3f3ce27aa83`. Later executable review artifacts are retained separately, not promoted to main.
- Preserve the six local worktrees, 697 enumerated files, 26 source pins and all accepted archives. Do not merge or pull this documentation into the frozen local baseline.
- Both locks remain false: `_7B_INTEGRATION_ACCEPTED = False`, `all_certified = False`. No integration, promotion, adaptive or recursive execution is newly authorized by this checkpoint.

## Mathematical Status at a Glance

The repository maintains a strict separation between **established algebraic derivations**, **explicit proof obligations**, and **numerical or architectural verification**:

- **Theorem 1 — Covering & Congruence Optics:**
  - *Established:* Two-sheeted covering $\widetilde{\Omega}_b$ over $\mathbb{R}^3 \setminus \mathcal{B}_b$, global $\alpha$-free tangent field $\mathbf{k}(\mathbf{x}, w)$, Minkowski null lift, optical scalar $\rho = 1/\bar{w}$ and vanishing shear $\sigma = 0$ for right-handed screen frames, disk-cut domain $\Omega_b^{\mathrm{cut}}$ with continuous branch selection, chart transition $\zeta\eta = 1$ on overlaps, and screen-frame phase relation $\mathbf{m}_{\mathrm{N}}(1/\zeta) = (\bar{\zeta}/\zeta)\mathbf{m}_{\mathrm{S}}(\zeta)$.
- **Theorem 2 — Restricted Robinson–Villarceau Correspondence ($\mathcal{D}_{\mathrm{Rob}} \leftrightarrow \mathcal{V}_{\mathrm{Rob}}$):**
  - *Established:* Forward map from rays in $\mathcal{D}_{\mathrm{Rob}}$ to $(\zeta, b) \in \mathbb{C}^* \times \mathbb{R}^*$, finiteness of the stereographic image, origin-passing plane $Z = aX - cY$ with normal $\mathbf{N} = (-a, c, 1)$, radial extrema relation $r_{\max} r_{\min} = b^2$, exact 3D circle center $\mathbf{C}_{\zeta,b} = (bc, ba, 0)$ and radius $R_{\zeta,b} = |b|\sqrt{1 + |\zeta|^2}$, distinct unparameterized $+b / -b$ point sets for $\zeta \ne 0, b \ne 0$, signed-$b$ recovery from the circle center and plane normal scaled to third component $1$, and explicit reconstruction of $(\mathbf{k}, \mathbf{x}_0)$ back into $\mathcal{D}_{\mathrm{Rob}}$.
  - *Proof obligation:* Independent geometric characterization and surjectivity if the target circle family is defined outside the forward image of $\mathcal{D}_{\mathrm{Rob}}$.
- **Bridge Lemmas 1–3 — Transport, Sheet Discontinuity, and Conditioning vs. Evaluation Error:**
  - *Documented in `docs/RENDERING_GUARANTEES.md`:* Infinitesimal conformal footprint transport, global sheet-merge mismatch $D(\mathbf{x}, w)$ vs. incremental continuation across lifted disk crossings, exact spatial sensitivity ($\|\nabla\mathbf{k}\|_2 = 1/|w|$, $\|\nabla w\| = \sqrt{M}/|w|$, $\|\nabla\rho\| = \sqrt{M}/|w|^3$), conditional arithmetic-model cancellation/continuation bounds, and gauge-covariant chart handoff.

## Core Modules & Test Suites

- `twistor/congruence.py` — Reference implementation of Theorems 1 and 2 (`torch.float64` / `torch.complex128`) with incremental sheet continuation and bounded two-chart evaluation.
- `twistor/congruence_adapter.py` — Non-breaking ray/position adapter exposing explicit `"principal"` vs. `"continued"` sheet evaluation and structured diagnostics without altering frozen legacy scripts.
- `tests/test_congruence.py` — Unit tests for congruence optics, reversed-$k_z$ shear regression, deck transformation, two-chart atlas, footprint transport, and Theorem 2 round-trips.
- `tests/diagnose_legacy_gpo_compat.py` — Read-only compatibility diagnostic against frozen `palatial_gpo_ray_optics.py` (verifying its SHA-256 hash without modifying it).
- `tests/test_lemma3_numerics.py` — Numerical regression suite for Bridge Lemma 3 (conditioning sweep, 80-digit `Decimal.from_float` cancellation check, continuation margin/perturbations, and equatorial chart handoff).
- `tests/test_congruence_adapter.py` — Before/after migration tests verifying frozen-artifact integrity, continued vs. principal sheet behavior across lifted-disk crossings, equatorial handoff, and $z$-axis regularity.

## Scope, Open Obligations & Repository Layout

- **Numerical & integrity verification:** Regression suites and SHA-256 baseline checks verify consistency on tested configurations and pinned files; they do not substitute for formal proofs or establish backend-independent error constants.
- **Renderer integration & architectural scope:** Full migration/integration of `twistor/congruence.py` into renderer pipelines and image-level guarantees remain separate engineering tasks; a non-raster, gimbal-free geometric formulation is an architectural choice rather than proof of universal rendering superiority.
- **Extended hypotheses:** Any higher-dimensional/five-component twistor extensions or quantum-number interpretations sit outside Theorems 1 and 2 unless explicitly formalized and verified.
- **Frozen scripts:** `palatial_gpo_ray_optics.py`, `argon_geometry_audit.py`, `argon_v4_adversarial_audit.py`, and `argon_v44_adversarial_audit.py` remain untouched when preserving audit baselines.