# Argon (`twistor-render-core` — Argon v4.4)

Argon (`twistor-render-core`) is a computational geometry and rendering engine exploring spinor/twistor parameterizations, Robinson congruences, and Hopf–Villarceau circle correspondences.

For formal theorem statements, proof status, and scope boundaries, see **[`docs/THEOREMS_AND_PROOFS.md`](docs/THEOREMS_AND_PROOFS.md)**. For rendering and numerical guarantees, scope, and evidence, see **[`docs/RENDERING_GUARANTEES.md`](docs/RENDERING_GUARANTEES.md)**.

## Research progress wall — 26 November 2026

> **Current checkpoint: 9I-D owner closed; 9I-E NOT_ACCEPTED.** Original adaptive R1 Windows campaign failed closed in group 27 on a transient DLL hash mismatch. Forensic stages 0–2 completed within diagnostic scope; cause remains unresolved. Owner cancelled BIOS flash and external software installation. Next: document original BIOS settings, disable XMP for JEDEC defaults, verify post-restart inventory, then ONE unchanged full R1 campaign. No new campaign or memory-change verification yet. Documentation only; not executable promotion.

| Research milestone | Qualified status |
| --- | --- |
| Bridge 7B–7D | Conditional T1–T4 / a posteriori bounds |
| Bridge 8F | Owner accepted; frozen local checkpoint |
| Bridge 9A–F / 9H | Mathematical scope and standalone simulation reference accepted |
| Bridge 9I-A R2 / B R3 / C R1 | Owner accepted and closed within qualified scope |
| Bridge 9I-D R1 | Owner accepted and closed; 70 transitions, 2590 predicates, 57 continuity bindings, 355 historical receipt checks |
| Bridge 9I-E.1–3 | Additive adaptive review and split local diagnostics complete |
| Bridge 9I-E.4 | Original pinned run 8/9 groups; 100 verified transitions, 3700 predicates, 83 continuity; NOT_ACCEPTED |
| Forensic 0–2 | Source audit, 600/600 repeated DLL comparisons, 114/114 instrumented DLL comparisons; no root cause established |
| Stage 3B-R | **Current: zero-flash JEDEC memory-only stabilization; post-change confirmation pending** |
| Bridge 9I-E.5 | OPEN; composite 101/3737/84 diagnostic only |
| Bridge 9I-F / G / 9J | Pending separate review and authorization |
| Renderer/video pipelines / standalone builds / aggregate certification | Deferred; locked |

**Next acceptance bar:** a single fresh full 9/9 pinned R1 run, 101 verified transitions, 3737 independent predicates, 84 continuity bindings, 697/697 pre/post preservation, 26/26 source pins, zero unexpected integrity failures and independent offline reconciliation. Expected injected controls remain expected. Stop on any unexpected authentication failure; preserve all evidence, no retry until green. Auditor sign-off and explicit owner closure required.

**[START HERE — full recovery checkpoint, evidence registry, forensic history and exact resumption instructions](docs/BRIDGE_9_PROGRESS_CHECKPOINT_2026-11-26.md)**

[Previous checkpoint (25 November)](docs/BRIDGE_9_PROGRESS_CHECKPOINT_2026-11-25.md)

### Publication and preservation boundary

- GitHub main remains at `f0f11adb8b08306b60fc103e3cf09de4b251d5be`; accepted local Bridge 8F checkpoint `76e6267a63d6e3ccde873959edecf3f3ce27aa83`. Later executable review packages are separately retained, not promoted.
- Preserve all six local worktrees, 697 enumerated files, 26 source pins, accepted and failed artifacts. Do not pull or merge this documentation into frozen baselines.
- Original host: i7-13700K, BIOS 0402 / reported microcode 0x10E, non-ECC DDR4. JEDEC target 2400 MT/s / 1.20 V; actual post-change state pending. BIOS/ME update and external diagnostic installs declined by owner. Known firmware risk and original unresolved transient failure remain disclosed.
- Both locks remain false: `_7B_INTEGRATION_ACCEPTED = False`, `all_certified = False`. T1–T4 conditional assumptions and Q1–Q4 qualified trace deferral remain. No universal hardware guarantee, guaranteed arrival, integration, promotion, pin replacement or downstream authorization.

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