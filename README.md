# Argon (`twistor-render-core` — Argon v4.4)

Argon (`twistor-render-core`) is a computational geometry and rendering engine exploring spinor/twistor parameterizations, Robinson congruences, and Hopf–Villarceau circle correspondences.

For formal theorem statements, proof status, and scope boundaries, see **[`docs/THEOREMS_AND_PROOFS.md`](docs/THEOREMS_AND_PROOFS.md)**. For rendering and numerical guarantees, scope, and evidence, see **[`docs/RENDERING_GUARANTEES.md`](docs/RENDERING_GUARANTEES.md)**.

## Research progress wall — 25 November 2026

> **Current research checkpoint: Bridge 9I-A.** Restricted live preservation evidence is owner-accepted and closed. The overall gate remains open for native affine instruction evidence and the final implementation contract and fixture freeze. This is a documentation-only research record, not a promotion of local numerical implementations to GitHub.

| Research milestone | Qualified status |
| --- | --- |
| Bridge 7B | Conditional T1–T4 numerical certification, restricted local scope |
| Bridge 7C–7D | Conditional a posteriori certificate and exact selection |
| Bridge 8 | Restricted rational-witness initialization and recursive chaining |
| Bridge 8F | Bounded adaptive rational-witness continuation — owner accepted and closed |
| Bridge 9A–9F | Exact algebraic geometry, degree-four bound and conditional theorem transfer — owner accepted |
| Bridge 9H | Standalone exact-algebraic reference — owner accepted and closed; simulation only |
| Bridge 9I-A: static source reconciliation | Complete within authenticated reconstructed-source scope; 26/26 source pins |
| Bridge 9I-A: live preservation evidence | Owner accepted and closed; 697/697 captured file hash records, six worktrees, 25/25 independent evidence checks |
| Bridge 9I-A: native affine instructions | **Next — unresolved** |
| Bridge 9I-A: final contract and fixtures | Pending review and freeze |
| Bridge 9I-B–G / 9J | Runtime implementation, verification, integration and promotion not authorized |
| Renderer migration / aggregate certification | Locked |

**Verified research evidence:** 14/14 standalone reference tests on Linux and Windows, 23/23 deterministic counters, 7,193 reported independent oracle assertions, 52 simulated schedules and zero native invocations in the 9H reference suite. Live preservation compares captured hash records; it is not continuous attestation or verification of ignored files.

**[Full research checkpoint, evidence fingerprints, limitations and exact resumption instructions](docs/BRIDGE_9_PROGRESS_CHECKPOINT_2026-11-25.md)**

### Publication and preservation boundary

- Pre-documentation GitHub main baseline: `d0db6f43827218ef8b0ea5447edc1a385fffd5e6` (Bridge 6).
- Remote main received a single documentation-only checkpoint commit `f0f11adb8b08306b60fc103e3cf09de4b251d5be`. This README wall is maintained separately on the documentation branch; no further main update is intended.
- Accepted local Bridge 8F checkpoint: `76e6267a63d6e3ccde873959edecf3f3ce27aa83`. The later local executable implementations are not implied to be on GitHub main.
- Preserve the six local worktrees, 697 enumerated files, accepted executable bytes, references and original 9H archive. Do not pull or merge this documentation into the frozen local baseline without explicit review and re-baselining.
- Both certification locks remain false in the accepted local scope: `_7B_INTEGRATION_ACCEPTED = False` and `all_certified = False`. No new runtime implementation or native probing is authorized.

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