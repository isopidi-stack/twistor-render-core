# Argon v4.4 — Rendering and Numerical Guarantees (Scope and Evidence)

## Status Convention

- **Established geometric / algebraic bridge result:** Derived from `docs/THEOREMS_AND_PROOFS.md` under explicit assumptions.
- **Verified on reference module (`twistor/congruence.py`):** Confirmed by unit and compatibility tests on the new reference module; does not modify frozen legacy scripts or prove full-renderer visual perfection.
- **Open numerical / renderer-level obligation:** Requires finite-precision rounding/cancellation bounds, live renderer migration/integration, or image-level experiments.

---

## 1. Infinitesimal Conformal Footprint Transport (Bridge Lemma 1)

Along any ray of the Theorem 1 congruence ($\frac{dw}{ds} = 1$, so $w(s) = w_0 + s$ with $w_0 \ne 0$) and a parallel-transported right-handed orthonormal screen basis $(\mathbf{e}_1, \mathbf{e}_2)$:

1. Because $\sigma = 0$ and $\rho = \theta + i\omega = \frac{1}{\bar{w}}$, the transverse gradient in right-handed screen coordinates takes the conformal form $B(s) = \theta(s)I - \omega(s)J$ with $J = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}$.
2. In complex screen coordinates $Z(s) = y_1(s) + i y_2(s)$, infinitesimal separation evolves via $\frac{dZ}{ds} = \bar{\rho}\,Z = \frac{1}{w(s)}Z$, integrating to

$$\boxed{Z(s) = \frac{w_0 + s}{w_0}\,Z(0), \qquad F(s) = \begin{pmatrix}\mathrm{Re}\,q & -\mathrm{Im}\,q \\ \mathrm{Im}\,q & \mathrm{Re}\,q\end{pmatrix},\quad q = \frac{w_0 + s}{w_0}.}$$

3. Consequently, for infinitesimal circular cross-sections transported along the congruence:

$$\boxed{\kappa_2(F) = 1, \qquad \frac{A(s)}{A(s_0)} = \frac{|w(s)|^2}{|w(s_0)|^2}.}$$

**Scope and boundary:** This governs infinitesimal transverse ray-bundle transport along the congruence. It does **not** by itself guarantee that finite pixels retain shapes after projection, shading, occlusion, non-congruence secondary effects, or discrete sampling.

---

## 2. Sheet Identity, Lifted-Disk Continuity, and Sheet-Merge Discontinuity (Bridge Lemma 2)

Under the deck transformation $\tau : (\mathbf{x}, w) \mapsto (\mathbf{x}, -w)$ on $\widetilde{\Omega}_b$, the tangent vector difference satisfies the global identity:

$$\boxed{D(\mathbf{x}, w) = \|\mathbf{k}(\mathbf{x}, w) - \mathbf{k}(\mathbf{x}, -w)\| = \frac{2|w|}{\sqrt{\alpha^2 + b^2}}.}$$

- **Strict separation off $\mathcal{B}_b$:** Because $w \ne 0$ throughout $\widetilde{\Omega}_b$, $D(\mathbf{x}, w) > 0$ everywhere away from the branch ring $\mathcal{B}_b$.
- **Spanning-disk interior vs. $z$-axis:** On the interior of the lifted spanning disk ($\alpha = 0$, $z = 0$, $r = \sqrt{x^2+y^2} < |b|$), the mismatch simplifies to $D(r) = 2\sqrt{1 - r^2/b^2}$, reaching $D = 2$ only at the disk center $(r = 0, z = 0)$ and tending to $0$ as $r \to |b|$. Globally across $\widetilde{\Omega}_b$, $D = 2$ holds everywhere on the $z$-axis ($x = y = 0$), including points outside the spanning disk, and approaches $2$ at large transverse radius for fixed $z$.
- **Incremental continuation vs. stateless square root:** Tracking sheet state incrementally across a lifted-disk crossing keeps $w(s) = w_0 + s$ and $\mathbf{k}$ smooth, whereas a stateless principal square root switching branches across the cut introduces a field jump of magnitude $D(r_*)$. Whether a field jump produces a visible image seam depends on the rendering and shading pipeline.

---

## 3. Sensitivity, Two-Chart Atlas, and Open Implementation Obligations

- **Intrinsic operator-norm sensitivity:** From $\|\nabla\mathbf{k}\|_F^2 = \frac{2}{|w|^2}$, geodesicity, and shear-freeness ($B^T B = \frac{1}{|w|^2}I_2$ in the transverse screen frame), the local operator norm satisfies $\|\nabla\mathbf{k}\|_2 = \frac{1}{|w|} \le \frac{\sqrt{2}}{|w|} = \|\nabla\mathbf{k}\|_F$. Intrinsic sensitivity grows as $|w| \to 0$ near $\mathcal{B}_b$, whereas the spanning-disk interior ($|w| = \sqrt{b^2 - r^2} > 0$) is geometrically nonsingular.
- **Bounded two-chart atlas:** $\zeta\eta = 1$ and $\mathbf{m}_{\mathrm{N}}(1/\zeta) = \frac{\bar{\zeta}}{\zeta}\mathbf{m}_{\mathrm{S}}(\zeta)$ provide a bounded-coordinate representation ($\min(|\zeta|, |\eta|) \le 1$) without $z$-axis $0/0$ division in `twistor/congruence.py`.
- **Open numerical and renderer obligations:**
  - Formal floating-point handoff error bounds near $|\zeta| = 1$ and cancellation/rounding bounds in evaluating $w$ or $\Delta$ (distinct from intrinsic sensitivity $\|\nabla\mathbf{k}\|_2$).
  - Full integration of `twistor/congruence.py` into camera/renderer pipelines and image-level visual verification remain separate engineering steps.

---

## 4. Measured Reference & Legacy Compatibility Evidence

The following values were measured in the reported test run on the reference module and frozen legacy file (`palatial_gpo_ray_optics.py`, SHA-256 `8809F3AF1D5BBB43202128C740306402D9BFF264911146CB5DB974EBF9716217`, left unmodified):

| Diagnostic | Reported value | Scope / notes |
| :--- | :--- | :--- |
| Legacy Sachs shear ($\max|\sigma|$) | `1.389178` | Frozen `legacy_gpo.verify_kerr_theorem_shear_free_pde()` with reversed $k_z$ sign |
| Corrected Sachs shear ($\max|\sigma|$) | `4.791e-11` | Controlled test flipping only $k_z$ sign and paired $\mathbf{m}_{\mathrm{S}}$ on identical grid |
| Corrected optical-scalar residual | `4.895e-11` | Max difference vs. $\rho = 1/\bar{w}$ on identical grid |
| Corrected tangent-field residual | `6.107e-16` | Max norm difference vs. `twistor.congruence.tangent_field` on identical grid |
| Reference unit tests | `7 passed` | `tests/test_congruence.py` (0.029s) |
| Legacy compatibility checks | `3 passed` | `tests/diagnose_legacy_gpo_compat.py` (hash, legacy vs. fixed shear, $z$-axis regularity) |

### Reproducibility Provenance

- **Repository commit (HEAD):** `ed6743d`
- **Python / PyTorch:** Python `3.13.1`, PyTorch `2.15.0+cpu`
- **Commands run:**
  - `& $Py -m unittest discover -s tests -p "test_congruence.py" -v`
  - `& $Py tests\diagnose_legacy_gpo_compat.py`

---

## 5. Intrinsic Conditioning vs. Numerical Evaluation Error (Bridge Lemma 3)

Let $M = x^2 + y^2 + z^2 + b^2$ and $\delta = |\widehat{\Delta} - \Delta|$ for $\Delta = x^2 + y^2 + (z + ib)^2$ and $w^2 = \Delta$ with $w \ne 0$.

### 5.1 Exact Geometric Conditioning (Away from $\mathcal{B}_b$)

- **Tangent field $\mathbf{k}$:** The real $3 \times 3$ spatial Jacobian has singular values $\left(\frac{1}{|w|},\; \frac{1}{|w|},\; 0\right)$ (with the zero singular value along the geodesic direction $\mathbf{k}$), giving

$$\boxed{\|\nabla\mathbf{k}\|_2 = \frac{1}{|w|}, \qquad \|\nabla\mathbf{k}\|_F^2 = \frac{2}{|w|^2}.}$$

- **Covering root $w$ and optical scalar $\rho = \frac{1}{\bar{w}}$:** From $\nabla w = \frac{(x, y, z + ib)}{w}$ and $\nabla\rho = -\frac{\nabla\bar{w}}{\bar{w}^2}$, their spatial derivative norms satisfy

$$\boxed{\|\nabla w\| = \frac{\sqrt{M}}{|w|}, \qquad \|\nabla\rho\| = \frac{\sqrt{M}}{|w|^3}.}$$

Near the branch ring $\mathcal{B}_b$ for fixed $b \ne 0$, $\|\nabla\mathbf{k}\|_2 \sim |w|^{-1}$ and $\|\nabla\rho\| \sim \sqrt{2}|b|\,|w|^{-3}$, whereas strictly inside the lifted spanning disk ($z = 0$, $r = \sqrt{x^2+y^2} < |b|$), $|w| = \sqrt{b^2 - r^2} > 0$ is geometrically nonsingular.

### 5.2 Conditional Arithmetic-Model Bounds and Sheet Continuation

- **Conditional evaluation error of $\widehat{\Delta}$ and $\widetilde{w}$:** Under a **conditional arithmetic-model bound** assuming a specified floating-point operation graph, finite normal intermediates without overflow/underflow, and defined complex arithmetic handling, suppose $\delta = |\widehat{\Delta} - \Delta| \le \gamma_c M$ for a graph-specific constant $c$ (not a backend-independent theorem). With $\widehat{\Delta} = \Delta + e$ ($|e| \le \delta < |\Delta|$) and cancellation-amplification factor $\mathcal{C}_\Delta = \frac{M}{|\Delta|}$, the square root $\widetilde{w}$ analytically continued from $w$ satisfies

$$\boxed{\frac{|\widetilde{w} - w|}{|w|} \le 1 - \sqrt{1 - \frac{\delta}{|\Delta|}} \le \frac{\delta}{|\Delta|},}$$

with relative-error scale $\approx \frac{\delta}{2|\Delta|}$ for small perturbations (excluding branch-cut flips of a stateless principal square root or additional backend square-root rounding error).

- **Sufficient condition for sheet continuation:** Selecting the root with positive real inner product against the preceding root is unambiguous when $\mu_n = \mathrm{Re}(w_n\overline{w_{n-1}}) > 0$, i.e., $|\arg(w_n/w_{n-1})| < \frac{\pi}{2}$. Given root error bounds $\varepsilon_n, \varepsilon_{n-1}$, sign selection is preserved under

$$\boxed{\mathrm{Re}(w_n\overline{w_{n-1}}) > |w_n|\varepsilon_{n-1} + |w_{n-1}|\varepsilon_n + \varepsilon_n\varepsilon_{n-1}.}$$

### 5.3 Two-Chart Handoff: Bounded Coordinates vs. Gauge Continuity

- Because $1 + k_z \ge 1$ for $k_z \ge 0$ and $1 - k_z \ge 1$ for $k_z < 0$, the selected coordinate chart satisfies $\min(|\zeta|, |\eta|) \le 1$ without pole blow-up.
- Because $\mathbf{m}_{\mathrm{N}}(1/\zeta) = \frac{\bar{\zeta}}{\zeta}\mathbf{m}_{\mathrm{S}}(\zeta)$ is a unitary phase transition rather than equality of raw frame components, numerical tests must verify both continuity of gauge-invariant quantities and explicit phase covariance of gauge-dependent screen quantities across chart switches.
- **Open backend obligation:** Backend-specific numerical guarantees remain open until tested empirically on the target PyTorch backend with a higher-precision oracle, accounting for input representation, complex square-root algorithm, IEEE-754 exceptions, and graph constants.

---

## 6. Conditional Sheet Continuation and Adaptive Step Control (Bridge Lemma 4)

- **Three-part structure (Lemmas 4A / 4B / 4C):**
  - *Lemma 4A (Proven):* Exact Robinson-ray phase margin $\mu_n = |w_n|^2 + h\alpha_n = \alpha_n(\alpha_n + h) + \beta_n^2$ and safe-step ceiling, distinguishing algorithmic phase failure from geometric singularities.
  - *Lemma 4B (Proven for endpoint sign preservation under explicit assumptions):* Preserving analytic continuation on a general curved path additionally requires a separate segment phase-variation bound preventing winding around $\mathcal{B}_b$; straight Robinson rays in Lemmas 4A/4C avoid winding via $w(s_n + t) = w_n + t$ and mandatory segment clearance.
  - *Lemma 4C (Conditional sufficient conditions derived; backend-certified enclosures open):* Two complementary Robinson-ray selectors (`phase_margin_step_bounds`, `ray_predictor_step_bounds`) plus mandatory segment clearance (`ray_segment_clearance_bounds`).
- **Controller failure & overflow policy (`adaptive_continue_ray_step`):**
  - **Contract errors raise `ValueError`:** Invalid shapes or dtypes (complex `pos`/`k_dir`, non-`(3,)` real vectors, non-scalar `prev_w` or candidate root), non-integer or boolean `max_subdivisions`, non-finite caller inputs, or invalid scalar bounds raise `ValueError`.
  - **Numerical step failures subdivide/reject:** Non-finite candidate-root values (`NaN`/`inf` from `eval_cand`), non-finite step intermediates (`pos_next`, `w_pred`, inner product, or distances), insufficient segment clearance, or subnormal step halving underflow (`reason="step_underflow"`) trigger controlled rejection via `SheetStepResult`.
  - **Overflow guarantee:** An overflowing criterion (or overflowing selector arithmetic for that criterion) cannot itself authorize acceptance (it evaluates to `criterion_satisfied=False`). In `dual` mode, another non-overflowing criterion may still authorize acceptance if its own arithmetic is finite and its assumptions hold.
  - **Dual-mode diagnostic semantics:** Mandatory segment clearance is checked first. If both criteria are satisfied, sign agreement is required (disagreement rejects with `reason="invariant_violation_dual_sign_disagreement"` as a defensive assumption-violation diagnostic). If exactly one is satisfied, the step is accepted with `satisfied_by` naming that criterion while the other is reported inconclusive. If neither is satisfied, the controller subdivides up to budget (or until `step_underflow`) and then rejects, reporting the exact last evaluated step size.
- **Conditional scope & preconditions:** The reference controller checks Lemma 4C inequalities against **caller-supplied** bounds (`eps_curr`, `pred_add_roundoff`, `eps_next`, `ip_roundoff`, `cmp_roundoff`) in `float64`; it does **not** internally derive those bounds, establish that `pos + h * k_dir` is an exact Robinson-ray endpoint, or certify ray-position, tangent, or image-space accuracy. Independent directed-rounding `Decimal` interval checks live in `tests/test_sheet_continuation.py` (they validate the tested inequalities independently without turning the controller's `float64` decisions into certified interval decisions).
