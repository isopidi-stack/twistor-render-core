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