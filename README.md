# Twistor-Render-Core (v3.7-su22-renderer-aabb-bvh)

A PyTorch library for $\mathrm{SL}(2,\mathbb{C})$ spinor algebra, 15-parameter $\mathrm{SU}(2,2)$ conformal twistor transformations, pole-separated Penrose contour transforms, $L^2(\mathbb{CP}^1)$ spherical harmonics, and recursive AABB BVH ray-splat rendering.

---

## 1. Mathematical & Algorithmic Verification Suite (22 Unit Tests)

All claims and identities are verified by automated unit tests in `tests/`:

| Module & Test Suite | Mathematical / Algorithmic Property Tested | Verified Result |
| :--- | :--- | :--- |
| **1. SL(2,C) & SU(2,2) Algebra** (`twistor/incidence.py`, `twistor/penrose_core.py`) | Cayley-Hamilton Lie exponential; contragredient boost covariance $\pi' = \Lambda^{-\dagger}\pi$; full 15-parameter conformal group $G = K(b)D(\lambda)P(\Lambda, T)$ preserving $G^\dagger \Sigma G = \Sigma$; Klein quadric ray intersection; forward-tube splitting $\mathbb{PT}^+$. | Boost covariance: `4.85e-16`<br>15-param SU(2,2): `3.97e-15`<br>Tube norm on PN: `0.00e+00` |
| **2. Pole-Separated Penrose Transform & Cech Coboundaries** (`twistor/cohomology_penrose.py`) | Elementary state $f(Z) = ((A\cdot Z)(B\cdot Z))^{-1} \in \mathcal{O}(-2)$ without artificial epsilon regularizers; explicit pole tracking $\|\zeta_A(x)\| = 0.254 \lt 1 \lt \|\zeta_B(x)\| = 2.564$; non-zero field $\|\phi(x)\| = 0.3493$ solving $\Box \phi = 0$; $U_0 \cap U_1$ Cech coboundary invariance $f \mapsto f + h_0 - h_1$. | Homogeneity err: `1.62e-15`<br>Residue match: `1.14e-16`<br>Wave eq $\|\Box\phi\|/\|\phi\|$: `0.00e+00`<br>Coboundary diff: `5.72e-17` |
| **3. Spinor vs. Euclidean Baselines** (`twistor/localization.py`, `twistor/renderer.py`) | Documents and verifies that the spinor transverse frame $(a, b, n)$ satisfies $a a^\mathsf{T} + b b^\mathsf{T} = I_3 - n n^\mathsf{T}$ and the spinor ray residual satisfies $\|D\pi\|^2 - |\pi^\dagger D\pi|^2 = \|d\|^2 - (d\cdot n)^2$, matching standard Euclidean line-intersection and point-to-ray formulas to float64 precision. | Projector identity: `6.66e-16`<br>Solver discrepancy: `1.20e-15 m`<br>Ray distance diff: `5.39e-16` |
| **4. Recursive AABB BVH Tree** (`twistor/bvh.py`) | Top-down median-split Axis-Aligned Bounding Box (AABB) hierarchy with vectorized Kay-Kajiya slab ray-box culling, benchmarked against brute-force `TwistorRenderer`. | Prunes **89.9%** of ray-point pairs on clustered scenes; max diff vs. brute-force: `1.62e-08` |
| **5. L2(CP1) Spherical Harmonics** (`twistor/harmonics.py`) | Evaluates $\ell \le 2$ spinor bilinear harmonics on $S^2 \cong \mathbb{CP}^1$, verified against `scipy.special.sph_harm_y`, autograd Laplace-Beltrami eigenvalues $\ell(\ell+1)$, and Gauss-Legendre quadrature. | vs. SciPy: `5.75e-16`<br>Autograd PDE: `1.51e-14`<br>Orthonormality: `3.33e-15` |

---

## 2. Running the Full 22-Test Audit Suite

```powershell
$env:PYTHONPATH = "."
python -m unittest discover -s tests -p "test_*.py" -v
```