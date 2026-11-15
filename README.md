# Twistor-Render-Core (v3.8-adaptive-contour-cech)

A PyTorch library for $\mathrm{SL}(2,\mathbb{C})$ spinor algebra, 15-parameter $\mathrm{SU}(2,2)$ conformal twistor transformations, adaptive-contour Penrose transforms, explicit $U_0/U_1$ Cech open-cover holomorphicity checks, $L^2(\mathbb{CP}^1)$ spherical harmonics, and recursive AABB BVH ray-splat rendering.

---

## 1. Mathematical & Algorithmic Verification Suite (24 Unit Tests)

All claims and identities are verified by 24 automated unit tests in `tests/` and enforced via GitHub Actions CI (`.github/workflows/ci.yml`):

| Module & Test Suite | Mathematical / Algorithmic Property Tested | Verified Result |
| :--- | :--- | :--- |
| **1. Adaptive-Radius Penrose Transform** (`twistor/cohomology_penrose.py`, `test_3`) | Elementary state $f(Z) = ((A\cdot Z)(B\cdot Z))^{-1} \in \mathcal{O}(-2)$ without artificial regularizers; adaptive geometric-mean contour radius $\rho(x) = \sqrt{\|\zeta_A(x)\| \, \|\zeta_B(x)\|}$ tested across 1,944 random 4D points in $[-1,1]^4$ (including near-boundary poles where fixed $\|\zeta\|=1$ degrades to `2.87e-01` relative error). | Homogeneity err: `1.62e-15`<br>**Adaptive $\rho$ max err (1,944 pts): `4.75e-16`**<br>Wave eq $\|\Box\phi\|/\|\phi\|$: `5.96e-17` |
| **2. Explicit $U_0/U_1$ Cech Patch Holomorphicity** (`CechCohomologyClassH1`, `test_4`) | Verifies that 0-cochain sections $h_0(Z) = c_0 (C\cdot Z)^{-2}$ and $h_1(Z) = d_1 (D\cdot Z)^{-2}$ have no poles inside $U_0^{(\rho)} = \{\|\zeta\| \le \rho\}$ and $U_1^{(\rho)} = \{\|\zeta\| \ge \rho\} \cup \{\infty\}$, and strictly rejects invalid in-patch poles (`ValueError`). | Valid coboundary diff: `2.78e-17`<br>In-patch pole rejected: `True` |
| **3. SL(2,C) & 15-Parameter SU(2,2) Algebra** (`twistor/incidence.py`, `twistor/penrose_core.py`) | Cayley-Hamilton Lie exponential; contragredient boost covariance $\pi' = \Lambda^{-\dagger}\pi$; full 15-parameter conformal matrix $G = K(b)D(\lambda)P(\Lambda, T)$ preserving $G^\dagger \Sigma G = \Sigma$; Klein quadric ray intersection; forward-tube splitting $\mathbb{PT}^+$. | Boost covariance: `4.85e-16`<br>15-param $\mathrm{SU}(2,2)$: `3.97e-15`<br>Tube norm on $\mathbb{PN}$: `0.00e+00` |
| **4. Conformal Camera Ray Action & Native Spinor Rasterization** (`twistor/renderer.py`, `test_3`, `test_5`) | `TwistorRenderer.render()` evaluates the spinor Pauli perpendicular residual $\mathcal{R}^2 = \|D\pi\|^2 - |\pi^\dagger D\pi|^2$ directly (`5.39e-16` vs. Euclidean). `test_5` verifies $\omega=0$ projective no-op (`2.22e-16`) vs. active $\omega \neq 0$ ray deformation under $K(b)+C$ (`1.9795`), $K(b)+T$ (`1.2514`), and $D(\lambda)+K(b)$ (`0.5758`). | Projector identity: `6.66e-16`<br>Ray distance diff: `5.39e-16`<br>Origin $\omega=0$ no-op: `2.22e-16` |
| **5. Recursive AABB BVH Tree** (`twistor/bvh.py`, `test_4`) | Top-down median-split Axis-Aligned Bounding Box (AABB) hierarchy with vectorized Kay-Kajiya slab ray-box culling, benchmarked against brute-force `TwistorRenderer`. | Prunes **89.9%** of ray-point pairs on clustered scenes; max diff vs. brute-force: `1.62e-08` |
| **6. $L^2(\mathbb{CP}^1)$ Spherical Harmonics** (`twistor/harmonics.py`) | Evaluates $\ell \le 2$ spinor bilinear harmonics on the celestial sphere $S^2 \cong \mathbb{CP}^1$ ($\mathrm{SU}(2)$ Casimir spectrum $\ell(\ell+1)$), verified against `scipy.special.sph_harm_y`, autograd PDE eigenvalues, and Gauss-Legendre quadrature. | vs. SciPy: `5.75e-16`<br>Autograd PDE: `1.51e-14`<br>Orthonormality: `3.33e-15` |

---

## 2. Running the Full 24-Test Audit Suite

```powershell
$env:PYTHONPATH = "."
python -m unittest discover -s tests -p "test_*.py" -v
```