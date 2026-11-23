# Argon v4.4 — Theorems, Proof Status, and Scope

## Status Convention

- **Established:** Derived algebraically under stated assumptions.
- **Proof obligation:** Requires a complete, self-contained formal derivation.
- **Numerical / architectural scope:** Verified on tested configurations or architectural choices, not a global mathematical or physical proof.

---

## Theorem 1 — Two-Sheeted Covering and Screen-Frame Transition

Let $b \in \mathbb{R} \setminus \{0\}$ and define

$$\Delta(x,y,z) = (z + ib)^2 + x^2 + y^2.$$

Its zero set is the branch ring

$$\mathcal{B}_b = \{z = 0,\; x^2 + y^2 = b^2\}.$$

Define the two-sheeted covering space

$$\widetilde{\Omega}_b = \{(\mathbf{x}, w) : \mathbf{x} \notin \mathcal{B}_b,\; w^2 = \Delta(\mathbf{x})\},$$

with sheet involution $w \mapsto -w$. On the disk-cut complement

$$\Omega_b^{\mathrm{cut}} = \mathbb{R}^3 \setminus \{z = 0,\; x^2 + y^2 \le b^2\},$$

$\Delta$ avoids the nonpositive real axis, so the principal square root is continuous there and selects the branch with $\operatorname{Re} w > 0$.

### Established — Chart Transition

Where both denominators are nonzero, define

$$\zeta = \frac{x + iy}{z + ib + w}, \qquad \eta = -\frac{x - iy}{z + ib - w}.$$

Since

$$(z + ib + w)(z + ib - w) = -(x^2 + y^2),$$

we obtain

$$\boxed{\zeta \eta = 1}$$

on the chart overlap.

### Established — Screen-Frame Transition

Define

$$\mathbf{m}_{\mathrm{S}}(\zeta) = \frac{(1 - \zeta^2,\; i(1 + \zeta^2),\; -2\zeta)}{\sqrt{2}(1 + |\zeta|^2)}$$

and

$$\mathbf{m}_{\mathrm{N}}(\eta) = \frac{(\eta^2 - 1,\; i(\eta^2 + 1),\; -2\eta)}{\sqrt{2}(1 + |\eta|^2)}.$$

Substituting $\eta = 1/\zeta$ gives

$$\boxed{\mathbf{m}_{\mathrm{N}}(1/\zeta) = \frac{\bar{\zeta}}{\zeta}\,\mathbf{m}_{\mathrm{S}}(\zeta)}.$$

For $\zeta = |\zeta|e^{i\phi}$, this phase factor has unit modulus ($\bar{\zeta}/\zeta = e^{-2i\phi}$), preserving complex screen orientation across the overlap.

### Proof Obligation — Optical Scalars

Deriving and certifying

$$\sigma = 0, \qquad \rho = \frac{1}{w}$$

requires an explicit congruence tangent field, derivative/projection operator, expansion/twist/shear sign conventions, and consistent sheet-to-sheet treatment on $\widetilde{\Omega}_b$.

---

## Theorem 2 — Restricted Robinson–Villarceau Construction and Explicit Inverse

Define the restricted oriented-ray domain

$$\mathcal{D}_{\mathrm{Rob}} = \{(\mathbf{x}_0, \mathbf{k}) : |\mathbf{k}| = 1,\; -1 < k_z < 1,\; z_0 = 0,\; \mathbf{x}_0 \ne \mathbf{0},\; x_0 k_x + y_0 k_y = 0\}.$$

The forward map from $(\mathbf{x}_0, \mathbf{k}) \in \mathcal{D}_{\mathrm{Rob}}$ to parameters $(\zeta, b) \in \mathbb{C}^* \times \mathbb{R}^*$ is given by

$$\zeta = \frac{k_x + i k_y}{1 + k_z}, \qquad b = -\frac{x_0 k_y - y_0 k_x}{1 - k_z^2}.$$

Because $k_z \in (-1, 1)$ and $\mathbf{x}_0 \ne \mathbf{0}$ with $x_0 k_x + y_0 k_y = 0$, we have $\zeta = a + ic \ne 0$ and $b \ne 0$. With $D = \sqrt{1 + a^2 + c^2}$, define the Hopf lift

$$(Z_0, Z_1) = \frac{e^{iu}(1, \zeta)}{D}$$

and stereographic image

$$\gamma_{\zeta,b}(u) = \frac{b}{1 - \operatorname{Im} Z_1}\bigl(\operatorname{Re} Z_0,\; \operatorname{Im} Z_0,\; \operatorname{Re} Z_1\bigr).$$

### Established — Finiteness, Containing Plane, Radial Extrema, 3D Center, and Explicit Inverse

1. **Finiteness and containing plane:** Setting $t = \frac{c\cos u + a\sin u}{D}$ gives $|\operatorname{Im} Z_1| \le \frac{|\zeta|}{D} < 1$, so $1 - \operatorname{Im} Z_1 > 0$ and $\gamma_{\zeta,b}(u)$ is a finite closed curve. Because $\operatorname{Re} Z_1 = a\operatorname{Re} Z_0 - c\operatorname{Im} Z_0$, $\gamma_{\zeta,b} = (X, Y, Z)$ satisfies

$$\boxed{Z = aX - cY},$$

lying on the plane through the origin with normal $\mathbf{N} = (-a, c, 1)$ (non-unit scaling).

2. **Radial extrema:** Since $|Z_0|^2 + |Z_1|^2 = 1$, $|\gamma_{\zeta,b}(u)|^2 = b^2\frac{1 + \operatorname{Im} Z_1(u)}{1 - \operatorname{Im} Z_1(u)}$. With $q = \frac{|\zeta|}{D} < 1$, the extremal radii satisfy

$$r_{\max}^2 = b^2\frac{1 + q}{1 - q}, \qquad r_{\min}^2 = b^2\frac{1 - q}{1 + q}, \qquad \boxed{r_{\max} r_{\min} = b^2}.$$

3. **Exact 3D circle center and radius:** Eliminating $t$ from $cX + aY = \frac{bt}{1-t}$ and $X^2 + Y^2 + Z^2 = b^2\frac{1+t}{1-t}$ gives

$$\boxed{X^2 + Y^2 + Z^2 = b^2 + 2b(cX + aY)},$$

which completes the square to

$$\boxed{(X - bc)^2 + (Y - ba)^2 + Z^2 = b^2(1 + a^2 + c^2)}.$$

Since $-a(bc) + c(ba) = 0$, the center lies on $Z = aX - cY$, yielding the exact 3D circle center and radius:

$$\boxed{\mathbf{C}_{\zeta,b} = (bc, ba, 0)}, \qquad \boxed{R_{\zeta,b} = |b|\sqrt{1 + |\zeta|^2}}.$$

4. **Point-set separation, explicit ray reconstruction, and injectivity:** Because $\mathbf{C}_{\zeta,-b} = -\mathbf{C}_{\zeta,b}$ and $\zeta \ne 0, b \ne 0$, we have $\mathbf{C}_{\zeta,b} \ne \mathbf{0}$, so the unparameterized point sets $\mathcal{S}_{\zeta,b}$ and $\mathcal{S}_{\zeta,-b}$ are distinct. Recovering $(a, c)$ from the plane normal scaled to have third component equal to 1, namely $(-a, c, 1)$, together with $\mathbf{C} = (C_x, C_y, C_z)$ gives

$$\boxed{b = \frac{c C_x + a C_y}{a^2 + c^2}},$$

and the oriented ray is reconstructed explicitly by

$$\mathbf{k} = \frac{(2a,\; 2c,\; 1 - a^2 - c^2)}{1 + a^2 + c^2}, \qquad \mathbf{x}_0 = (-b k_y,\; b k_x,\; 0).$$

Substituting the recovered $(a, c, b)$ into the forward formulas reproduces the original $(\zeta, b)$. The reconstructed $\mathbf{k}$ has unit norm and satisfies $-1 < k_z < 1$; the reconstructed $\mathbf{x}_0$ lies in the horizontal plane, is nonzero, and is orthogonal to $(k_x, k_y)$. Hence the inverse maps the forward-defined circle family back into $\mathcal{D}_{\mathrm{Rob}}$, establishing injectivity on the restricted domain (with surjectivity onto the forward image immediate by definition).

### Proof Obligations — Independent Target Characterization

- If the target circle family is defined geometrically without reference to the forward map from $\mathcal{D}_{\mathrm{Rob}}$, prove geometric characterization and surjectivity explicitly.

---

## Scope & Evidence Boundaries

- **Numerical regression & SHA-256 checks:** Reported regression suites and digests verify consistency on tested configurations and pinned files; they do not substitute for formal proofs or establish historical novelty.
- **Architectural & scientific scope:** Non-raster / gimbal-free design choices do not establish infinite precision or universal renderer superiority, and separate hypotheses (such as $\mathbb{C}^5$ extensions or quantum-number interpretations) are outside Theorems 1 and 2.