# Argon v4.4 — Theorems, Proof Status, and Scope

## Status Convention

- **Established:** Derived algebraically under stated assumptions.
- **Proof obligation:** Requires a complete, self-contained formal derivation.
- **Numerical / architectural scope:** Verified on tested configurations or architectural choices, not a global mathematical or physical proof.

---

## Theorem 1 — Two-Sheeted Covering, Congruence Optics, and Screen-Frame Transition

Let $b \in \mathbb{R} \setminus \{0\}$ and define

$$\Delta(x,y,z) = (z + ib)^2 + x^2 + y^2.$$

Its zero set is the branch ring

$$\mathcal{B}_b = \{z = 0,\; x^2 + y^2 = b^2\}.$$

Define the two-sheeted covering space

$$\widetilde{\Omega}_b = \{(\mathbf{x}, w) : \mathbf{x} \notin \mathcal{B}_b,\; w^2 = \Delta(\mathbf{x})\},$$

with deck involution $\tau : (\mathbf{x}, w) \mapsto (\mathbf{x}, -w)$. On the disk-cut complement

$$\Omega_b^{\mathrm{cut}} = \mathbb{R}^3 \setminus \{z = 0,\; x^2 + y^2 \le b^2\},$$

$\Delta$ avoids the nonpositive real axis, so the principal square root is continuous there and selects the branch with $\operatorname{Re} w > 0$.

### Established — Chart and Screen-Frame Transitions

Where both denominators are nonzero, define

$$\zeta = \frac{x + iy}{z + ib + w}, \qquad \eta = -\frac{x - iy}{z + ib - w}.$$

Since

$$(z + ib + w)(z + ib - w) = -(x^2 + y^2),$$

we obtain

$$\boxed{\zeta \eta = 1}$$

on the chart overlap.

With

$$\mathbf{m}_{\mathrm{S}}(\zeta) = \frac{(1 - \zeta^2,\; i(1 + \zeta^2),\; -2\zeta)}{\sqrt{2}(1 + |\zeta|^2)}, \qquad \mathbf{m}_{\mathrm{N}}(\eta) = \frac{(\eta^2 - 1,\; i(\eta^2 + 1),\; -2\eta)}{\sqrt{2}(1 + |\eta|^2)},$$

substituting $\eta = 1/\zeta$ gives

$$\boxed{\mathbf{m}_{\mathrm{N}}(1/\zeta) = \frac{\bar{\zeta}}{\zeta}\,\mathbf{m}_{\mathrm{S}}(\zeta)},$$

which has unit modulus ($\bar{\zeta}/\zeta = e^{-2i\phi}$ for $\zeta = |\zeta|e^{i\phi}$) and preserves complex screen orientation across the overlap.

### Established — Congruence Tangent Field, Null Lift, Optical Tensor, and Sheet Transformation

Writing $w = \alpha + i\beta$ ($\alpha, \beta \in \mathbb{R}$) with $\alpha^2 - \beta^2 = x^2 + y^2 + z^2 - b^2$ and $\alpha\beta = bz$, define the real unit vector field on $\widetilde{\Omega}_b$:

$$\mathbf{k} = \left(\frac{\alpha x + by}{\alpha^2 + b^2},\; \frac{\alpha y - bx}{\alpha^2 + b^2},\; \frac{\beta}{b}\right).$$

Equipping $\mathbb{R}^{1,3}$ with Minkowski metric $g_{\mu\nu} = \operatorname{diag}(1, -1, -1, -1)$ and taking $L^\mu = \frac{1}{2}(1, -\mathbf{k})$ and $m^\mu = (0, \mathbf{m}_{\mathrm{S}})$ (so $K^2 = L^2 = 0$, $K \cdot L = 1$, and $m \cdot \bar{m} = -1$), the stationary four-dimensional null lift

$$K^\mu = (1, k_x, k_y, k_z)$$

satisfies $K^\mu K_\mu = 1 - |\mathbf{k}|^2 = 0$ and $K^\nu \partial_\nu K^\mu = (0, (\mathbf{k}\cdot\nabla)\mathbf{k})$, so spatial geodesicity implies affine null geodesicity ($K^\nu \partial_\nu K^\mu = 0$). For spatial screen vectors $E_A^\mu = (0, \mathbf{e}_A)$, the covariant Minkowski screen projection satisfies $E_{A\mu} E_B^\nu \partial_\nu K^\mu = -e_A^i e_B^j \partial_j k_i = -B_{AB}$, so the Euclidean spatial tensor below corresponds to the null screen-projected gradient up to this explicit metric sign convention.

At any point where a local orthonormal frame $(\mathbf{e}_1, \mathbf{e}_2, \mathbf{k})$ is chosen to be right-handed, define the transverse gradient matrix

$$B_{AB} = e_A^i e_B^j \partial_j k_i \qquad (A, B \in \{1, 2\}),$$

with expansion $\theta$, twist $\omega$, complex shear $\sigma$, and complex optical scalar $\rho$ defined by

$$\theta = \frac{1}{2}(B_{11} + B_{22}), \qquad \omega = \frac{1}{2}(B_{12} - B_{21}), \qquad \sigma = \frac{1}{2}(B_{11} - B_{22}) + \frac{i}{2}(B_{12} + B_{21}), \qquad \rho = \theta + i\omega.$$

1. **Explicit derivatives on $\{\alpha \ne 0\}$:** From $w^2 = (z + ib)^2 + x^2 + y^2$, we have $\partial_x w = x/w$, $\partial_y w = y/w$, and $\partial_z w = (z + ib)/w$. On $\{\alpha \ne 0\}$, $\beta = bz/\alpha$, $x^2 + y^2 = \frac{(\alpha^2 + b^2)(\alpha^2 - z^2)}{\alpha^2}$, $k_x + ik_y = \frac{x + iy}{\alpha + ib}$, and $k_z = \frac{z}{\alpha}$:
   - **Directional derivative and geodesicity:**

$$\mathbf{k} \cdot \nabla w = \frac{1}{w}\left[\frac{\alpha(x^2 + y^2)}{\alpha^2 + b^2} + \frac{z(z + ib)}{\alpha}\right] = \frac{\alpha^2 + ibz}{\alpha w} = 1,$$

so $\mathbf{k} \cdot \nabla \alpha = 1$, $\mathbf{k} \cdot \nabla \beta = 0$, $(\mathbf{k} \cdot \nabla)k_z = 0$, and $(\mathbf{k}\cdot\nabla)(k_x + ik_y) = \frac{(k_x + ik_y)(\alpha + ib) - (x + iy)(\mathbf{k}\cdot\nabla\alpha)}{(\alpha + ib)^2} = 0$, giving $(\mathbf{k} \cdot \nabla)\mathbf{k} = \mathbf{0}$.
   - **Divergence:** Differentiating $\alpha = \operatorname{Re} w$ gives $\partial_x \alpha = \frac{x\alpha}{\alpha^2+\beta^2}$, $\partial_y \alpha = \frac{y\alpha}{\alpha^2+\beta^2}$, and $\partial_z \alpha = \frac{z(\alpha^2+b^2)}{\alpha(\alpha^2+\beta^2)}$. Summing $\partial_x k_x + \partial_y k_y = \operatorname{Re}\!\left[\frac{2}{\alpha+ib} - \frac{(x^2+y^2)\alpha}{(\alpha^2+\beta^2)(\alpha+ib)^2}\right]$ with $\partial_z k_z = \frac{1}{\alpha} - \frac{z^2(\alpha^2+b^2)}{\alpha^3(\alpha^2+\beta^2)}$ cancels all $z^2$ terms to leave

$$\nabla \cdot \mathbf{k} = \frac{2\alpha}{\alpha^2 + \beta^2}.$$

   - **Curl / Beltrami derivation:** Computing $(\nabla \times \mathbf{k})_z = \partial_x k_y - \partial_y k_x = \operatorname{Im}\!\left[\frac{2}{\alpha+ib} - \frac{(x^2+y^2)\alpha}{(\alpha^2+\beta^2)(\alpha+ib)^2}\right]$ gives

$$\partial_x k_y - \partial_y k_x = -\frac{2b}{\alpha^2 + b^2} + \frac{2b(\alpha^2 - z^2)}{(\alpha^2 + \beta^2)(\alpha^2 + b^2)} = -\frac{2b(\beta^2 + z^2)}{(\alpha^2 + \beta^2)(\alpha^2 + b^2)}.$$

Using $\beta^2 + z^2 = \frac{z^2(\alpha^2 + b^2)}{\alpha^2}$ and $\beta = bz/\alpha$ yields $(\nabla \times \mathbf{k})_z = -\frac{2\beta}{\alpha^2+\beta^2}\frac{z}{\alpha} = -\frac{2\beta}{\alpha^2+\beta^2}k_z$. Likewise, $\partial_y k_z = -\frac{yz}{\alpha(\alpha^2+\beta^2)}$ and $\partial_z(k_x + ik_y) = -\frac{(x+iy)\partial_z\alpha}{(\alpha+ib)^2} = -\frac{z(\alpha - ib)(x+iy)}{\alpha(\alpha^2+\beta^2)(\alpha+ib)}$, so

$$(\nabla \times \mathbf{k})_x = \partial_y k_z - \partial_z k_y = -\frac{2\beta(\alpha x + by)}{(\alpha^2+\beta^2)(\alpha^2+b^2)} = -\frac{2\beta}{\alpha^2+\beta^2}k_x,$$

and similarly $(\nabla \times \mathbf{k})_y = -\frac{2\beta}{\alpha^2+\beta^2}k_y$. Hence

$$\nabla \times \mathbf{k} = -\frac{2\beta}{\alpha^2 + \beta^2}\,\mathbf{k}.$$

For right-handed $(\mathbf{e}_1, \mathbf{e}_2, \mathbf{k})$, $B_{12} - B_{21} = -(\nabla \times \mathbf{k})\cdot\mathbf{k} = \frac{2\beta}{\alpha^2+\beta^2}$, so $\theta = \frac{\alpha}{\alpha^2+\beta^2}$, $\omega = \frac{\beta}{\alpha^2+\beta^2}$, and

$$\boxed{\rho = \theta + i\omega = \frac{1}{\bar{w}}}.$$

*(Convention note: $\rho = 1/w$ corresponds to redefining the complex scalar with the opposite twist sign, $\tilde{\rho} = \theta - i\omega$, rather than rotating $(\mathbf{e}_1, \mathbf{e}_2)$ within the oriented screen plane.)*

2. **Global Frobenius norm and vanishing shear (no division by $\alpha$):** Set $u = x + iy$, $P = \alpha + ib$, $R^2 = \alpha^2 + \beta^2$, $f = k_x + ik_y = u/P$, and $k_z = \beta/b$. From $\partial_j w = \frac{\bar{w}(x, y, z + ib)_j}{R^2}$, we have $\partial_x \alpha = \frac{x\alpha}{R^2}$, $\partial_y \alpha = \frac{y\alpha}{R^2}$, $\partial_z \alpha = \frac{\alpha z + b\beta}{R^2}$, $\partial_x \beta = -\frac{x\beta}{R^2}$, $\partial_y \beta = -\frac{y\beta}{R^2}$, and $\partial_z \beta = \frac{b\alpha - \beta z}{R^2}$. Using $\alpha^2 - \beta^2 = |u|^2 + z^2 - b^2$ and $\alpha\beta = bz$ gives $b^2|u|^2 = |P|^2(b^2 - \beta^2)$, so with $s = 1 - \frac{\beta^2}{b^2}$ we have $\frac{|u|^2}{|P|^2} = s$ and $\frac{\beta^2}{b^2} = 1 - s$:
   - *Horizontal block* $H = |\partial_x f|^2 + |\partial_y f|^2$: from $\partial_x f = \frac{1}{P} - \frac{u x \alpha}{R^2 P^2}$ and $\partial_y f = \frac{i}{P} - \frac{u y \alpha}{R^2 P^2}$, the cross-term sums to $-2\operatorname{Re}\!\left(\frac{|u|^2\alpha}{R^2|P|^2 P}\right) = -\frac{2|u|^2\alpha^2}{R^2|P|^4}$, giving

$$H = \frac{2}{|P|^2} - \frac{2\alpha^2 s}{R^2|P|^2} + \frac{\alpha^2 s^2}{R^4} = \frac{2(1 - s)}{R^2} + \frac{\alpha^2 s^2}{R^4},$$

since $R^2 - \alpha^2 s = \alpha^2(1 - s) + \beta^2 = \frac{\beta^2}{b^2}|P|^2$.
   - *Mixed blocks:* since $(\alpha z + b\beta)^2 = |P|^2(\beta^2 + z^2)$ and $b^2(\beta^2 + z^2) = |P|^2\beta^2$, both mixed blocks are equal:

$$|\partial_z f|^2 = \frac{|u|^2(\alpha z + b\beta)^2}{R^4|P|^4} = \frac{|u|^2\beta^2}{b^2 R^4} = (\partial_x k_z)^2 + (\partial_y k_z)^2,$$

so their sum is $M_{\mathrm{mix}} = \frac{2|P|^2 s(1 - s)}{R^4}$.
   - *Vertical square:* from $b^2 z^2 = \alpha^2\beta^2$, $(b\alpha - \beta z)^2 = b^2\alpha^2 s^2$, so $(\partial_z k_z)^2 = \frac{\alpha^2 s^2}{R^4}$.
   - *Total sum:* combining $H + M_{\mathrm{mix}} + (\partial_z k_z)^2$ yields

$$\sum_{i,j}(\partial_j k_i)^2 = \frac{2(1 - s)}{R^2} + \frac{2s}{R^4}\bigl[\alpha^2 s + (\alpha^2 + b^2)(1 - s)\bigr] = \frac{2(1 - s)}{R^2} + \frac{2s R^2}{R^4} = \frac{2}{R^2} = \frac{2}{\alpha^2 + \beta^2} = 2(\theta^2 + \omega^2).$$

Because $M_{ij} = \partial_j k_i$ satisfies $k^i M_{ij} = 0$ and $k^j M_{ij} = 0$, its Frobenius norm equals $\sum_{A,B} B_{AB}^2 = 2(\theta^2 + \omega^2 + |\sigma|^2)$, forcing $|\sigma|^2 = 0$ and hence $\sigma = 0$.

3. **Lemma — Smooth Extension Across $\widetilde{\Omega}_b$ by Density:** Because $b \ne 0$ implies $\alpha^2 + b^2 \ge b^2 > 0$ and $w \ne 0$ everywhere on $\widetilde{\Omega}_b$, both $\mathbf{k}$ and $dw = \frac{x\,dx + y\,dy + (z + ib)\,dz}{w}$ are smooth across the spanning-disk lift $\{z = 0,\; x^2 + y^2 < b^2\}$ where $\alpha = 0$. Since $\{\alpha \ne 0\}$ is open and dense in $\widetilde{\Omega}_b$, the smooth identities

$$\boxed{(\mathbf{k} \cdot \nabla)\mathbf{k} = \mathbf{0}, \qquad \sigma = 0, \qquad \rho = \frac{1}{\bar{w}}}$$

extend by continuity to the entire two-sheeted covering space $\widetilde{\Omega}_b$.

4. **Deck transformation law:** Under $\tau : w \mapsto -w$ (so $\alpha \mapsto -\alpha$, $\beta \mapsto -\beta$),

$$\boxed{\mathbf{k} \circ \tau = \left(\frac{-\alpha x + by}{\alpha^2 + b^2},\; \frac{-\alpha y - bx}{\alpha^2 + b^2},\; -\frac{\beta}{b}\right)}, \qquad \rho \circ \tau = -\rho.$$

In general, $\mathbf{k} \circ \tau \ne -\mathbf{k}$ (for example, at $(x,y,z) = (2,0,0)$ with $b = 1$ and $w = \sqrt{3}$, the two tangent vectors are $(\frac{\sqrt{3}}{2}, -\frac{1}{2}, 0)$ and $(-\frac{\sqrt{3}}{2}, -\frac{1}{2}, 0)$, not antipodal).

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

---

## Bridge Lemma 4 — Conditional Sheet Continuation and Adaptive Step Control

### Lemma 4A (Exact Robinson-Ray Phase Margin and Safe-Step Criterion)

Along an exact forward Robinson congruence ray ($w(s) = w_0 + s = \alpha(s) + i\beta(s)$, $\beta_n = \mathrm{Im}(w_n)$ constant) with step $h > 0$ and $w_{n+1} = w_n + h$, the exact phase margin $\mu_n = \mathrm{Re}(w_{n+1}\overline{w_n})$ satisfies

$$\boxed{\mu_n = |w_n|^2 + h\alpha_n = \alpha_n(\alpha_n + h) + \beta_n^2.}$$

Consequently, in exact arithmetic, $\mu_n > 0$ holds if and only if

$$\boxed{\begin{cases} h > 0, & \alpha_n \ge 0, \\[4pt] 0 < h < \dfrac{|w_n|^2}{-\alpha_n} = -\alpha_n + \dfrac{\beta_n^2}{-\alpha_n}, & \alpha_n < 0. \end{cases}}$$

When $\beta_n \ne 0$ and $\alpha_n < 0$, the exact ray crosses the lifted spanning disk ($\alpha = 0$) smoothly without meeting $\mathcal{B}_b$; failure of $\mu_n > 0$ for $h \ge |w_n|^2 / (-\alpha_n)$ is an **algorithmic phase-step failure**, not a geometric singularity. When $\beta_n = 0$ and $\alpha_n < 0$, $h = -\alpha_n$ hits $w = 0$ on the branch ring $\mathcal{B}_b$.

### Lemma 4B (Perturbation-Stable Endpoint Sign Selection)

Let $\mathbf{x}(s)$ be a continuous trajectory in $\mathbb{R}^3 \setminus \mathcal{B}_b$ with an **initial sheet assignment** at $s_0$ and exact analytically continued roots $w_n = w(s_n)$ and $w_{n+1} = w(s_{n+1})$ along the segment $[s_n, s_{n+1}]$. Suppose computed approximations $\widehat{w}_n, \widehat{w}_{n+1}$ (for the correctly matched roots) satisfy $|\widehat{w}_n - w_n| \le \varepsilon_n$ and $|\widehat{w}_{n+1} - w_{n+1}| \le \varepsilon_{n+1}$, and inner-product evaluation incurs rounding error at most $\delta_{\mathrm{ip}} \ge 0$. Writing $\widehat{w}_k = w_k + \eta_k$ ($|\eta_k| \le \varepsilon_k$) gives

$$\widehat{w}_{n+1}\overline{\widehat{w}_n} - w_{n+1}\overline{w_n} = \eta_{n+1}\overline{w_n} + w_{n+1}\overline{\eta_n} + \eta_{n+1}\overline{\eta_n},$$

so $|\widehat{\mu}_n - \mu_n| \le E_n = |w_n|\varepsilon_{n+1} + |w_{n+1}|\varepsilon_n + \varepsilon_n\varepsilon_{n+1}$. Whenever

$$\boxed{\mu_n > E_n + \delta_{\mathrm{ip}},}$$

the numerically evaluated inner product strictly satisfies $\widehat{\mu}_n > 0$, preserving the sign of the endpoint inner product.

**Path-continuation qualification:** For a general continuous trajectory, positive endpoint inner product alone does **not** prove that the selected endpoint root matches analytic continuation along $[s_n, s_{n+1}]$, because a curved path can wind around $\mathcal{B}_b$ and change sheets while returning to a positive endpoint inner product. Establishing sheet continuation on a general path requires a separately proved bound restricting phase variation (no branch-locus winding) across the entire intervening segment. In the exact straight Robinson-ray setting of Lemmas 4A and 4C, $w(s_n + t) = w_n + t$ ($t \in [0, h]$) moves along a straight line in $\mathbb{C}$ and cannot wind around $0$ once segment clearance $d_{\min} > 0$ holds.

### Lemma 4C (Computable A-Priori Criteria and Mandatory Segment Clearance for Robinson Rays)

Along a forward Robinson ray ($w(s_n + t) = w_n + t$ for $t \in [0, h]$), let $a = \widehat{w}_n$ satisfy $|a - w_n| \le \varepsilon_n$, and assume the computed endpoint candidate root lies within $\varepsilon_{\mathrm{cand}} = \varepsilon_{n+1}$ of one of the two exact roots $\pm w_{n+1}$ (where $\varepsilon_{\mathrm{cand}}$ must enclose all errors arising from the computed endpoint $\mathbf{x}_n + h\mathbf{k}$, evaluation of $\Delta$, and the complex square root).

1. **Mandatory segment-clearance condition:** The exact minimum distance to the singular root $w = 0$ along the segment $t \in [0, h]$ is
$$\boxed{d_{\min} = \min_{t \in [0, h]} |w_n + t| = \sqrt{\beta_n^2 + \mathrm{dist}(-\alpha_n, [0, h])^2}.}$$
Because $|(w_n + t) - (a + t)| = |w_n - a| \le \varepsilon_n$ uniformly in $t$, the computable lower bound
$$\boxed{d_{\min}^{\mathrm{lower}}(h) = \max\!\left(0,\; \sqrt{(\mathrm{Im}\,a)^2 + \mathrm{dist}(-\mathrm{Re}\,a, [0, h])^2} - \varepsilon_n\right) \le d_{\min}}$$
must satisfy $d_{\min}^{\mathrm{lower}}(h) > \tau_{\mathrm{clear}} \ge 0$ to ensure the entire ray segment avoids $\mathcal{B}_b$.

2. **Robinson-ray phase-margin criterion (4C-1):** With $P(h) = |a|^2 + h\,\mathrm{Re}(a)$ and $\overline{E}_n(h) = (|a| + \varepsilon_n)\varepsilon_{n+1} + (|a + h| + \varepsilon_n)\varepsilon_n + \varepsilon_n\varepsilon_{n+1}$,
$$\boxed{P(h) - \varepsilon_n(2|a| + \varepsilon_n + h) > \overline{E}_n(h) + \delta_{\mathrm{ip}} + \tau_{\mathrm{guard}}}$$
is a computable sufficient condition guaranteeing $\mu_n > E_n + \delta_{\mathrm{ip}}$ prior to sign selection.

3. **Exact-ray predictor criterion (4C-2):** Let $p = \widehat{w}_{\mathrm{pred}} = \mathrm{fl}(a + h)$ satisfy $|p - w_{n+1}| \le \varepsilon_{\mathrm{pred}}$ (with $\varepsilon_{\mathrm{pred}} = \varepsilon_n + \delta_{\mathrm{add}}$ supplied as a precondition bounding both the stored-root error on the correct sheet and predictor addition roundoff), let $r_+$ denote the correctly matched candidate ($|r_+ - w_{n+1}| \le \varepsilon_{\mathrm{cand}}$) and $r_- = -r_+$ its opposite, and set $q = \varepsilon_{\mathrm{pred}} + \varepsilon_{\mathrm{cand}}$. By the triangle inequality,
$$|r_+ - p| \le q, \qquad |r_- - p| \ge 2|w_{n+1}| - q,$$
so the exact distance separation satisfies
$$D = |r_- - p| - |r_+ - p| \ge 2|w_{n+1}| - 2q.$$
If computing the distance difference incurs absolute rounding error at most $\delta_{\mathrm{cmp}} \ge 0$, strict positivity of the computed difference is guaranteed whenever $2|w_{n+1}| - 2q > \delta_{\mathrm{cmp}}$, i.e., $|w_{n+1}| > q + \tfrac{1}{2}\delta_{\mathrm{cmp}}$. Combining this with $|w_{n+1}| \ge |p| - \varepsilon_{\mathrm{pred}}$ and adding a conservative guard $\tau_{\mathrm{guard}} \ge 0$ yields the computable sufficient condition
$$\boxed{|p| > 2\varepsilon_{\mathrm{pred}} + \varepsilon_{\mathrm{cand}} + \tfrac{1}{2}\delta_{\mathrm{cmp}} + \tau_{\mathrm{guard}}.}$$

**Conditional certification note:** Lemma 4C guarantees sheet preservation only when (i) the caller supplies valid conservative enclosures for $\varepsilon_n, \delta_{\mathrm{add}}, \varepsilon_{\mathrm{cand}}, \delta_{\mathrm{ip}}, \delta_{\mathrm{cmp}}$ accounting for position, ray-step, $\Delta$, complex-square-root, and comparison errors relative to the exact Robinson-ray endpoint, and (ii) the inequalities themselves are evaluated without overflow via outward-rounded interval arithmetic or another justified enclosure method. Ordinary `float64` evaluation is empirical rather than backend-certified. $\square$

---

## Bridge Lemma 5 — Stereographic Screen-Gauge Covariance and Invariant Transport

Let $\mathbf{k} \in S^2$ be a fixed Robinson-congruence direction on a specified covering sheet, and let $\mathbf{m}_{\mathrm{S}}, \mathbf{m}_{\mathrm{N}}$ be normalized complex screen frames satisfying

$$\mathbf{m}_{\mathrm{N}}(1/\zeta) = \frac{\bar{\zeta}}{\zeta}\,\mathbf{m}_{\mathrm{S}}(\zeta)$$

on the stereographic overlap ($\zeta = \frac{k_x + i k_y}{1 + k_z}$, $\eta = \frac{k_x - i k_y}{1 - k_z} = \frac{1}{\zeta}$ for $\mathbf{k} \notin \{(0,0,\pm 1)\}$).

1. **Unitary inverse transition:** Setting $\Phi_{\mathrm{S}\to\mathrm{N}}(\zeta) = \frac{\bar{\zeta}}{\zeta}$ gives $|\Phi_{\mathrm{S}\to\mathrm{N}}(\zeta)| = 1$ and inverse transition $\Phi_{\mathrm{N}\to\mathrm{S}}(1/\zeta) = \Phi_{\mathrm{S}\to\mathrm{N}}(\zeta)^{-1} = \overline{\Phi_{\mathrm{S}\to\mathrm{N}}(\zeta)}$ on the overlap.
2. **Spin-weight covariance:** A screen quantity built from $p$ factors of $\mathbf{m}$ and $q$ factors of $\bar{\mathbf{m}}$ has spin weight $s = p - q$ under $Q_{\mathrm{N}} = \Phi^s Q_{\mathrm{S}}$ (e.g., $Z = \mathbf{u}\cdot\mathbf{m} \mapsto \Phi Z$, $\bar{Z} \mapsto \bar{\Phi}\bar{Z}$, $Z^2 \mapsto \Phi^2 Z^2$, and $Z\bar{Z}$ invariant).
3. **Hermitian invariance for equal spin weight:** For screen frames or quantities sharing spin weight $s$ ($\mathbf{m}_{\mathrm{N}} = \Phi^s \mathbf{m}_{\mathrm{S}}$, $\mathbf{n}_{\mathrm{N}} = \Phi^s \mathbf{n}_{\mathrm{S}}$, $|\Phi| = 1$), Hermitian inner products and norms satisfy $\overline{\mathbf{m}_{\mathrm{N}}}\cdot\mathbf{n}_{\mathrm{N}} = \bar{\Phi}^s \Phi^s (\overline{\mathbf{m}_{\mathrm{S}}}\cdot\mathbf{n}_{\mathrm{S}}) = \overline{\mathbf{m}_{\mathrm{S}}}\cdot\mathbf{n}_{\mathrm{S}}$.
4. **Metric, area-transport, and optical-scalar invariance on a fixed sheet:** For a fixed covering sheet, the transverse metric, infinitesimal Robinson-ray area transport $\frac{A(s)}{A(s_0)} = \frac{|w(s)|^2}{|w(s_0)|^2}$, and intrinsic optical scalars ($\rho = 1/\bar{w}$, expansion $\theta = \mathrm{Re}\,\rho$, twist $\omega = \mathrm{Im}\,\rho$) are invariant across stereographic chart handoffs (and vary with the covering-sheet sign $w \mapsto -w$ when changing sheets).
5. **Shear-free invariance vs. shear-coefficient covariance:** The shear-free condition ($\sigma = 0$) is gauge-invariant across charts, whereas a general nonzero complex shear coefficient transforms covariantly with its spin weight under screen-frame rotation.
6. **Algebraic identity vs. numerical handoff:** These statements are exact algebraic and geometric identities on the stated domain; numerical chart-handoff residuals $R_{\mathrm{handoff}} = \|\mathbf{m}_{\mathrm{N}}(1/\zeta) - (\bar{\zeta}/\zeta)\mathbf{m}_{\mathrm{S}}(\zeta)\|_2$ and chart-selection conditioning (selecting $\zeta$ for $k_z \ge 0$ and $\eta$ for $k_z < 0$ yields $|\text{selected coordinate}| \le 1$, while the unused chart degenerates near a pole) require separately justified floating-point error bounds. $\square$
