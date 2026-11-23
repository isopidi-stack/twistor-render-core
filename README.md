# Argon (`twistor-render-core`)

Argon (`twistor-render-core`) is a computational geometry and rendering engine exploring spinor/twistor parameterizations, Robinson congruences, and Hopf–Villarceau circle correspondences.

For full theorem statements, proof status, and scope boundaries, see **[`docs/THEOREMS_AND_PROOFS.md`](docs/THEOREMS_AND_PROOFS.md)**.

## Mathematical Status at a Glance

The repository maintains a strict separation between **established algebraic derivations**, **explicit proof obligations**, and **numerical or architectural verification**:

- **Theorem 1 — Covering & Screen-Frame Transition:**
  - *Established:* The two-sheeted covering $\widetilde{\Omega}_b$ over $\mathbb{R}^3 \setminus \mathcal{B}_b$, disk-cut domain $\Omega_b^{\mathrm{cut}}$ with continuous branch selection ($\operatorname{Re} w > 0$), chart transition $\zeta\eta = 1$ on overlaps, and screen-frame phase relation $\mathbf{m}_{\mathrm{N}}(1/\zeta) = (\bar{\zeta}/\zeta)\mathbf{m}_{\mathrm{S}}(\zeta)$.
  - *Proof obligation:* Full derivation and sign/sheet certification of the optical scalars $\sigma = 0$ and $\rho = 1/w$ from an explicit congruence tangent field and optical projection/derivative operator.
- **Theorem 2 — Restricted Robinson–Villarceau Correspondence ($\mathcal{D}_{\mathrm{Rob}} \leftrightarrow \mathcal{V}_{\mathrm{Rob}}$):**
  - *Established:* Forward map from rays in $\mathcal{D}_{\mathrm{Rob}}$ to $(\zeta, b) \in \mathbb{C}^* \times \mathbb{R}^*$, finiteness of the stereographic image, origin-passing plane $Z = aX - cY$ with normal $\mathbf{N} = (-a, c, 1)$, radial extrema relation $r_{\max} r_{\min} = b^2$, exact 3D circle center $\mathbf{C}_{\zeta,b} = (bc, ba, 0)$ and radius $R_{\zeta,b} = |b|\sqrt{1 + |\zeta|^2}$, distinct unparameterized $+b / -b$ point sets for $\zeta \ne 0, b \ne 0$, signed-$b$ recovery from the circle center and plane normal scaled to third component $1$, and explicit reconstruction of $(\mathbf{k}, \mathbf{x}_0)$ back into $\mathcal{D}_{\mathrm{Rob}}$.
  - *Proof obligation:* Independent geometric characterization and surjectivity if the target circle family is defined outside the forward image of $\mathcal{D}_{\mathrm{Rob}}$.

## Scope & Evidence Boundaries

- **Numerical & integrity verification:** Regression suites and SHA-256 baseline checks verify consistency on tested configurations and pinned files; they do not substitute for formal proofs or establish historical novelty.
- **Architectural claims:** Using a non-raster, gimbal-free geometric formulation is an architectural choice rather than proof of infinite precision or universal superiority over conventional renderers.
- **Extended hypotheses:** Any higher-dimensional/five-component twistor extensions or quantum-number interpretations sit outside Theorems 1 and 2 unless explicitly formalized and verified.

## Repository Layout

- `docs/THEOREMS_AND_PROOFS.md` — Formal theorem statements, proof status, and scope boundaries.
- `argon_geometry_audit.py`, `argon_v4_adversarial_audit.py`, `argon_v44_adversarial_audit.py` — Audit and regression scripts (keep frozen/untouched when preserving established hashes).