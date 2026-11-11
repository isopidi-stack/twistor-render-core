# Twistor Render Core

An open-source, experimental 3D rendering pipeline built on **Penrose's Twistor Theory** and **complex 2-spinors** using PyTorch.

Rather than relying on standard $4 \times 4$ real Euclidean transformation matrices and Bounding Volume Hierarchy (BVH) tree traversals, **Twistor Render Core** maps 3D spatial points into $2 \times 2$ complex Hermitian matrices $X^{AA'}$ and evaluates ray-point intersections via Penrose's incidence relation in complex projective space ($\mathbb{C}P^3$).

---

## 1. Mathematical Intuition

### Spacetime to Complex Hermitian Mapping
Points in Minkowski spacetime $x^a = (t, x, y, z)$ map to $2 \times 2$ complex Hermitian matrices $X^{AA'}$ via Pauli spin matrices:

$$X^{AA'} = \begin{pmatrix} t + z & x - iy \\ x + iy & t - z \end{pmatrix}$$

For 3D spatial coordinates $(x, y, z)$, light-cone radial distance $t = \sqrt{x^2 + y^2 + z^2}$ is evaluated so that spatial points satisfy null geodesic constraints along primary light paths originating from the camera center $(0,0,0)$.

### Penrose Incidence Relation
A light ray in twistor space is parameterized by a pair of 2-component complex spinors $Z^\alpha = (\omega^A, \pi_{A'})$. For primary rays passing through origin $X_0 = 0$, position spinor $\omega^A = 0$. A point $X^{AA'}$ lies on the null ray defined by momentum spinor $\pi_{A'}$ if and only if it satisfies Penrose's incidence relation:

$$\omega^A = i X^{AA'} \pi_{A'}$$

### Residual Collision Kernel & Volumetric Thresholding
In continuous space, a 1D null ray has zero measure for exact collision with a 0D point. Collision detection evaluates whether the complex norm residual falls within a geometric tolerance $\epsilon$:

$$\text{Residual} = \|\omega^A - i X^{AA'} \pi_{A'}\| < \epsilon$$

This residual tolerance acts as an implicit volumetric bounding radius around point cloud elements.

---

## 2. Engine Architecture

- `twistor/geometry.py`: Spatial-to-Hermitian conversions, complex spinor operations, and SL(2, C) Lorentz transformations.
- `twistor/incidence.py`: Vectorized Penrose incidence residual kernels and hit evaluation.
- `twistor/renderer.py`: Camera ray mapping into $\mathbb{C}P^3$ twistor space and 2D image buffer rasterization.
- `benchmarks/run_suite.py`: Automated performance and throughput benchmark suite.

---

## 3. Empirical CPU Benchmarks

Throughput measures single-ray-to-$N$-point batch incidence evaluations ($O(N)$ brute-force tensor contractions) across scaled point cloud sizes on a standard CPU backend using PyTorch SIMD vectorization:

| Point Count ($N$) | Execution Time (ms) | Throughput (Ray-Point Ops/sec) |
| :--- | :--- | :--- |
| **1,000** | 0.194 ms | 5,152,196 |
| **10,000** | 1.561 ms | 6,406,043 |
| **100,000** | 7.236 ms | 13,819,175 |
| **1,000,000** | 82.739 ms | 12,086,184 |

*Note: Current benchmarks measure unindexed $O(N)$ tensor operations. Future iterations will introduce twistor-space spatial tree partitioning.*

---

## 4. Quickstart

```bash
# Clone repository
git clone [https://github.com/isopidi-stack/twistor-render-core.git](https://github.com/isopidi-stack/twistor-render-core.git)
cd twistor-render-core

# Setup virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install torch matplotlib

# Run benchmark suite
python benchmarks/run_suite.py
---

## 5. Visual Render Output

## Performance & Architecture Benchmarks

Benchmarked on **Intel Arc B580 (12 GB)** via PyTorch `xpu` backend against a standard PyTorch Euclidean ray-point intersector ($32 \times 32$ resolution):

| Points ($N$) | Twistor Incidence (ms) | Euclidean Vector (ms) | Overhead Factor |
| :--- | :--- | :--- | :--- |
| **1,000** | 60.32 ms | 0.25 ms | **242.5x** |
| **10,000** | 602.20 ms | 1.69 ms | **356.0x** |
| **100,000** | 6,028.14 ms | 18.33 ms | **328.8x** |

### Architectural Bottleneck Analysis
The throughput gap on GPU (~300x) compared to CPU (~10x–18x) is driven by PyTorch framework allocation patterns rather than Penrose geometry:
1. **Unfused Complex GPU Kernels:** PyTorch handles `torch.complex64` on GPU by splitting operations into separate real/imaginary sub-kernels. Each spinor contraction triggers intermediate VRAM reads/writes instead of executing inside GPU register memory.
2. **Data Payload:** A $2 \times 2$ Hermitian matrix $X^{AA'}$ requires 8 real floats versus 3 real floats for Euclidean 3D vectors ($2.66\times$ VRAM bandwidth).
3. **SYCL Chunking:** Point clouds above $N = 2,000$ are chunked to satisfy Intel SYCL 32-bit work-item index limits, introducing sequential kernel launch overhead.

**v0.2 Optimization Target:** Decompose 2-spinor complex operations into real FP32 tensor contractions ($\mathbb{R}^4$) to leverage native SIMD float execution lanes on Intel Arc hardware.