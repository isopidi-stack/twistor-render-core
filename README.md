# twistor-render-core

Experimental differentiable 3D rendering engine built on **Roger Penrose's Twistor Theory** ($\mathbb{C}P^3$) and complex 2-spinors using **PyTorch & Intel XPU**.

Instead of traditional Euclidean $4 \times 4$ matrices and discrete rasterization pipelines, this engine evaluates null-geodesic geometry, conformal $SU(2,2)$ relativistic transformations, and complex projective incidence relations ($\omega^A = i X^{AA'} \pi_{A'}$) via native `torch.complex64` dual-spinor contractions.

---

## Core Architecture (v0.3)

### 1. Native `complex64` Dual-Spinor Incidence Kernel
* **Direct Complex Contractions:** Converts 3D spatial points into native $2 \times 2$ complex Hermitian matrices ($X^{AA'}$) via Pauli decomposition and evaluates $\omega^A = i X^{AA'} \pi_{A'}$ in `torch.complex64`, cutting intermediate tensor memory payloads in half compared to real FP32 block expansions.
* **SYCL-Index-Safe 3D GEMM:** Replaces 4D broadcasting tensors with direct 3D matrix multiplications (`[B, 2, 2] @ [2, R]`), avoiding SYCL 32-bit index overflow errors on Intel Level Zero backends.
* **End-to-End Complex AutoDiff:** Preserves full PyTorch autograd graph flow across complex Hermitian transformations, enabling gradient-based camera pose optimization directly from pixel-wise residuals.

### 2. Hermitian Bounding-Sphere Cluster Culling (`twistor/bvh.py`)
* **Spatial Acceleration:** Mitigates the $O(M \cdot N)$ brute-force ray-point bottleneck by sorting points along depth on CPU, grouping them into local Hermitian bounding spheres, and culling non-visible clusters via vectorized XPU masking before dense matrix contractions.
* **Memory-Safe Super-Batching:** Evaluates surviving clusters in cache-friendly super-batches (default `2048`) to prevent VRAM thrashing and Level Zero driver resource exhaustion on Intel Arc hardware.

---

## Performance Telemetry (Intel XPU)

### Spatial Cluster Culling vs. Brute-Force Scaling
Measured at $64 \times 64$ resolution on Intel XPU hardware:

| Points ($N$) | Brute-Force (ms) | Clustered Culling (ms) | Speedup |
| :--- | :--- | :--- | :--- |
| **10,000** | 37.98 ms | Warmup path | Baseline |
| **50,000** | 1,236.89 ms | 50.24 ms | **24.62x** |
| **100,000** | 7,606.16 ms | 111.29 ms | **68.34x** |

---

## Quickstart & Demos

1. **Setup Environment:**
   ```powershell
   # Activate your XPU virtual environment and install dependencies
   pip install torch torchvision