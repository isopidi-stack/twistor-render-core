# twistor-render-core

Experimental differentiable 3D rendering engine built on Roger Penrose's Twistor Theory ($\mathbb{CP}^3$) and complex 2-spinors using PyTorch & Intel XPU.

Instead of traditional Euclidean $4\times 4$ matrices and discrete rasterization pipelines, this engine evaluates null-geodesic geometry, $SU(2)$ spinor conjugation ($X_{\text{cam}} = R_A X R_A^\dagger$), $SL(2,\mathbb{C})$ Lorentz boosts, and complex projective eigenspinor incidence relations ($\omega^A = i X^{AA'} \pi_{A'}$) via native `torch.complex64` 3D GEMM contractions.

![Topological Shape Morphing via Complex Spinor Flow](assets/topological_morphing.gif)
*Real-time topological shape morphing driven by continuous complex eigenspinor incidence flow.*

![SL(2,C) Relativistic Lorentz Aberration](assets/relativistic_aberration.gif)
*Relativistic Lorentz aberration ($SL(2,\mathbb{C})$ spinor boost sweeping from $-0.85c$ to $+0.85c$) with zero-Z-buffer eigenspinor depth relief shading.*

![End-to-end differentiable 6-DoF camera pose recovery](assets/pose_recovery_showcase.png)
*End-to-end differentiable 6-DoF camera pose recovery (3D position + $SU(2)$ yaw/pitch rotation) optimizing directly through complex spinor incidence residuals on Intel XPU: [Initial Guess | Epoch 45 | Final Recovered | Ground Truth].*

---

## Why Twistors? (4 Core Advantages)

1. **Singularity-Free $SU(2)$ Spinor Manifold (Zero Gimbal Lock):** Camera rotations are evaluated via unitary spinor conjugation ($X_{\text{cam}} = R_A X R_A^\dagger$), providing smooth, out-of-the-box PyTorch gradients for 6-DoF pose recovery without Euler singularities.
2. **Twistor Duality & Zero-Z-Buffer Signed Depth:** Null rays map directly to single points in projective twistor space ($\mathbb{CP}^3$). A single Hermitian spinor contraction $\langle \pi \mid \text{diff} \rangle$ simultaneously yields transverse ray distance and exact signed longitudinal depth (`-Im(<pi | diff>)`) without sorting or Z-buffers.
3. **Static Tensor Shapes & Native `complex64` GEMM Speedup:** All ray-point evaluations run on fixed-shape $[P, 2, 2] \times [2, R]$ complex contractions with zero dynamic boolean masking, eliminating host-device synchronization stalls across Intel XPU, NVIDIA CUDA, and Apple MPS.
4. **Native Relativistic & Non-Euclidean Optics ($SL(2,\mathbb{C})$):** Celestial ray spinors transform directly under $SL(2,\mathbb{C})$ Lorentz boosts (`boost_v`) and conformal perturbations, enabling differentiable relativistic aberration and inverse velocity recovery without rewriting the pipeline.

---

## Repository Structure

    twistor-render-core/
    РІвЂќСљРІвЂќР‚РІвЂќР‚ twistor/          # Core complex64 spinor kernels, SU(2)/SL(2,C) transforms & Hermitian BVH
    РІвЂќСљРІвЂќР‚РІвЂќР‚ examples/         # Interactive sandboxes, relativistic aberration, lensing, morphing & 6-DoF demos
    РІвЂќСљРІвЂќР‚РІвЂќР‚ benchmarks/       # XPU / CUDA vs. brute-force scaling telemetry
    РІвЂќвЂќРІвЂќР‚РІвЂќР‚ assets/           # Showcase renders and visual documentation

---

## Core Architecture (v0.7)

### 1. Native `complex64` Eigenspinor Incidence & $SL(2,\mathbb{C})$ Kernel (`twistor/renderer.py`)
- **Projective Eigenspinor Testing & Depth Relief:** Evaluates exact perpendicular incidence residuals and signed longitudinal depth (`-Im(<pi | diff>)`) in native `torch.complex64`, cutting intermediate tensor memory payloads in half compared to real FP32 block expansions.
- **Differentiable 6-DoF $SU(2)$ + $SL(2,\mathbb{C})$ Camera Transforms:** Combines unitary $SU(2)$ spinor conjugation with $SL(2,\mathbb{C})$ Lorentz boost matrices (`B_boost`), preserving full PyTorch autograd flow across spatial pose (`x, y, z, yaw, pitch`) and relativistic velocity (`boost_v = v/c`).
- **SYCL-Index-Safe Static 3D GEMM:** Replaces 4D broadcasting tensors with fixed-shape 3D matrix multiplications (`[P, 2, 2] @ [2, R]`) and static `torch.where` bounds, preventing SYCL 32-bit index overflow errors and sync stalls on Intel Level Zero backends.

### 2. Hermitian Bounding-Sphere Cluster Culling (`twistor/bvh.py`)
- **Spatial Acceleration:** Mitigates the $O(M \cdot N)$ brute-force ray-point bottleneck by sorting points along depth, grouping them into local Hermitian bounding spheres, and culling non-visible clusters before dense matrix contractions.
- **Memory-Safe Super-Batching:** Evaluates surviving clusters in cache-friendly super-batches (`2048`) to prevent VRAM thrashing and Level Zero driver resource exhaustion on Intel Arc hardware.

---

## Performance Telemetry (Intel XPU)

Measured at 64x64 resolution on Intel XPU hardware:

| Points (N) | Brute-Force (ms) | Clustered Culling (ms) | Speedup |
| :--- | :--- | :--- | :--- |
| **10,000** | 37.98 ms | Warmup path | Baseline |
| **50,000** | 1,236.89 ms | 50.24 ms | **24.62x** |
| **100,000** | 7,606.16 ms | 111.29 ms | **68.34x** |

---

## Quickstart & Interactive Sandboxes

### 1. Setup Environment
Install PyTorch along with matplotlib and Pillow for real-time interactive rendering:

    pip install torch torchvision matplotlib pillow

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/isopidi-stack/twistor-render-core/blob/main/examples/twistor_colab_demo.ipynb)

### 2. Relativistic Lorentz Aberration & Inverse Velocity Recovery (`examples/relativistic_aberration.py`)
Applies an exact $SL(2,\mathbb{C})$ spinor Lorentz boost (`boost_v = v/c`) to celestial ray spinors, demonstrating Penrose-Terrell relativistic headlight compression alongside gradient-based velocity recovery from 2D pixels.

    python examples/relativistic_aberration.py

- **What it runs:** Recovers an unknown relativistic speed ($\beta = 0.78c$) purely via `loss.backward()` pixel gradients, then renders a continuous sweep from $-0.85c$ to $+0.85c$ into `assets/relativistic_aberration.gif`.
- **Why it matters:** Standard $4\times 4$ Euclidean camera pipelines break down when space and time mix under relativistic boostsРІР‚вЂќin twistor space, Lorentz transformations are native linear operators on 2-spinors.

### 3. Real-Time Gravitational Lensing (`examples/interactive_lensing.py`)
Simulates Schwarzschild-like spacetime curvature and null geodesic light deflection (producing Einstein rings and optical caustics) driven by complex spinor perturbations rather than screen-space post-processing shaders.

    python examples/interactive_lensing.py

- **Controls:** Move your mouse cursor across the window to drag the gravitational mass center in real time.
- **Why it matters:** Because the entire ray-bundle projection is differentiable via PyTorch autograd, this architecture scales to inverse astrophysics problemsРІР‚вЂќreconstructing hidden lensing mass distributions directly from observed distortions.

### 4. Topological Shape Morphing (`examples/topological_morphing.py`)
Demonstrates continuous, fluid 3D shape transitions (morphing seamlessly between a toroidal manifold and a harmonic sphere) governed by complex eigenspinor incidence blending rather than rigid triangle-mesh re-rigging.

    python examples/topological_morphing.py

- **Controls:** Click and drag inside the 3D viewport to orbit around the morphing manifold while it deforms continuously.
- **Why it matters:** Evaluating surfaces through spinor incidence parameters creates smooth, continuous gradient paths across arbitrary shape spaces without mesh tearing or discrete vertex correspondence constraints.
---

## Robot Vision & Full $SU(2,2)$ Conformal Optics (v0.8)

Because Sir Roger Penrose's twistor geometry maps light rays to points in projective twistor space ($\mathbb{CP}^3$), 3D camera localization and lens self-calibration collapse into unified spinor linear algebra:

* **Closed-Form "Kidnapped Robot" Localization (examples/instant_robot_snap.py):** Matching 3+ landmark rays solves the robot's exact (x, y, z) position in **one linear matrix step** (~0.95 ms, 5.15e-07 error) with zero initial guess and zero iterative optimization.
* **Full $SU(2,2)$ Conformal Self-Calibration (examples/robot_su22_calibration.py):** Jointly recovers 3D orientation ($SU(2)$), optical zoom dilation ($), and wide-angle/fisheye distortion ((b)$) simultaneously from a single image (2e-6 pixel MSE).

<p align="center">
  <img src="assets/kidnapped_robot_snap.png" alt="Kidnapped Robot One-Shot Twistor Localization" width="48%" />
  <img src="assets/su22_robot_calibration.gif" alt="Full SU(2,2) Conformal Robot Self-Calibration" width="48%" />
</p>


### Outlier-Robust Localization: Batched Spinor RANSAC (40% Outliers + Sensor Noise)

In real-world visual odometry, false feature matches break standard least-squares solvers. `twistor/ransac.py` evaluates **512 minimal 3-landmark spinor hypotheses in parallel** on GPU/XPU before refining on clean inliers (`python -m examples.benchmark_ransac`):

| Landmarks ($N$) | Raw Least-Squares Error (40% Outliers) | Batched Spinor RANSAC Error | Inliers Found | Execution Time (XPU) |
| :--- | :---: | :---: | :---: | :---: |
| **500** | `2.997 m` | **`0.00082 m` (0.82 mm)** | `300 / 500` (100%) | `1.877 ms` |
| **2,000** | `2.998 m` | **`0.00095 m` (0.95 mm)** | `1,200 / 2,000` (100%) | `2.136 ms` |
| **10,000** | `2.986 m` | **`0.00046 m` (0.46 mm)** | `5,998 / 10,000` (99.97%) | `3.738 ms` |



### Real-World Drone Flight Validation: 600-Frame 3D MAV Trajectory (`examples/euroc_drone_benchmark.py`)

Evaluated end-to-end across a 30-second (600-frame) 3D Micro Aerial Vehicle (MAV) trajectory with full 3D roll/pitch/yaw IMU quaternion-to-$SU(2)$ spinor mapping, continuous pixel sensor noise, and **35% corrupted false feature matches**:

| Telemetry Regime (35% Outliers + Pixel Noise) | ATE RMSE | Max Error | Inlier Recall | Mean Latency (XPU) |
| :--- | :---: | :---: | :---: | :---: |
| **Tactical / Optical-Grade Attitude** | **`0.88 mm`** | `2.24 mm` | **`100.0%` (`65.0% / 65.0%`)** | **`2.95 ms` / frame (`340 Hz`)** |
| **MEMS IMU + `0.15В°` Attitude Jitter** | **`17.70 mm`** | `52.34 mm` | **`100.0%` (`65.0% / 65.0%`)** | **`2.62 ms` / frame (`382 Hz`)** |

![EuRoC Drone Trajectory](assets/euroc_drone_trajectory.png)

## Contributing & Open Research Directions

PRs, GPU/XPU/CUDA/MPS benchmarks, and experimental spinor shaders are warmly welcomed! Good starting areas if you fork the repo:
* **Custom Fused Incidence Kernels (Triton / CUDA / SYCL):** Fusing the [P, 2, 2] @ [2, R] complex contraction and LogSumExp depth reduction into a single custom kernel.
* **End-to-End Robot SLAM / Visual Odometry Pipeline:** Chaining solve_camera_position_closed_form with dense $SU(2,2)$ photometric tracking on real camera streams.
* **Oriented Surface Normals & Splat Primitives:** Extending Hermitian point-cloud incidence residuals to oriented surface elements.


## ??? Robot Vision in Action: Direct $\mathbb{CP}^3$ Fusion
*Real-time spatial tracking and outlier rejection under heavy sensor noise.*

<p align="center">
  <img src="assets/fpv_hud_frame_1.png" width="700" alt="Twistor-Vision FPV HUD Preview">
</p>

### ?? Comprehensive Scaling & Stress Benchmarks
Tested across sparse point-cloud scaling and heavy outlier poisoning:

| Test Regime | Scale / Contamination | Retained Inliers | Execution Latency | Throughput | Position Error |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Point-Cloud Scale Stress** *(40% Outliers)* | 1,000 Points | 607 / 1,000 (60.7%) | 4.916 ms | 203 Hz | 1.215 mm |
| | 5,000 Points | 2,946 / 5,000 (58.9%) | 4.985 ms | 201 Hz | 0.257 mm |
| | 10,000 Points | 6,040 / 10,000 (60.4%) | 9.790 ms | 102 Hz | 0.530 mm |
| | **20,000 Points** | **11,920 / 20,000 (59.6%)** | **23.679 ms** | **42 Hz** | **0.272 mm** |
| **Outlier Poisoning Stress** *(10,000 Points)* | 20% Contamination | 7,996 / 10,000 (80.0%) | 9.167 ms | 109 Hz | 0.116 mm |
| | 35% Contamination | 6,537 / 10,000 (65.4%) | 8.865 ms | 113 Hz | 0.290 mm |
| | 50% Contamination | 5,060 / 10,000 (50.6%) | 7.645 ms | 131 Hz | 0.337 mm |
| | **60% Contamination** | **4,097 / 10,000 (41.0%)** | **7.721 ms** | **130 Hz** | **0.695 mm** |

---

## ### 🛡️ AI Vector Shield & RAG Poisoning Defense

Differentiable $\mathbb{CP}^3$ manifold verification for high-dimensional LLM embeddings and vector databases.

<p align="center">
  <img src="assets/cp3_manifold_shield.png" alt="CP^3 Twistor Manifold Vector Shield" width="750"/>
</p>

#### 📊 Adversarial Poisoning & Tier-X Stealth Stress Benchmarks
Evaluated across 512-dim complex projective embedding spaces against both severe Out-of-Distribution (OOD) noise (60% contamination) and **Tier-X Manifold-Tangent Stealth Injections** (40% structural mimicry matching L2 norm):

| Vector Scale ($) | Attack Class | Latency (ms) | Throughput (Single-Thread) | Precision | Recall | F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **1,000** | OOD Outliers (60% Contam.) | 18.096 ms | 88,417 Hz | **1.000** | **1.000** | **1.000** |
| **1,000** | Tier-X Tangent Stealth (40%) | 7.187 ms | 194,788 Hz | **1.000** | **1.000** | **1.000** |
| **5,000** | OOD Outliers (60% Contam.) | 49.894 ms | 160,341 Hz | **1.000** | **1.000** | **1.000** |
| **5,000** | Tier-X Tangent Stealth (40%) | 41.074 ms | 170,423 Hz | **1.000** | **1.000** | **1.000** |
| **10,000** | OOD Outliers (60% Contam.) | 104.884 ms | 152,549 Hz | **1.000** | **1.000** | **1.000** |
| **10,000** | Tier-X Tangent Stealth (40%) | 80.555 ms | 173,794 Hz | **1.000** | **1.000** | **1.000** |
| **20,000** | OOD Outliers (60% Contam.) | 198.303 ms | 161,370 Hz | **1.000** | **1.000** | **1.000** |
| **20,000** | Tier-X Tangent Stealth (40%) | 178.677 ms | 156,708 Hz | **1.000** | **1.000** | **1.000** |