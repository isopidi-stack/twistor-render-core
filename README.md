# twistor-render-core

Experimental 3D rendering engine built on **Roger Penrose's Twistor Theory** ($\mathbb{C}P^3$) and complex 2-spinors using **PyTorch & Intel XPU**.

Instead of traditional Euclidean $4 \times 4$ matrices and rasterization pipelines, this engine explores null-geodesic geometry, conformal $SU(2,2)$ relativistic transformations, and complex projective incidence relations ($\omega^A = i X^{AA'} \pi_{A'}$) computed entirely via real FP32 tensor contractions.

---

## 🚀 Key Feature: Native AutoDiff Inverse Rendering
Unlike traditional Euclidean ray-tracers or rasterizers—where ray-geometry intersections are discrete and non-differentiable—our **Penrose Incidence Kernel** is fully implemented in PyTorch tensors.

This enables **native end-to-end automatic differentiation**:
* **Camera Pose Optimization:** Optimize unknown camera positions and yaw orientations purely through gradient descent by minimizing pixel-wise Mean Squared Error against a target frame.
* **Non-Euclidean Differentiability:** Backpropagate straight through relativistic conformal mappings and complex 2-spinor incidence relations.

Run the inverse optimization demo:
```powershell
python demos/inverse_render_demo.py