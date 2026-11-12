# Twistor Render Core

An open-source, experimental 3D rendering pipeline built on **Penrose's Twistor Theory** ($\mathbb{C}P^3$ geometry) and complex 2-spinors using PyTorch and Intel XPU[cite: 3].

Rather than relying on standard $4 \times 4$ real Euclidean transformation matrices and Bounding Volume Hierarchies (BVH), this engine maps rays to points in twistor space and spatial points to complex lines via the **Penrose incidence relation**:
$$\omega^A = i X^{AA'} \pi_{A'}$$

---

## Quickstart

```powershell
# Clone repository
git clone [https://github.com/isopidi-stack/twistor-render-core.git](https://github.com/isopidi-stack/twistor-render-core.git)
cd twistor-render-core

# Setup virtual environment
python -m venv venv
.venv\Scripts\Activate.ps1

# Install dependencies
pip install torch matplotlib

# Run benchmark suite
python benchmarks/compare_baselines.py

# Run SU(2,2) camera and ASCII demo
python demos/camera_demo.py

