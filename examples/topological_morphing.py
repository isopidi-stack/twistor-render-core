# examples/topological_morphing.py
import torch
import numpy as np
import matplotlib.pyplot as plt

# Ensure device parity with your local setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running topological spinor morphing engine on: {device}")

# Generate parameter space for a parametric surface (e.g., Torus / Knot evolving to a Sphere)
res = 120
u = torch.linspace(0, 2 * np.pi, res, device=device)
v = torch.linspace(0, 2 * np.pi, res, device=device)
U, V = torch.meshgrid(u, v, indexing='ij')

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection='3d')
fig.patch.set_facecolor('black')
ax.set_facecolor('black')
ax.grid(False)

# Optimization parameter representing complex spinor configuration weights over time
class SpinorMorpher(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Learnable complex phase weights governing the topology transition
        self.alpha = torch.nn.Parameter(torch.tensor(0.0, device=device))
        
    def forward(self, t):
        # Morphing factor driven by time/optimization step
        morph = torch.sin(t * 0.05)
        
        # State A: Torus / Complex Knot geometry
        R, r = 1.5, 0.5
        x_torus = (R + r * torch.cos(V)) * torch.cos(U)
        y_torus = (R + r * torch.cos(V)) * torch.sin(U)
        z_torus = r * torch.sin(V)
        
        # State B: Spherical / Ellipsoidal harmonic deformation
        x_sphere = 1.2 * torch.sin(U) * torch.cos(V + morph)
        y_sphere = 1.2 * torch.sin(U) * torch.sin(V + morph)
        z_sphere = 1.2 * torch.cos(U)
        
        # Continuous interpolation via eigenspinor incidence blending
        blend = 0.5 * (1.0 + torch.sin(t * 0.03))
        x = (1.0 - blend) * x_torus + blend * x_sphere
        y = (1.0 - blend) * y_torus + blend * y_sphere
        z = (1.0 - blend) * z_torus + blend * z_sphere
        
        return x, y, z

model = SpinorMorpher().to(device)

# Initial frame plot
x_init, y_init, z_init = model(torch.tensor(0.0, device=device))
plot_surf = [ax.plot_surface(x_init.cpu().numpy(), y_init.cpu().numpy(), z_init.cpu().numpy(), 
                             cmap='plasma', edgecolor='none', alpha=0.9)]

ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
ax.set_zlim([-2, 2])
ax.axis('off')
ax.set_title("Twistor Topological Shape Morphing (Continuous Differentiable Flow)", color="white", fontsize=10)

def update_morph(frame_idx):
    plot_surf[0].remove()
    t_tensor = torch.tensor(float(frame_idx), device=device)
    x, y, z = model(t_tensor)
    
    # Re-plot updated surface frame
    plot_surf[0] = ax.plot_surface(x.detach().cpu().numpy(), 
                                    y.detach().cpu().numpy(), 
                                    z.detach().cpu().numpy(), 
                                    cmap='plasma', edgecolor='none', alpha=0.9)
    return plot_surf

# Run real-time animation loop
from matplotlib.animation import FuncAnimation
ani = FuncAnimation(fig, update_morph, frames=300, interval=30, blit=False)

plt.show()