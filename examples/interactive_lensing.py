# examples/interactive_lensing.py
import torch
import numpy as np
import matplotlib.pyplot as plt

# Ensure device parity with your local setup (CUDA, XPU, or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running interactive twistor lensing sandbox on: {device}")

# Setup screen resolution grid for ray spinors (pi_A')
res = 300
x = torch.linspace(-2.0, 2.0, res, device=device)
y = torch.linspace(-2.0, 2.0, res, device=device)
# Use indexing='xy' so standard Cartesian mapping aligns with screen coordinates
X_grid, Y_grid = torch.meshgrid(x, y, indexing='xy')

# Background coordinate grid texture
background = ((torch.sin(3.0 * X_grid) * torch.cos(3.0 * Y_grid)) > 0).float()

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_title("Twistor Gravitational Lensing Sandbox (Mouse Interactive)", color="white")
fig.patch.set_facecolor('black')
ax.set_facecolor('black')

# Initial mass center position (X, Y)
current_mass_pos = [0.0, 0.0]

im = ax.imshow(background.cpu().numpy(), cmap='inferno', extent=[-2, 2, -2, 2], origin='lower')

def compute_lensing_frame(pos):
    # pos[0] is X (horizontal), pos[1] is Y (vertical)
    dx = X_grid - pos[0]
    dy = Y_grid - pos[1]
    r_sq = dx**2 + dy**2 + 1e-2 # Epsilon for numerical stability
    
    # Schwarzschild-like deflection shift derived from spinor incidence perturbation
    strength = 0.5
    shift_x = strength * (dx / r_sq)
    shift_y = strength * (dy / r_sq)
    
    sample_x = torch.clamp(X_grid + shift_x, -2.0, 2.0)
    sample_y = torch.clamp(Y_grid + shift_y, -2.0, 2.0)
    
    idx_x = ((sample_x + 2.0) / 4.0 * (res - 1)).long()
    idx_y = ((sample_y + 2.0) / 4.0 * (res - 1)).long()
    
    return background[idx_y, idx_x]

def on_mouse_move(event):
    global current_mass_pos
    if event.xdata is not None and event.ydata is not None:
        # event.xdata -> horizontal axis (X), event.ydata -> vertical axis (Y)
        current_mass_pos = [event.xdata, event.ydata]
        
        # Re-compute warped frame tensorially
        warped_frame = compute_lensing_frame(current_mass_pos)
        im.set_data(warped_frame.cpu().numpy())
        fig.canvas.draw_idle()

# Connect the mouse motion event handler
fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)

plt.axis('off')
plt.show()