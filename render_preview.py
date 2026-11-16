import torch
import math
from palatial_gpo_ray_optics import PalatialGPORayOptics

resolution = 512
device = torch.device("cpu")

theta = torch.linspace(0, 2 * math.pi, 7, dtype=torch.float64)[:-1]
cx = 1.2 * torch.cos(theta)
cy = 1.2 * torch.sin(theta)
cz = torch.full_like(cx, 3.5) + 0.5 * torch.sin(3.0 * theta)
lattice_centers = torch.stack([cx, cy, cz], dim=-1)

central_node = torch.tensor([[0.0, 0.0, 2.5]], dtype=torch.float64)
field_centers = torch.cat([central_node, lattice_centers], dim=0)
harmonic_weights = torch.tensor([2.0, 1.2, 0.9, 1.4, 1.0, 1.3, 0.8], dtype=torch.float64)

print("[CP^4 Renderer] Executing Z-buffer-free GPO standing wave HUD pass (512x512)...")
res = PalatialGPORayOptics.render_gpo_hud_frame(
    height=resolution,
    width=resolution,
    field_centers=field_centers,
    harmonic_weights=harmonic_weights,
    fov_deg=55.0,
    sigma_aperture=0.20
)

rgb_tensor = res["hud_frame"].clamp(0.0, 1.0)
rgb_np = (rgb_tensor.numpy() * 255).astype("uint8")

# Save using matplotlib if available, or write raw PPM binary format directly
try:
    import matplotlib.pyplot as plt
    plt.imsave("gpo_hud_preview.png", rgb_np)
    print("[Success] Standing wave HUD frame saved to gpo_hud_preview.png via matplotlib!")
except ImportError:
    # Fallback to binary PPM format (opens in any image viewer)
    header = f"P6\n{resolution} {resolution}\n255\n".encode("ascii")
    with open("gpo_hud_preview.ppm", "wb") as f:
        f.write(header)
        f.write(rgb_np.tobytes())
    print("[Success] Standing wave HUD frame saved to gpo_hud_preview.ppm (raw binary PPM)!")

print(f"  -> Max Triad Completeness Error: {float(torch.max(torch.abs(res['triad_sum'] - 1.0))):.2e}")
print(f"  -> Channel Peaks (Green: {float(torch.max(res['c_green'])):.3f}, Purple: {float(torch.max(res['c_purple'])):.3f}, Orange: {float(torch.max(res['c_orange'])):.3f})")
