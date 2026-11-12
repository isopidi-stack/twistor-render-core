"""
SU(2,2) Camera Transform Demo (Normalized ASCII Visualizer)
==========================================================
"""

import sys
import os
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import twistor.renderer as twistor_r


def print_ascii(img_tensor: torch.Tensor):
    """Prints a normalized tensor image as high-contrast ASCII art."""
    chars = " .:-=+*#%@"
    h, w = img_tensor.shape
    
    # Normalize image tensor to [0, 1] range for rich ASCII gradients
    min_v = img_tensor.min()
    max_v = img_tensor.max()
    if max_v > min_v:
        normalized = (img_tensor - min_v) / (max_v - min_v)
    else:
        normalized = img_tensor

    for y in range(h):
        line = ""
        for x in range(w):
            val = normalized[y, x].item()
            idx = int(val * (len(chars) - 1))
            line += chars[idx] * 2  # Double width for aspect ratio
        print(line)
    print()


def main():
    device = "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu"
    print(f"Running SU(2,2) Camera Demo on {device.upper()}...\n")

    N = 2000
    phi = torch.linspace(0.1, 3.0, 40)
    theta = torch.linspace(0, 2 * 3.14159, 50)
    grid_phi, grid_theta = torch.meshgrid(phi, theta, indexing="ij")
    
    x = 0.5 * torch.sin(grid_phi) * torch.cos(grid_theta)
    y = 0.5 * torch.sin(grid_phi) * torch.sin(grid_theta)
    z = 0.5 * torch.cos(grid_phi) + 1.5

    pts = torch.stack([x, y, z], dim=-1).reshape(-1, 3).to(device)
    threshold = 2.4

    print("1. Standard Identity Camera:")
    img_id = twistor_r.render_point_cloud(pts, 16, 16, threshold=threshold, zoom=1.0, debug=False)
    print(f"Hit pixels: {int((img_id > 0).sum().item())} / 256")
    print_ascii(img_id)

    print("2. Conformal Zoom (1.5x Scale):")
    img_zoom = twistor_r.render_point_cloud(pts, 16, 16, threshold=threshold, zoom=1.5, debug=False)
    print(f"Hit pixels: {int((img_zoom > 0).sum().item())} / 256")
    print_ascii(img_zoom)

    print("3. Relativistic Lorentz Boost (v = 0.3c along Z):")
    img_boost = twistor_r.render_point_cloud(pts, 16, 16, threshold=threshold, boost_v=0.3, debug=False)
    print(f"Hit pixels: {int((img_boost > 0).sum().item())} / 256")
    print_ascii(img_boost)

    print("SU(2,2) Camera Transformation Test Successful!")


if __name__ == "__main__":
    main()