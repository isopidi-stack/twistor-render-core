"""
Twistor Render Core - Visual Render Sample
==========================================
Generates a 3D point cloud sphere, renders it via Penrose incidence kernels,
and exports the resulting image buffer to twistor_render.png.
"""

import sys
import os

# Include root directory in module search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import matplotlib.pyplot as plt
import twistor.renderer as r


def generate_sphere_points(num_points: int = 8000, radius: float = 0.6, z_offset: float = 2.0) -> torch.Tensor:
    """Generates uniform 3D surface points for a sphere positioned in front of the camera."""
    phi = torch.rand(num_points) * 2.0 * 3.141592653589793
    costheta = torch.rand(num_points) * 2.0 - 1.0
    theta = torch.acos(costheta)

    x = radius * torch.sin(theta) * torch.cos(phi)
    y = radius * torch.sin(theta) * torch.sin(phi)
    z = radius * torch.cos(theta) + z_offset

    return torch.stack([x, y, z], dim=-1)


def main():
    print("Generating 3D sphere point cloud...")
    pts = generate_sphere_points(num_points=8000, radius=0.6, z_offset=2.0)

    print("Rendering 128x128 buffer via Twistor incidence kernels...")
    img_buffer = r.render_point_cloud(pts, width=128, height=128, threshold=0.08)

    plt.figure(figsize=(6, 6))
    plt.imshow(img_buffer.cpu().numpy(), cmap="inferno", origin="upper")
    plt.title("Twistor Render Core - 3D Sphere (128x128)", fontsize=12, pad=12)
    plt.axis("off")
    plt.tight_layout()

    output_path = "twistor_render.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Success! Image exported to {output_path}")


if __name__ == "__main__":
    main()