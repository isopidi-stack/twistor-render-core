"""
Twistor vs Euclidean Baseline Comparison (Real FP32 Core)
=========================================================
Compares execution time and throughput between:
1. Real FP32 Twistor Incidence Kernel (Decomposed 2x2 Spinors)
2. Euclidean Ray-Point Vector Distance Baseline
"""

import sys
import os
import time
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import twistor.renderer as twistor_r


def euclidean_ray_point_render(points: torch.Tensor, width: int = 32, height: int = 32, threshold: float = 0.08) -> torch.Tensor:
    """Standard Euclidean Ray-Point Distance baseline in PyTorch."""
    u = torch.linspace(-1, 1, width, device=points.device)
    v = torch.linspace(-1, 1, height, device=points.device)
    grid_y, grid_x = torch.meshgrid(v, u, indexing="ij")

    dirs = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
    dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)

    p = points.unsqueeze(0)  # (1, N, 3)
    d = dirs.unsqueeze(1)    # (M, 1, 3)

    p_dot_d = torch.sum(p * d, dim=-1)  # (M, N)
    p_sq = torch.sum(p * p, dim=-1)      # (1, N)

    dist_sq = torch.clamp(p_sq - p_dot_d ** 2, min=0.0)
    hits = (dist_sq < (threshold ** 2)) & (p_dot_d > 0)

    return hits.any(dim=-1).reshape(height, width).float()


def batched_eval(render_fn, pts: torch.Tensor, width: int = 32, height: int = 32, chunk_size: int = 2000) -> torch.Tensor:
    """Evaluates points in chunks to avoid SYCL 32-bit range overflow on GPU."""
    num_pts = pts.shape[0]
    if num_pts <= chunk_size:
        return render_fn(pts, width, height)

    out_image = torch.zeros((height, width), device=pts.device)
    for i in range(0, num_pts, chunk_size):
        chunk = pts[i : i + chunk_size]
        res = render_fn(chunk, width, height)
        out_image = torch.maximum(out_image, res)
    return out_image


def sync_device(device: str):
    """Flushes GPU execution queues for accurate timing."""
    if device == "xpu" and hasattr(torch, "xpu"):
        torch.xpu.synchronize()
    elif device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def run_comparison():
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        device = "xpu"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print("==================================================")
    print(" Twistor Real FP32 vs Euclidean Baseline")
    print(f" Execution Device: {device.upper()}")
    print("==================================================", flush=True)

    point_counts = [1000, 10000, 100000]

    header = f"{'Points (N)':<10} | {'Twistor FP32 (ms)':<18} | {'Euclidean (ms)':<14} | {'Overhead Factor':<15}"
    print(header, flush=True)
    print("-" * len(header), flush=True)

    for N in point_counts:
        pts = torch.randn(N, 3, device=device) + torch.tensor([0.0, 0.0, 2.0], device=device)

        try:
            # Warmup
            _ = batched_eval(twistor_r.render_point_cloud, pts, 32, 32)
            _ = batched_eval(euclidean_ray_point_render, pts, 32, 32)
            sync_device(device)

            # Time Real FP32 Twistor kernel
            sync_device(device)
            t0 = time.perf_counter()
            for _ in range(10):
                _ = batched_eval(twistor_r.render_point_cloud, pts, 32, 32)
            sync_device(device)
            t_twistor = (time.perf_counter() - t0) / 10.0 * 1000.0

            # Time Euclidean baseline
            sync_device(device)
            t0 = time.perf_counter()
            for _ in range(10):
                _ = batched_eval(euclidean_ray_point_render, pts, 32, 32)
            sync_device(device)
            t_euc = (time.perf_counter() - t0) / 10.0 * 1000.0

            ratio = t_twistor / t_euc if t_euc > 0 else 1.0
            print(f"{N:<10,d} | {t_twistor:<18.3f} | {t_euc:<14.3f} | {ratio:<14.2f}x", flush=True)

        except Exception as e:
            print(f"{N:<10,d} | FAILED on {device.upper()}: {e}", flush=True)

    print("\n==================================================")
    print(" Benchmark Complete!")
    print("==================================================", flush=True)


if __name__ == "__main__":
    run_comparison()