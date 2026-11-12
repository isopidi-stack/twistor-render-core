"""
Twistor Cluster Culling vs Brute-Force Performance Benchmark
=============================================================
"""

import sys
import os
import time
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import twistor.renderer as twistor_r
import twistor.bvh as twistor_bvh


def main():
    device = "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu"
    print("=" * 60)
    print(f" Twistor Cluster Culling Acceleration Benchmark")
    print(f" Execution Device: {device.upper()}")
    print("=" * 60)

    point_counts = [10000, 50000, 100000]
    width, height = 64, 64
    threshold = 2.0

    print(f"{'Points (N)':<12} | {'Brute-Force (ms)':<18} | {'Clustered (ms)':<16} | {'Speedup':<10}")
    print("-" * 65)

    for N in point_counts:
        if device == "xpu":
            torch.xpu.empty_cache()

        near_pts = torch.randn(N // 2, 3, device=device) * 0.5 + torch.tensor([0.0, 0.0, 1.2], device=device)
        far_pts = torch.randn(N // 2, 3, device=device) * 2.0 + torch.tensor([0.0, 0.0, 30.0], device=device)
        pts = torch.cat([near_pts, far_pts], dim=0)

        # 1. Measure Clustered Culling First
        sorted_pts, cluster_ids, centers, radii = twistor_bvh.create_point_clusters(pts, max_points_per_cluster=1000)

        start_cull = time.perf_counter()
        _ = twistor_bvh.cull_and_render_clustered(
            sorted_pts, cluster_ids, centers, radii, width, height, threshold, device=device
        )
        if device == "xpu":
            torch.xpu.synchronize()
        time_cull_ms = (time.perf_counter() - start_cull) * 1000.0

        # Free culling tensors
        del sorted_pts, cluster_ids, centers, radii
        if device == "xpu":
            torch.xpu.empty_cache()

        # 2. Measure Brute-Force Timing
        start_bf = time.perf_counter()
        _ = twistor_r.render_point_cloud(pts, width, height, threshold, device=device)
        if device == "xpu":
            torch.xpu.synchronize()
        time_bf_ms = (time.perf_counter() - start_bf) * 1000.0

        del pts
        if device == "xpu":
            torch.xpu.empty_cache()

        speedup = time_bf_ms / max(time_cull_ms, 0.001)

        print(f"{N:<12,d} | {time_bf_ms:<18.3f} | {time_cull_ms:<16.3f} | {speedup:<9.2f}x")

    print("=" * 60)
    print(" Benchmark Complete!")


if __name__ == "__main__":
    main()