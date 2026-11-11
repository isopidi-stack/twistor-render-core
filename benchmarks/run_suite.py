"""
Twistor Render Core - Phase 4 Benchmark Suite
=============================================
Profiles performance and throughput of Penrose twistor incidence calculations
across scaled point cloud sizes (N=1,000 to 1,000,000) on the active hardware backend.
"""

import sys
import os

# Ensure the root directory is on the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import torch
import twistor.geometry as g
import twistor.incidence as inc


def run_benchmark():
    device = g.get_best_device()
    print("==================================================")
    print(" Twistor Render Core - Performance Benchmark Suite")
    print(f" Active Hardware Acceleration Device: {device}")
    print("==================================================\n")

    # Ray direction configuration (+Z direction)
    omega = torch.zeros(2, 1, dtype=torch.complex64, device=device)
    pi = torch.tensor([[0.0 + 0.0j], [1.0 + 0.0j]], dtype=torch.complex64, device=device)

    point_counts = [1_000, 10_000, 100_000, 1_000_000]

    print(f"{'Point Count (N)':<18} | {'Execution Time (ms)':<20} | {'Throughput (Points/sec)':<25}")
    print("-" * 70)

    for N in point_counts:
        # Generate random 3D points in front of camera
        pts = torch.rand(N, 3, device=device) * 4.0 - 2.0
        pts[:, 2] = torch.abs(pts[:, 2]) + 1.0  # Ensure z > 0

        # Warmup iteration
        H = g.points_to_hermitian(pts)
        _ = inc.compute_incidence_residual(H, omega, pi)
        if device.type == "cuda":
            torch.cuda.synchronize()

        # Benchmark iterations
        runs = 50 if N <= 100_000 else 10
        start_time = time.perf_counter()
        for _ in range(runs):
            H = g.points_to_hermitian(pts)
            _ = inc.compute_incidence_residual(H, omega, pi)
            if device.type == "cuda":
                torch.cuda.synchronize()
        end_time = time.perf_counter()

        avg_time_ms = ((end_time - start_time) / runs) * 1000.0
        throughput = N / (avg_time_ms / 1000.0)

        print(f"{N:<18,d} | {avg_time_ms:<20.3f} | {throughput:<25,.0f}")

    print("\n==================================================")
    print(" Benchmark Complete!")
    print("==================================================")


if __name__ == "__main__":
    run_benchmark()