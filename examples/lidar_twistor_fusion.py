import time
import torch
import numpy as np
import twistor

def generate_synthetic_lidar_and_camera(num_points: int = 2000, device: str = "cpu"):
    np.random.seed(42)
    z = np.random.uniform(1.0, 8.0, num_points)
    x = np.random.uniform(-3.0, 3.0, num_points)
    y = np.random.uniform(-1.5, 1.5, num_points)
    points_3d = np.stack([x, y, z], axis=-1).astype(np.float32)

    fx, fy, cx, cy = 500.0, 500.0, 320.0, 240.0
    u = (fx * (x / z)) + cx
    v = (fy * (y / z)) + cy

    u += np.random.normal(0.0, 0.5, num_points)
    v += np.random.normal(0.0, 0.5, num_points)
    pixels = np.stack([u, v], axis=-1).astype(np.float32)

    outlier_mask = np.random.rand(num_points) < 0.35
    pixels[outlier_mask, 0] = np.random.uniform(0, 640, np.sum(outlier_mask))
    pixels[outlier_mask, 1] = np.random.uniform(0, 480, np.sum(outlier_mask))

    return (
        torch.tensor(points_3d, dtype=torch.float32, device=device),
        torch.tensor(pixels, dtype=torch.float32, device=device),
        (fx, fy, cx, cy)
    )

def run_lidar_twistor_fusion():
    device = twistor.get_best_device()
    print(f"=== Twistor-Vision LiDAR + Camera Fusion Engine [{str(device).upper()}] ===")

    points_3d, pixels, (fx, fy, cx, cy) = generate_synthetic_lidar_and_camera(2000, device)
    uv_t = twistor.pixel_to_stereographic_uv(pixels[:, 0], pixels[:, 1], fx, fy, cx, cy)

    q_identity = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=device)
    w, x, y, z = q_identity[0], q_identity[1], q_identity[2], q_identity[3]
    R_A = torch.stack([
        torch.stack([torch.complex(w, z), torch.complex(y, x)], dim=-1),
        torch.stack([torch.complex(-y, x), torch.complex(w, -z)], dim=-1),
    ], dim=-2).to(torch.complex64)

    print(f"Fused Points (N)     : {len(points_3d)}")
    print(f"Outlier Ratio        : ~35%")
    print(f"Computing batched spinor incidence contraction in CP^3...")

    t0 = time.perf_counter()
    pos_est, inliers = twistor.localize_ransac(
        points_3d,
        uv_t,
        R_A=R_A,
        num_hypotheses=256,
        inlier_threshold=0.034,
    )
    if str(device) == "cuda":
        torch.cuda.synchronize()
    elif str(device) == "xpu":
        torch.xpu.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    inliers_np = inliers.cpu().numpy()
    n_inliers = int(inliers_np.sum())
    recovered_pos = pos_est.cpu().numpy()

    print("\n================ LIDAR-TWISTOR FUSION RESULTS ================")
    print(f"Execution Time       : {elapsed_ms:.3f} ms ({1000.0/max(elapsed_ms, 1e-3):.0f} Hz)")
    print(f"Inliers Retained     : {n_inliers} / {len(points_3d)} ({1000.0*n_inliers/len(points_3d)/10.0:.1f}%)")
    print(f"Estimated Position   : X={recovered_pos[0]:+.4f} m, Y={recovered_pos[1]:+.4f} m, Z={recovered_pos[2]:+.4f} m")
    print(f"Ground Truth Position: X=+0.0000 m, Y=+0.0000 m, Z=+0.0000 m")
    print(f"Position Error       : {float(np.linalg.norm(recovered_pos)):.5f} meters")
    print("==============================================================\n")
    print("SUCCESS: Direct CP^3 null-ray sensor fusion completed without Z-buffers or 4x4 matrices!")

if __name__ == "__main__":
    run_lidar_twistor_fusion()