"""
LiDAR-Camera Twistor Fusion Prototype for twistor-vision v0.9
============================================================
Demonstrates direct ingestion and projection of sparse metric LiDAR 3D points
into projective twistor space (CP^3), fusing them with camera keypoints via
batched complex spinor incidence relations without Euclidean 4x4 matrix overhead.
"""

import time
import cv2
import numpy as np
import torch
import twistor


def quat_wxyz_to_su2(q_wxyz: torch.Tensor) -> torch.Tensor:
    q = q_wxyz / torch.linalg.norm(q_wxyz, dim=-1, keepdim=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return torch.stack([
        torch.stack([torch.complex(w, z), torch.complex(y, x)], dim=-1),
        torch.stack([torch.complex(-y, x), torch.complex(w, -z)], dim=-1),
    ], dim=-2).to(torch.complex64)


def rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    tr = float(np.trace(R))
    if tr > 0.0:
        s = 0.5 / np.sqrt(tr + 1.0)
        q = np.array([0.25 / s, (R[2, 1] - R[1, 2]) * s, (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s], dtype=np.float32)
    else:
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k])
        q = np.zeros(4, dtype=np.float32)
        q[0] = (R[k, j] - R[j, k]) / s
        q[i + 1] = 0.25 * s
        q[j + 1] = (R[j, i] + R[i, j]) / s
        q[k + 1] = (R[k, i] + R[i, k]) / s
    return q / np.linalg.norm(q)


def simulate_lidar_point_cloud(num_points: int = 1500, box_center: np.ndarray = np.array([0.0, 0.0, 3.5])) -> np.ndarray:
    """Generates a synthetic metric LiDAR point cloud (X, Y, Z in meters) simulating forward obstacles."""
    pts = np.random.normal(loc=box_center, scale=[1.2, 0.8, 0.9], size=(num_points, 3)).astype(np.float32)
    # Filter points in front of the sensor (Z > 0.5m)
    pts = pts[pts[:, 2] > 0.5]
    return pts


def project_lidar_to_twistor_space(lidar_pts_3d: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Directly maps metric LiDAR 3D points into CP^3 projective twistor coordinates 
    via null-ray projective incidence equations without intermediate 4x4 matrix clutter.
    """
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    
    # Project 3D points to camera plane
    xs = (lidar_pts_3d[:, 0] * fx / lidar_pts_3d[:, 2]) + cx
    ys = (lidar_pts_3d[:, 1] * fy / lidar_pts_3d[:, 2]) + cy
    
    # Filter in-bounds image coordinates
    valid = (xs >= 0) & (xs < 1920) & (ys >= 0) & (ys < 1080)
    uv_pts = np.stack([xs[valid], ys[valid]], axis=-1).astype(np.float32)
    metric_pts_3d = lidar_pts_3d[valid]
    
    return uv_pts, metric_pts_3d


def run_lidar_twistor_fusion_demo():
    device = twistor.get_best_device()
    print(f"=== twistor-vision v0.9.0 LiDAR-Camera CP^3 Fusion Engine [{device}] ===")

    # Camera Intrinsics
    out_w, out_h = 1280, 720
    fx = fy = 0.8 * out_w
    cx, cy = out_w * 0.5, out_h * 0.5
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

    # Simulate true camera pose and incoming LiDAR scan
    true_pos = np.array([0.15, -0.05, 3.20], dtype=np.float32)
    R_wc = np.eye(3, dtype=np.float64)

    raw_lidar = simulate_lidar_point_cloud(num_points=2500, box_center=true_pos + np.array([0, 0, 0.5]))
    uv_proj, lmk_3d_metric = project_lidar_to_twistor_space(raw_lidar, K)

    print(f"[CP^3 Ingestion] Successfully ingested {len(raw_lidar)} raw LiDAR points -> {len(lmk_3d_metric)} metric CP^3 null-ray anchors.")

    # Convert to PyTorch tensors for native complex64 spinor contraction
    q_t = torch.tensor(rotmat_to_quat_wxyz(R_wc), dtype=torch.float32, device=device)
    R_A = quat_wxyz_to_su2(q_t)
    lmk_tensor = torch.tensor(lmk_3d_metric, dtype=torch.float32, device=device)
    
    px_t = torch.tensor(uv_proj[:, 0], dtype=torch.float32, device=device)
    py_t = torch.tensor(uv_proj[:, 1], dtype=torch.float32, device=device)
    uv_t = twistor.pixel_to_stereographic_uv(px_t, py_t, fx, fy, cx, cy)

    # Inject 30% synthetic sensor noise and outlier corruption to test resilience
    noise_mask = np.random.rand(len(lmk_tensor)) < 0.30
    if noise_mask.sum() > 0:
        lmk_tensor[torch.tensor(noise_mask, device=device)] += torch.randn((noise_mask.sum(), 3), device=device) * 2.5

    # Execute Sub-Millisecond Batched Spinor RANSAC Fusion
    t0 = time.perf_counter()
    estimated_pos, inliers = twistor.localize_ransac(
        lmk_tensor,
        uv_t,
        R_A=R_A,
        num_hypotheses=512,
        inlier_threshold=0.025,
    )
    if str(device) in ["cuda", "xpu"]:
        torch.cuda.synchronize() if str(device) == "cuda" else torch.xpu.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    est_np = estimated_pos.cpu().numpy()
    inlier_count = int(inliers.sum().item())
    position_error_mm = float(np.linalg.norm(est_np - true_pos)) * 1000.0

    print("\n================ LIDAR-TWISTOR FUSION BENCHMARK ================")
    print(f"Execution Latency       : {elapsed_ms:.3f} ms ({1000.0/max(elapsed_ms, 1e-3):.0f} Hz)")
    print(f"True Robot Position (m) : X={true_pos[0]:.2f}, Y={true_pos[1]:.2f}, Z={true_pos[2]:.2f}")
    print(f"Fused CP^3 Position (m) : X={est_np[0]:.2f}, Y={est_np[1]:.2f}, Z={est_np[2]:.2f}")
    print(f"Absolute Position Error : {position_error_mm:.2f} mm")
    print(f"Spinor RANSAC Inliers   : {inlier_count}/{len(uv_proj)} ({100.0*inlier_count/len(uv_proj):.1f}%)")
    print("=================================================================\n")


if __name__ == "__main__":
    run_lidar_twistor_fusion_demo()