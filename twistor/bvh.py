"""
Twistor Space Bounding Sphere Clustering & Culling (v0.3 - Complex64)
=====================================================================
Memory-efficient clustering and super-batched complex64 dual-spinor culling 
for Intel XPU Level Zero backends.
"""

import torch


def create_point_clusters(points, max_points_per_cluster=1000):
    """
    Performs Z-sorting and cluster bound generation on CPU to conserve XPU VRAM.
    """
    target_device = points.device
    pts_cpu = points.cpu()
    N = pts_cpu.shape[0]

    # Sort along Z-axis on CPU
    _, sort_indices = torch.sort(pts_cpu[:, 2])
    sorted_pts_cpu = pts_cpu[sort_indices].contiguous()

    if N <= max_points_per_cluster:
        center_cpu = sorted_pts_cpu.mean(dim=0, keepdim=True)
        radius_cpu = torch.max(torch.norm(sorted_pts_cpu - center_cpu, dim=-1)).unsqueeze(0)
        cluster_ids_cpu = torch.zeros(N, dtype=torch.long)
        return (
            sorted_pts_cpu.to(target_device),
            cluster_ids_cpu.to(target_device),
            center_cpu.to(target_device),
            radius_cpu.to(target_device),
        )

    num_clusters = max(1, N // max_points_per_cluster)
    chunk_size = (N + num_clusters - 1) // num_clusters

    centers_list = []
    radii_list = []
    cluster_ids_cpu = torch.zeros(N, dtype=torch.long)

    for cid in range(num_clusters):
        start_idx = cid * chunk_size
        end_idx = min((cid + 1) * chunk_size, N)
        if start_idx >= end_idx:
            break

        chunk = sorted_pts_cpu[start_idx:end_idx]
        center = chunk.mean(dim=0)
        radius = torch.max(torch.norm(chunk - center, dim=-1))

        centers_list.append(center)
        radii_list.append(radius)
        cluster_ids_cpu[start_idx:end_idx] = cid

    centers_cpu = torch.stack(centers_list)
    radii_cpu = torch.stack(radii_list)

    return (
        sorted_pts_cpu.to(target_device),
        cluster_ids_cpu.to(target_device),
        centers_cpu.to(target_device),
        radii_cpu.to(target_device),
    )


def cull_and_render_clustered(
    sorted_pts, cluster_ids, centers, radii, width, height, threshold,
    yaw=0.0, pitch=0.0, boost_v=0.0, zoom=1.0, super_batch_size=2048, device="cpu"
):
    """
    Culls non-visible clusters and evaluates surviving points using native 
    complex64 dual-spinor contractions in SYCL-index-safe super-batches.
    """
    from twistor.renderer import (
        generate_su22_camera_matrix,
        points_to_complex_hermitian,
        generate_transformed_ray_spinors,
    )

    R_A, B_boost, B_conf = generate_su22_camera_matrix(
        yaw=yaw, pitch=pitch, boost_v=boost_v, zoom=zoom, device=device
    )
    c_tensor = torch.tensor([0.0, 0.0, -2.0], dtype=torch.float32, device=device)

    omega_spinor, pi_spinor = generate_transformed_ray_spinors(width, height, c_tensor, R_A, B_boost, B_conf, zoom=zoom, device=device)

    cam_world_pos = torch.tensor([0.0, 0.0, -2.0], dtype=torch.float32, device=device)

    # Vectorized bounding-sphere culling pass
    dists_to_cam = torch.norm(centers - cam_world_pos, dim=-1)
    visible_cluster_mask = (dists_to_cam - radii <= 15.0) & (dists_to_cam + radii >= 0.1)

    point_mask = visible_cluster_mask[cluster_ids]
    visible_pts = sorted_pts[point_mask]

    if visible_pts.shape[0] == 0:
        return torch.zeros((height, width), dtype=torch.float32, device=device)

    num_rays = pi_spinor.shape[0]
    global_min_sq_res = torch.full((num_rays,), float("inf"), dtype=torch.float32, device=device)

    # Transpose pi_spinor to [2, R] for direct GEMM: [B, 2, 2] x [2, R] -> [B, 2, R]
    pi_t = pi_spinor.mT.contiguous()  # [2, R] complex64

    R_A_dag = R_A.conj().mT
    for batch_pts in torch.split(visible_pts, super_batch_size, dim=0):
        X_world = points_to_complex_hermitian(batch_pts)    # [B, 2, 2]
        X_complex = torch.matmul(R_A, torch.matmul(X_world, R_A_dag))

        # 2D/3D GEMM avoids SYCL 4D broadcast range overflow on Level Zero
        # [B, 2, 2] @ [2, R] -> [B, 2, R]
        pred_omega = 1j * torch.matmul(X_complex, pi_t)     # [B, 2, R]
        pred_omega = pred_omega.permute(2, 0, 1)            # [R, B, 2]

        diff = omega_spinor.unsqueeze(1) - pred_omega       # [R, B, 2]

        total_sq = torch.sum(torch.abs(diff) ** 2, dim=-1)  # [R, B]
        longitudinal = torch.sum(pi_spinor.conj().unsqueeze(1) * diff, dim=-1)
        sq_residual = torch.relu(total_sq - (torch.abs(longitudinal) ** 2))
        batch_min_sq_res, _ = torch.min(sq_residual, dim=-1)

        global_min_sq_res = torch.minimum(global_min_sq_res, batch_min_sq_res)

    intensity = torch.clamp(1.0 - (torch.sqrt(global_min_sq_res) / threshold), min=0.0, max=1.0)
    return intensity.reshape(height, width)