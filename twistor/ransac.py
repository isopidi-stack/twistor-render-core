import torch
from twistor.localization import spinor_to_transverse_basis, solve_camera_position_spinor, verify_cheirality

def solve_camera_position_ransac(
    pi_world: torch.Tensor,
    landmarks: torch.Tensor,
    num_hypotheses: int = 256,
    angle_threshold_rad: float = 0.008,
    min_inliers: int = 3,
    sample_size: int = 3,
    cond_threshold: float = 1e4
):
    """
    Robust spinor camera localization with:
      1. Sampling without replacement (distinct landmarks per hypothesis)
      2. Exact angular residual: theta_i = arcsin(clamp(e_i / d_i, 0, 1)) (no erroneous sqrt(2) factor)
      3. SVD singular-value conditioning gate + positive-depth cheirality gate
      4. Explicit status reporting ('robust_success' vs 'failed')
    """
    device = landmarks.device
    n_pts = landmarks.shape[0]
    if n_pts < sample_size:
        raise ValueError(f"Requires at least {sample_size} landmarks, got {n_pts}.")

    a, b, n = spinor_to_transverse_basis(pi_world)

    # 1. Sample distinct indices per hypothesis without replacement
    rand_scores = torch.rand((num_hypotheses, n_pts), device=device)
    idx = torch.topk(rand_scores, k=sample_size, dim=-1).indices  # (H, sample_size)

    a_sub = a[idx]          # (H, S, 3)
    b_sub = b[idx]          # (H, S, 3)
    X_sub = landmarks[idx]  # (H, S, 3)

    A_sub = torch.cat([a_sub, b_sub], dim=1)  # (H, 2*S, 3)
    d_sub = torch.cat([
        torch.sum(a_sub * X_sub, dim=-1),
        torch.sum(b_sub * X_sub, dim=-1)
    ], dim=1)  # (H, 2*S)

    # 2. Check conditioning via singular values and reject degenerate hypotheses
    _, S_vals, _ = torch.linalg.svd(A_sub, full_matrices=False)
    valid_cond = (S_vals[:, -1] > 1e-5) & ((S_vals[:, 0] / torch.clamp(S_vals[:, -1], min=1e-12)) < cond_threshold)

    # Solve all hypotheses in parallel
    C_hyps = torch.linalg.lstsq(A_sub, d_sub.unsqueeze(-1)).solution.squeeze(-1)  # (H, 3)

    # 3. Compute exact perpendicular distance e_i and exact angular error theta_i = arcsin(clamp(e_i / d_i, 0, 1))
    disp = landmarks.unsqueeze(0) - C_hyps.unsqueeze(1)  # (H, N, 3)
    proj_a = torch.sum(a.unsqueeze(0) * disp, dim=-1)
    proj_b = torch.sum(b.unsqueeze(0) * disp, dim=-1)
    e_i = torch.sqrt(proj_a**2 + proj_b**2)              # (H, N)
    d_i = torch.norm(disp, dim=-1)                       # (H, N)

    # Enforce positive depth (cheirality: landmark in front of camera) and minimum distance > 1e-4
    depths = torch.sum(disp * n.unsqueeze(0), dim=-1)
    valid_geom = (depths > 0.0) & (d_i > 1e-4)

    sin_theta = torch.clamp(e_i / torch.clamp(d_i, min=1e-12), 0.0, 1.0)
    theta_rad = torch.asin(sin_theta)

    inlier_matrix = (theta_rad < angle_threshold_rad) & valid_geom & valid_cond.unsqueeze(1)
    inlier_counts = torch.sum(inlier_matrix, dim=-1)

    best_count, best_idx = torch.max(inlier_counts, dim=0)
    best_inliers = inlier_matrix[best_idx]

    if int(best_count.item()) < min_inliers:
        return {
            "status": "failed",
            "position": None,
            "inliers": torch.zeros(n_pts, dtype=torch.bool, device=device),
            "num_inliers": int(best_count.item())
        }

    # Polish on best inlier set
    polished = solve_camera_position_spinor(pi_world[best_inliers], landmarks[best_inliers], cond_threshold=cond_threshold)
    return {
        "status": "robust_success" if polished["is_valid"] else "degenerate_inlier_set",
        "position": polished["position"],
        "inliers": best_inliers,
        "num_inliers": int(best_count.item()),
        "condition_number": polished["condition_number"]
    }