"""
Batched GPU Spinor RANSAC for Outlier-Robust 3D Camera Localization
===================================================================
Evaluates hundreds of minimal 3-landmark twistor hypotheses in parallel
on GPU/XPU/CPU using scale-invariant angular spinor residuals, rejecting
false feature matches (outliers) before running a final least-squares
spinor refinement over clean inliers.
"""

import torch
from examples.instant_robot_snap import solve_camera_position_closed_form


def _build_spinor_linear_rows(landmarks_xyz, ray_spinors_pi):
    """
    Constructs real/imaginary twistor incidence linear rows A_r, A_i [N, 3]
    and RHS vectors b_r, b_i [N] from 3D landmarks and world null spinors.
    """
    a0 = ray_spinors_pi[:, 0].real
    b0 = ray_spinors_pi[:, 0].imag
    a1 = ray_spinors_pi[:, 1].real
    b1 = ray_spinors_pi[:, 1].imag

    re_p0_sq = a0 * a0 - b0 * b0
    im_p0_sq = 2.0 * a0 * b0
    re_p1_sq = a1 * a1 - b1 * b1
    im_p1_sq = 2.0 * a1 * b1
    re_p0_p1 = a0 * a1 - b0 * b1
    im_p0_p1 = a0 * b1 + b0 * a1

    r_cx = re_p0_sq - re_p1_sq
    r_cy = -(im_p0_sq + im_p1_sq)
    r_cz = -2.0 * re_p0_p1

    i_cx = im_p0_sq - im_p1_sq
    i_cy = re_p0_sq + re_p1_sq
    i_cz = -2.0 * im_p0_p1

    x = landmarks_xyz[:, 0]
    y = landmarks_xyz[:, 1]
    z = landmarks_xyz[:, 2]

    A_r = torch.stack([r_cx, r_cy, r_cz], dim=-1)  # [N, 3]
    A_i = torch.stack([i_cx, i_cy, i_cz], dim=-1)  # [N, 3]
    b_r = r_cx * x + r_cy * y + r_cz * z           # [N]
    b_i = i_cx * x + i_cy * y + i_cz * z           # [N]
    return A_r, A_i, b_r, b_i


def solve_camera_position_ransac(
    landmarks_xyz,
    ray_spinors_pi,
    num_hypotheses=512,
    inlier_threshold=0.008,
):
    """
    Outlier-robust 3-DoF camera localization via batched twistor RANSAC.

    Args:
        landmarks_xyz: [N, 3] float32 world coordinates
        ray_spinors_pi: [N, 2] complex64 world-rotated null ray spinors
        num_hypotheses: Number of parallel 3-point hypotheses (default 512)
        inlier_threshold: Scale-invariant angular spinor residual cutoff in radians
                          (default 0.008 rad ~ 0.45 deg, ~3-sigma for 1.5e-3 UV noise)

    Returns:
        refined_pos: [3] float32 camera position refined on all inliers
        inlier_mask: [N] bool tensor of detected inliers
    """
    n_pts = landmarks_xyz.shape[0]
    device = landmarks_xyz.device

    if n_pts <= 4:
        pos = solve_camera_position_closed_form(landmarks_xyz, ray_spinors_pi)
        return pos, torch.ones(n_pts, dtype=torch.bool, device=device)

    A_r, A_i, b_r, b_i = _build_spinor_linear_rows(landmarks_xyz, ray_spinors_pi)

    # 1. Sample [B, 3] random landmark indices in parallel
    idx = torch.randint(0, n_pts, (num_hypotheses, 3), device=device)

    # Gather [B, 3, 3] real & imag design blocks and stack into [B, 6, 3]
    A_sub = torch.cat([A_r[idx], A_i[idx]], dim=1)                # [B, 6, 3]
    b_sub = torch.cat([b_r[idx], b_i[idx]], dim=1).unsqueeze(-1)  # [B, 6, 1]

    # 2. Batched 3x3 normal-equation solve across all B hypotheses simultaneously
    eye3 = 1e-7 * torch.eye(3, device=device).unsqueeze(0)
    ATA = torch.matmul(A_sub.mT, A_sub) + eye3                    # [B, 3, 3]
    ATb = torch.matmul(A_sub.mT, b_sub)                           # [B, 3, 1]
    candidates = torch.linalg.solve(ATA, ATb).squeeze(-1)         # [B, 3]

    # 3. Scale-invariant angular spinor residual across [B, N]
    # Normalize raw spinor incidence residual by sqrt(2) * ||X_i - C_b||
    res_r = torch.matmul(candidates, A_r.mT) - b_r.unsqueeze(0)   # [B, N]
    res_i = torch.matmul(candidates, A_i.mT) - b_i.unsqueeze(0)   # [B, N]
    raw_res = torch.sqrt(res_r * res_r + res_i * res_i)           # [B, N]

    # Depth normalization: ||X_i - C_b|| = sqrt(||X||^2 + ||C||^2 - 2 C.X)
    x_norm_sq = (landmarks_xyz * landmarks_xyz).sum(dim=-1).unsqueeze(0)  # [1, N]
    c_norm_sq = (candidates * candidates).sum(dim=-1).unsqueeze(1)        # [B, 1]
    dist = torch.sqrt(
        torch.clamp(x_norm_sq + c_norm_sq - 2.0 * torch.matmul(candidates, landmarks_xyz.mT), min=1e-6)
    )
    angular_residuals = raw_res / (1.41421356 * dist)             # [B, N] in radians

    inlier_counts = (angular_residuals < inlier_threshold).sum(dim=-1)  # [B]
    best_b = torch.argmax(inlier_counts)
    inlier_mask = angular_residuals[best_b] < inlier_threshold

    if inlier_mask.sum() < 3:
        pos = solve_camera_position_closed_form(landmarks_xyz, ray_spinors_pi)
        return pos, torch.ones(n_pts, dtype=torch.bool, device=device)

    # 4. Final closed-form least-squares refinement on all clean inliers
    refined_pos = solve_camera_position_closed_form(
        landmarks_xyz[inlier_mask], ray_spinors_pi[inlier_mask]
    )
    return refined_pos, inlier_mask
