import torch

def pixel_to_stereographic_spinor(px: torch.Tensor, py: torch.Tensor, fx: float, fy: float, cx: float, cy: float) -> torch.Tensor:
    xn = (px - cx) / fx
    yn = (py - cy) / fy
    r = torch.sqrt(1.0 + xn**2 + yn**2)
    u = xn / (r + 1.0)
    v = yn / (r + 1.0)
    norm = torch.sqrt(1.0 + u**2 + v**2)
    p0 = torch.complex(1.0 / norm, torch.zeros_like(norm))
    p1 = torch.complex(u / norm, v / norm)
    return torch.stack([p0, p1], dim=-1)

def spinor_to_transverse_basis(pi: torch.Tensor):
    # Enforce unit spinor normalization
    norm = torch.sqrt(torch.sum(torch.abs(pi)**2, dim=-1, keepdim=True) + 1e-12)
    pi = pi / norm
    p0, p1 = pi[..., 0], pi[..., 1]
    p0_sq, p1_sq, p0_p1 = p0**2, p1**2, p0 * p1

    a = torch.stack([
        (p0_sq - p1_sq).real,
        -(p0_sq + p1_sq).imag,
        -2.0 * p0_p1.real
    ], dim=-1)

    b = torch.stack([
        (p0_sq - p1_sq).imag,
        (p0_sq + p1_sq).real,
        -2.0 * p0_p1.imag
    ], dim=-1)

    n = torch.stack([
        2.0 * (p0 * torch.conj(p1)).real,
        -2.0 * (p0 * torch.conj(p1)).imag,
        torch.abs(p0)**2 - torch.abs(p1)**2
    ], dim=-1)
    return a, b, n

def verify_cheirality(C: torch.Tensor, landmarks: torch.Tensor, rays_n: torch.Tensor) -> torch.Tensor:
    """
    Verifies positive landmark depth (lambda_i > 0) along forward camera rays.
    """
    disp = landmarks - C.unsqueeze(0)
    depths = torch.sum(disp * rays_n, dim=-1)
    return depths > 0.0

def solve_camera_position_spinor(pi_world: torch.Tensor, landmarks: torch.Tensor, cond_threshold: float = 1e5):
    """
    Exact linear spinor incidence solver with SVD rank/conditioning diagnostics and cheirality verification.
    """
    a, b, n = spinor_to_transverse_basis(pi_world)
    N = landmarks.shape[0]
    A = torch.empty((2 * N, 3), dtype=landmarks.dtype, device=landmarks.device)
    d = torch.empty((2 * N,), dtype=landmarks.dtype, device=landmarks.device)

    A[0::2] = a
    A[1::2] = b
    d[0::2] = torch.sum(a * landmarks, dim=-1)
    d[1::2] = torch.sum(b * landmarks, dim=-1)

    # Conditioning & Rank Check via Singular Value Decomposition
    _, S, _ = torch.linalg.svd(A, full_matrices=False)
    rank = int(torch.sum(S > 1e-6).item())
    cond = float((S[0] / torch.clamp(S[-1], min=1e-12)).item()) if S.numel() >= 3 else float('inf')
    is_well_conditioned = (rank == 3) and (cond < cond_threshold)

    C_est = torch.linalg.lstsq(A, d.unsqueeze(-1)).solution.squeeze(-1)
    cheirality_mask = verify_cheirality(C_est, landmarks, n)

    return {
        "position": C_est,
        "rank": rank,
        "condition_number": cond,
        "is_valid": is_well_conditioned and bool(torch.all(cheirality_mask).item()),
        "cheirality_mask": cheirality_mask
    }

def solve_camera_position_euclidean_reference(rays_n: torch.Tensor, landmarks: torch.Tensor):
    """
    Independent Euclidean orthogonal projector solver (P_i = I - n_i n_i^T) solved via QR/SVD lstsq.
    Does not reuse spinor coefficients.
    """
    rays_n = rays_n / torch.norm(rays_n, dim=-1, keepdim=True)
    I = torch.eye(3, dtype=landmarks.dtype, device=landmarks.device).unsqueeze(0)
    P = I - torch.einsum("ni,nj->nij", rays_n, rays_n)
    A = P.reshape(-1, 3)
    b = torch.einsum("nij,nj->ni", P, landmarks).reshape(-1)
    C_ref = torch.linalg.lstsq(A, b.unsqueeze(-1)).solution.squeeze(-1)
    return C_ref