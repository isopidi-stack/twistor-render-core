import torch

EPSILON = torch.tensor([[0.0, 1.0], [-1.0, 0.0]], dtype=torch.complex64)

def sl2c_from_lie_algebra(axis_angle: torch.Tensor, boost_rapidity: torch.Tensor = None) -> torch.Tensor:
    """
    Constructs Lambda in SL(2,C) (or SU(2) when boost_rapidity is None) via the exact 
    analytical Cayley-Hamilton Lie algebra exponential exp(-i/2 * theta_k * sigma_k - 1/2 * eta_k * sigma_k).
    """
    wx, wy, wz = axis_angle[..., 0], axis_angle[..., 1], axis_angle[..., 2]
    if boost_rapidity is not None:
        ex, ey, ez = boost_rapidity[..., 0], boost_rapidity[..., 1], boost_rapidity[..., 2]
        cx, cy, cz = torch.complex(wx, -ex), torch.complex(wy, -ey), torch.complex(wz, -ez)
    else:
        z = torch.zeros_like(wx)
        cx, cy, cz = torch.complex(wx, z), torch.complex(wy, z), torch.complex(wz, z)

    m00 = -0.5j * cz
    m01 = -0.5j * (cx - 1j * cy)
    m10 = -0.5j * (cx + 1j * cy)
    m11 =  0.5j * cz

    M = torch.stack([torch.stack([m00, m01], dim=-1),
                     torch.stack([m10, m11], dim=-1)], dim=-2)
    omega_sq = -0.25 * (cx**2 + cy**2 + cz**2)
    omega = torch.sqrt(omega_sq + 1e-18j)
    cosh_w = torch.cosh(omega)
    sinh_over_w = torch.where(torch.abs(omega) > 1e-7, torch.sinh(omega) / omega, torch.ones_like(omega))
    eye = torch.eye(2, dtype=M.dtype, device=M.device).expand_as(M)
    return cosh_w.unsqueeze(-1).unsqueeze(-1) * eye + sinh_over_w.unsqueeze(-1).unsqueeze(-1) * M

def sl2c_inverse_dagger(U: torch.Tensor) -> torch.Tensor:
    """
    Computes (U^dagger)^{-1} for U in SL(2,C) analytically using det(U) = 1:
    For U^dagger = [[a, b], [c, d]], (U^dagger)^{-1} = [[d, -b], [-c, a]] = eps @ U^* @ eps^T.
    """
    U_dag = torch.conj(U.mT)
    a, b = U_dag[..., 0, 0], U_dag[..., 0, 1]
    c, d = U_dag[..., 1, 0], U_dag[..., 1, 1]
    row0 = torch.stack([d, -b], dim=-1)
    row1 = torch.stack([-c, a], dim=-1)
    return torch.stack([row0, row1], dim=-2)

def rotate_spinor(U: torch.Tensor, X: torch.Tensor, pi: torch.Tensor):
    """
    Exact Lorentz-covariant SL(2,C) transformation on Hermitian matrices and spinors:
      X'^{AA'} = U^A_B X^{BB'} (U^dagger)_{B'}^{A'}
      pi'_{A'} = ((U^dagger)^{-1})_{A'}^{B'} pi_{B'}
      omega'^A = i X'^{AA'} pi'_{A'} == U^A_B omega^B  (Exact covariance!)
    """
    U_dag = torch.conj(U.mT)
    U_inv_dag = sl2c_inverse_dagger(U)
    X_rot = torch.einsum("...ab,...bc,...cd->...ad", U, X, U_dag)
    pi_rot = torch.einsum("...ab,...b->...a", U_inv_dag, pi)
    omega_rot = 1j * torch.einsum("...ab,...b->...a", X_rot, pi_rot)
    return X_rot, omega_rot, pi_rot

def epsilon_inner_product(xi: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
    """
    SL(2,C)-invariant symplectic spinor contraction: [xi, eta] = epsilon_{AB} xi^A eta^B
    """
    eps = EPSILON.to(dtype=xi.dtype, device=xi.device)
    return torch.einsum("...a,ab,...b->...", xi, eps, eta)