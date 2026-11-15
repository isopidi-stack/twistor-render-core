import torch

EPSILON = torch.tensor([[0.0, 1.0], [-1.0, 0.0]], dtype=torch.complex64)

def sl2c_from_lie_algebra(axis_angle: torch.Tensor, boost_rapidity: torch.Tensor = None) -> torch.Tensor:
    """
    Constructs U in SL(2,C) (or SU(2) when boost_rapidity is None) directly via the 
    Lie algebra exponential map exp(-i/2 * theta_k * sigma_k - 1/2 * eta_k * sigma_k),
    bypassing Euler angle charts and gimbal singularities.
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
    return torch.linalg.matrix_exp(M)

def rotate_spinor(U: torch.Tensor, X: torch.Tensor, pi: torch.Tensor):
    """
    Coordinate-free SL(2,C) / SU(2) transformation on Hermitian null-cones and spinors:
      X'^{AA'} = U^A_B X^{BB'} (U^dagger)_{B'}^{A'}
      pi'_{A'} = (U^*)_{A'}^{B'} pi_{B'}
      omega'^A = i X'^{AA'} pi'_{A'}
    """
    U_dag = torch.conj(U.mT)
    X_rot = torch.einsum("...ab,...bc,...cd->...ad", U, X, U_dag)
    pi_rot = torch.einsum("...ab,...b->...a", torch.conj(U), pi)
    omega_rot = 1j * torch.einsum("...ab,...b->...a", X_rot, pi_rot)
    return X_rot, omega_rot, pi_rot

def epsilon_inner_product(xi: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
    """
    SL(2,C)-invariant symplectic spinor contraction: [xi, eta] = epsilon_{AB} xi^A eta^B
    """
    eps = EPSILON.to(dtype=xi.dtype, device=xi.device)
    return torch.einsum("...a,ab,...b->...", xi, eps, eta)