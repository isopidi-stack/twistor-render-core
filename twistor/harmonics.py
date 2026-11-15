import math
import torch

class PenroseTwistorHarmonicEngine:
    """
    Genuine Conformal Harmonic Engine on the Celestial Spinor Sphere CP^1 ~= S^2.
    
    Mathematical Foundation:
      - Hilbert Space: H = L^2(CP^1, d_mu), d_mu = sin(theta) d_theta d_phi
      - Operator: H_op = -Delta_{CP^1} (SU(2) Casimir / Laplace-Beltrami operator)
      - Eigenvalue Equation: -Delta_{CP^1} Y_{l,m}(pi) = l(l+1) Y_{l,m}(pi)
      - Conformal Energy Functional: E[psi] = <psi | (-Delta_{CP^1} + 1/4) | psi>
      - Conformal Eigen-Frequencies: omega_l = sqrt(l(l+1) + 1/4) = l + 1/2
    """
    def __init__(self, max_l: int = 2, device=None):
        self.device = device or torch.device("cpu")
        self.max_l = max_l

    @staticmethod
    def points_to_celestial_spinors(pts_world: torch.Tensor, camera_center: torch.Tensor = None) -> torch.Tensor:
        """
        Maps 3D world points X_i relative to camera_center C onto normalized projective 
        spinors pi_{A'} = (p_0, p_1)^T on CP^1. Invariant to translation (X_i, C) -> (X_i + T, C + T)
        and radial depth scaling X_i - C -> lambda_i (X_i - C).
        """
        if camera_center is not None:
            rel = pts_world - camera_center.view(*([1] * (pts_world.ndim - 1)), 3)
        else:
            rel = pts_world

        r = torch.norm(rel, dim=-1, keepdim=True).clamp(min=1e-12)
        n = rel / r
        nx, ny, nz = n[..., 0], n[..., 1], n[..., 2]

        # Stereographic projection from South Pole (nz = -1) to CP^1 spinor
        # Handle near-south-pole numerical stability cleanly
        denom = torch.clamp(1.0 + nz, min=1e-12)
        p0_mag = torch.sqrt(0.5 * denom)
        p0 = torch.complex(p0_mag, torch.zeros_like(p0_mag))
        p1 = torch.complex(nx, ny) / (2.0 * p0_mag)

        # Exact south-pole limit (nz -> -1 => p0 = 0, p1 = 1)
        south_mask = (1.0 + nz) < 1e-10
        if torch.any(south_mask):
            p0 = torch.where(south_mask, torch.zeros_like(p0), p0)
            p1 = torch.where(south_mask, torch.ones_like(p1), p1)

        return torch.stack([p0, p1], dim=-1)

    @staticmethod
    def spinor_to_unit_ray(pi: torch.Tensor) -> torch.Tensor:
        """
        Maps normalized spinor pi = (p_0, p_1) to unit ray n in S^2 via Pauli contraction:
        n_k = <pi | sigma_k | pi> / <pi | pi>.
        """
        norm_sq = torch.sum(torch.abs(pi)**2, dim=-1, keepdim=True).clamp(min=1e-12)
        p0, p1 = pi[..., 0], pi[..., 1]
        nx = 2.0 * (p0 * torch.conj(p1)).real
        ny = 2.0 * (torch.conj(p0) * p1).imag
        nz = (torch.abs(p0)**2 - torch.abs(p1)**2)
        return torch.stack([nx, ny, nz], dim=-1) / norm_sq

    def evaluate_complex_eigenmodes(self, pi: torch.Tensor):
        """
        Evaluates exact analytical complex eigenfunctions Y_{l,m}(pi) of -Delta_{CP^1}
        for l = 0 (monopole, lambda=0) and l = 1 (dipole, lambda=2) in Condon-Shortley convention.
        """
        n = self.spinor_to_unit_ray(pi)
        nx, ny, nz = n[..., 0], n[..., 1], n[..., 2]

        # l = 0 Monopole Ground State (lambda_0 = 0)
        Y_0_0 = torch.full_like(nz, 1.0 / math.sqrt(4.0 * math.pi), dtype=torch.complex64 if pi.dtype == torch.complex64 else torch.complex128)

        # l = 1 Dipole Eigenmodes (lambda_1 = 2)
        c10 = math.sqrt(3.0 / (4.0 * math.pi))
        c11 = math.sqrt(3.0 / (8.0 * math.pi))
        Y_1_m1 =  c11 * torch.complex(nx, -ny)
        Y_1_0  =  c10 * torch.complex(nz, torch.zeros_like(nz))
        Y_1_p1 = -c11 * torch.complex(nx,  ny)

        return {
            (0, 0):  Y_0_0,
            (1, -1): Y_1_m1,
            (1, 0):  Y_1_0,
            (1, 1):  Y_1_p1
        }

    def evaluate_real_basis(self, pi: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the 9-component real orthonormal spherical harmonic basis Y_{l,m}^R(pi)
        for l = 0 (1 mode), l = 1 (3 modes), l = 2 (5 modes) directly from spinor bilinears.
        Returns shape (..., 9).
        """
        n = self.spinor_to_unit_ray(pi)
        nx, ny, nz = n[..., 0], n[..., 1], n[..., 2]

        # l = 0 (lambda = 0, omega = 0.5)
        y00 = torch.full_like(nx, 0.5 * math.sqrt(1.0 / math.pi))

        # l = 1 (lambda = 2, omega = 1.5)
        k1 = math.sqrt(3.0 / (4.0 * math.pi))
        y1_m1 = k1 * ny
        y1_0  = k1 * nz
        y1_p1 = k1 * nx

        # l = 2 (lambda = 6, omega = 2.5)
        k2_a = 0.5 * math.sqrt(15.0 / math.pi)
        k2_b = 0.25 * math.sqrt(5.0 / math.pi)
        k2_c = 0.25 * math.sqrt(15.0 / math.pi)
        y2_m2 = k2_a * nx * ny
        y2_m1 = k2_a * ny * nz
        y2_0  = k2_b * (3.0 * nz**2 - 1.0)
        y2_p1 = k2_a * nx * nz
        y2_p2 = k2_c * (nx**2 - ny**2)

        return torch.stack([y00, y1_m1, y1_0, y1_p1, y2_m2, y2_m1, y2_0, y2_p1, y2_p2], dim=-1)

    def compute_conformal_energy_functional(self, pi: torch.Tensor, field_values: torch.Tensor, weights: torch.Tensor = None):
        """
        Projects a scalar or multi-channel field psi(pi) onto the L^2(CP^1) eigenbasis,
        extracts the spectral coefficients c_{l,m}, and evaluates the genuine Conformal
        Casimir Energy Functional:
            E_dirichlet = sum_{l,m} l(l+1) |c_{l,m}|^2
            E_conformal = sum_{l,m} (l(l+1) + 1/4) |c_{l,m}|^2
        """
        B = self.evaluate_real_basis(pi)  # (N, 9)
        if field_values.ndim == 1:
            f = field_values.unsqueeze(-1)
        else:
            f = field_values

        # Solve least-squares spectral projection on CP^1 (robust to non-uniform point sampling on robot cameras!)
        if weights is not None:
            w_sqrt = torch.sqrt(weights).unsqueeze(-1)
            coeffs = torch.linalg.lstsq(B * w_sqrt, f * w_sqrt).solution  # (9, C)
        else:
            coeffs = torch.linalg.lstsq(B, f).solution  # (9, C)

        # Eigenvalues lambda_l = l(l+1) for the 9 modes: [0, 2, 2, 2, 6, 6, 6, 6, 6]
        eigenvalues = torch.tensor([0.0, 2.0, 2.0, 2.0, 6.0, 6.0, 6.0, 6.0, 6.0], dtype=B.dtype, device=B.device).unsqueeze(-1)
        conformal_eigenvalues = eigenvalues + 0.25  # (l + 1/2)^2

        power_per_mode = coeffs**2
        total_norm_sq = torch.sum(power_per_mode, dim=0).clamp(min=1e-12)
        dirichlet_energy = torch.sum(eigenvalues * power_per_mode, dim=0)
        conformal_energy = torch.sum(conformal_eigenvalues * power_per_mode, dim=0)

        # Effective Rayleigh quotient eigenvalue: <psi | -Delta | psi> / <psi | psi>
        rayleigh_lambda = dirichlet_energy / total_norm_sq
        effective_omega = torch.sqrt(conformal_energy / total_norm_sq)

        # Reconstruct smooth l <= 2 harmonic field and local point-wise Dirichlet residual
        reconstructed = B @ coeffs
        monopole_ground_state = B[:, 0:1] @ coeffs[0:1, :]
        dipole_perturbation = B[:, 1:4] @ coeffs[1:4, :]

        return {
            "coefficients": coeffs,
            "dirichlet_energy": dirichlet_energy,
            "conformal_energy": conformal_energy,
            "rayleigh_eigenvalue": rayleigh_lambda,
            "effective_frequency": effective_omega,
            "monopole_ground_state": monopole_ground_state.squeeze(-1),
            "dipole_state": dipole_perturbation.squeeze(-1),
            "reconstructed_field": reconstructed.squeeze(-1)
        }