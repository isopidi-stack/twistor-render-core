import math
import torch


class PalatialGPORayOptics:
    """Unifies twistor-render-core ray bundles with Palatial CP^4 Standing Waves.

    Strictly abides by Penrose incidence geometry, Hermitian matrix ray parameters,
    and harmonic frequency integration without Z-buffering or gimbals.
    """

    SIGMA_5 = torch.tensor(
        [
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )

    @staticmethod
    def build_gpo_projectors(device=None):
        device = device or torch.device("cpu")
        dtype = torch.complex128
        P_G = torch.zeros((5, 5), dtype=dtype, device=device)
        P_P = torch.zeros((5, 5), dtype=dtype, device=device)
        for a in (0, 1):
            vg = torch.zeros((5, 1), dtype=dtype, device=device)
            vg[a, 0] = 1.0 / math.sqrt(2.0)
            vg[a + 2, 0] = 1.0 / math.sqrt(2.0)
            P_G = P_G + (vg @ vg.conj().T)

            vp = torch.zeros((5, 1), dtype=dtype, device=device)
            vp[a, 0] = 1.0 / math.sqrt(2.0)
            vp[a + 2, 0] = -1.0 / math.sqrt(2.0)
            P_P = P_P + (vp @ vp.conj().T)

        P_O = torch.zeros((5, 5), dtype=dtype, device=device)
        P_O[4, 4] = 1.0 + 0.0j
        return P_G, P_P, P_O

    @classmethod
    def lift_rays_to_cp4(cls, ray_origins: torch.Tensor, ray_dirs: torch.Tensor, curvature_horizon: torch.Tensor = None) -> torch.Tensor:
        """Lifts 3D ray origins and unit directions (N, 3) into 5D homogeneous
        CP^4 twistor ray bundles Z^I = (omega^0, omega^1, pi_0', pi_1', phi_horizon).
        """
        origins = ray_origins.to(dtype=torch.float64)
        dirs = ray_dirs.to(dtype=torch.float64)
        dirs = dirs / torch.linalg.norm(dirs, dim=-1, keepdim=True).clamp(min=1e-12)
        device = dirs.device
        N = dirs.shape[0]

        nx, ny, nz = dirs[:, 0], dirs[:, 1], dirs[:, 2]
        p0 = torch.sqrt(((1.0 + nz) * 0.5).clamp(min=1e-14)).to(torch.complex128)
        p1 = ((nx + 1j * ny) / (2.0 * p0)).to(torch.complex128)
        pi_spinor = torch.stack([p0, p1], dim=-1)

        # Penrose incidence relation: omega^A = i x^{AA'} pi_{A'} via Hermitian matrix ray parameter
        x0, x1, x2 = origins[:, 0], origins[:, 1], origins[:, 2]
        w0 = 1j * (x2 * p0 + (x0 - 1j * x1) * p1)
        w1 = 1j * ((x0 + 1j * x1) * p0 - x2 * p1)
        omega_spinor = torch.stack([w0, w1], dim=-1)

        omega_norm = torch.linalg.norm(omega_spinor, dim=-1, keepdim=True)
        omega_unit = torch.where(omega_norm > 1e-12, omega_spinor / omega_norm.clamp(min=1e-12), pi_spinor)

        if curvature_horizon is None:
            z4 = torch.zeros((N, 1), dtype=torch.complex128, device=device)
        else:
            z4 = curvature_horizon.to(dtype=torch.complex128, device=device).reshape(N, 1)

        Z5 = torch.cat([omega_unit, pi_spinor, z4], dim=-1)
        return Z5 / torch.linalg.norm(Z5, dim=-1, keepdim=True).clamp(min=1e-14)

    @classmethod
    def evaluate_robot_vision_gpo_rgb(cls, Z5_hat: torch.Tensor) -> dict:
        """Evaluates Hermitian ray density operators and maps GPO standing-wave
        barycentric coordinates to native RGB channels for robot vision.
        """
        Z5_hat = Z5_hat / torch.linalg.norm(Z5_hat, dim=-1, keepdim=True).clamp(min=1e-14)
        P_G, P_P, P_O = cls.build_gpo_projectors(device=Z5_hat.device)

        rho = torch.einsum("ni,nj->nij", Z5_hat, Z5_hat.conj())

        c_G = torch.real(torch.einsum("nij,ji->n", rho, P_G))
        c_P = torch.real(torch.einsum("nij,ji->n", rho, P_P))
        c_O = torch.real(torch.einsum("nij,ji->n", rho, P_O))

        rgb = torch.stack(
            [
                (c_O + 0.5 * c_P).clamp(0.0, 1.0),
                c_G.clamp(0.0, 1.0),
                c_P.clamp(0.0, 1.0),
            ],
            dim=-1,
        )

        return {
            "c_green": c_G,
            "c_purple": c_P,
            "c_orange": c_O,
            "triad_sum": c_G + c_P + c_O,
            "rgb_channels": rgb,
        }
