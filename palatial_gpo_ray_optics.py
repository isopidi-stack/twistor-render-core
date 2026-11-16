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

    @staticmethod
    def _lift_dirs_to_pi_spinor(dirs: torch.Tensor) -> torch.Tensor:
        """Stereographic Hopf lift from unit 3D ray directions (N, 3) to pi_{A'} in C^{N x 2}."""
        dirs = dirs / torch.linalg.norm(dirs, dim=-1, keepdim=True).clamp(min=1e-12)
        nx, ny, nz = dirs[:, 0], dirs[:, 1], dirs[:, 2]
        p0 = torch.sqrt(((1.0 + nz) * 0.5).clamp(min=1e-14)).to(torch.complex128)
        p1 = ((nx + 1j * ny) / (2.0 * p0)).to(torch.complex128)
        pi_spinor = torch.stack([p0, p1], dim=-1)
        return pi_spinor / torch.linalg.norm(pi_spinor, dim=-1, keepdim=True).clamp(min=1e-14)

    @staticmethod
    def _hermitian_incidence_omega(points: torch.Tensor, pi_spinor: torch.Tensor) -> torch.Tensor:
        """Evaluates Penrose incidence omega^A = i X^{AA'} pi_{A'} using the 2x2
        Hermitian spatial matrix X^{AA'} without Z-buffering or Euclidean depth sorting.
        """
        x0, x1, x2 = points[:, 0], points[:, 1], points[:, 2]
        p0, p1 = pi_spinor[:, 0], pi_spinor[:, 1]
        w0 = 1j * (x2 * p0 + (x0 - 1j * x1) * p1)
        w1 = 1j * ((x0 + 1j * x1) * p0 - x2 * p1)
        return torch.stack([w0, w1], dim=-1)

    @classmethod
    def lift_rays_to_cp4(
        cls,
        ray_origins: torch.Tensor,
        ray_dirs: torch.Tensor,
        curvature_horizon: torch.Tensor = None,
    ) -> torch.Tensor:
        """Lifts 3D ray origins and unit directions (N, 3) into 5D homogeneous
        CP^4 twistor ray bundles Z^I = (omega^0, omega^1, pi_0', pi_1', phi_horizon).
        """
        origins = ray_origins.to(dtype=torch.float64)
        dirs = ray_dirs.to(dtype=torch.float64)
        device = dirs.device
        N = dirs.shape[0]

        pi_spinor = cls._lift_dirs_to_pi_spinor(dirs)
        omega_spinor = cls._hermitian_incidence_omega(origins, pi_spinor)

        omega_norm = torch.linalg.norm(omega_spinor, dim=-1, keepdim=True)
        omega_unit = torch.where(
            omega_norm > 1e-12,
            omega_spinor / omega_norm.clamp(min=1e-12),
            pi_spinor,
        )

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

    @classmethod
    def render_gpo_hud_frame(
        cls,
        height: int,
        width: int,
        field_centers: torch.Tensor,
        harmonic_weights: torch.Tensor = None,
        camera_origin: torch.Tensor = None,
        fov_deg: float = 60.0,
        sigma_aperture: float = 0.25,
    ) -> dict:
        """Primary Z-buffer-free CP^4 ray-bundle rendering pass producing a live
        (H, W, 3) GPO HUD frame directly from Penrose spinor incidence and
        harmonic frequency integration.

        Parameters
        ----------
        height, width : int
            Output HUD frame resolution (H, W).
        field_centers : torch.Tensor, shape (M, 3)
            3D spatial coordinates of active field primitives (evaluated via Hermitian matrices).
        harmonic_weights : torch.Tensor, optional, shape (M,)
            Complex or real harmonic frequency weights for each primitive.
        camera_origin : torch.Tensor, optional, shape (3,)
            3D camera origin (default: origin [0, 0, 0]).
        fov_deg : float
            Field of view in degrees.
        sigma_aperture : float
            Gaussian spinor incidence envelope width.
        """
        device = field_centers.device
        centers = field_centers.to(dtype=torch.float64, device=device)
        M = centers.shape[0]
        N = height * width

        if camera_origin is None:
            cam_orig = torch.zeros((1, 3), dtype=torch.float64, device=device)
        else:
            cam_orig = camera_origin.to(dtype=torch.float64, device=device).reshape(1, 3)

        # 1. Construct camera ray bundle directions without Euler/gimbal angles
        aspect = float(width) / float(max(1, height))
        scale = math.tan(math.radians(fov_deg * 0.5))
        ys = torch.linspace(scale, -scale, height, dtype=torch.float64, device=device)
        xs = torch.linspace(-scale * aspect, scale * aspect, width, dtype=torch.float64, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        dirs = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1), torch.ones(N, dtype=torch.float64, device=device)], dim=-1)
        dirs = dirs / torch.linalg.norm(dirs, dim=-1, keepdim=True).clamp(min=1e-12)

        # 2. Lift ray directions to projective spinors pi_{A'} in C^{N x 2}
        pi_spinor = cls._lift_dirs_to_pi_spinor(dirs)  # (N, 2)
        origins_expanded = cam_orig.expand(N, 3)
        omega_cam = cls._hermitian_incidence_omega(origins_expanded, pi_spinor)  # (N, 2)

        # 3. Evaluate spinor transverse incidence residual d_perp^2 = ||D||^2 - |pi^dag D|^2
        #    for all M field primitives without Z-buffer depth sorting
        p0 = pi_spinor[:, 0].unsqueeze(1)  # (N, 1)
        p1 = pi_spinor[:, 1].unsqueeze(1)  # (N, 1)
        cx = centers[:, 0].unsqueeze(0)     # (1, M)
        cy = centers[:, 1].unsqueeze(0)     # (1, M)
        cz = centers[:, 2].unsqueeze(0)     # (1, M)

        # Incidence moment of field centers: omega_center^A = i X_center^{AA'} pi_{A'}
        wc0 = 1j * (cz * p0 + (cx - 1j * cy) * p1)  # (N, M)
        wc1 = 1j * ((cx + 1j * cy) * p0 - cz * p1)  # (N, M)

        D0 = omega_cam[:, 0].unsqueeze(1) - wc0      # (N, M)
        D1 = omega_cam[:, 1].unsqueeze(1) - wc1      # (N, M)

        norm_D_sq = torch.real(D0 * D0.conj() + D1 * D1.conj())
        pi_dot_D = p0.conj() * D0 + p1.conj() * D1
        d_perp_sq = (norm_D_sq - torch.real(pi_dot_D * pi_dot_D.conj())).clamp(min=0.0)

        # 4. Spinor harmonic frequency modulation Y_{1,m}(pi) across ray bundle
        y10 = torch.real(p0 * p0.conj() - p1 * p1.conj())  # (N, 1)
        y11 = torch.real(p0 * p1.conj())                   # (N, 1)
        harmonic_carrier = 1.0 + 0.25 * y10 + 0.25 * y11

        if harmonic_weights is None:
            h_weights = torch.ones((1, M), dtype=torch.float64, device=device)
        else:
            h_weights = torch.abs(harmonic_weights).to(dtype=torch.float64, device=device).reshape(1, M)

        gaussian_incidence = torch.exp(-d_perp_sq / (2.0 * (sigma_aperture ** 2))) * h_weights * harmonic_carrier
        field_intensity = torch.sum(gaussian_incidence, dim=-1)  # (N,)

        # 5. Compute 2D screen-space wavefront curvature / occlusion horizon Z^4
        intensity_2d = field_intensity.reshape(height, width)
        lap_2d = torch.zeros_like(intensity_2d)
        if height >= 3 and width >= 3:
            lap_2d[1:-1, 1:-1] = (
                0.25 * (
                    intensity_2d[:-2, 1:-1]
                    + intensity_2d[2:, 1:-1]
                    + intensity_2d[1:-1, :-2]
                    + intensity_2d[1:-1, 2:]
                )
                - intensity_2d[1:-1, 1:-1]
            ).abs()
        curvature_horizon = (1.5 * lap_2d.reshape(N, 1)).to(dtype=torch.complex128)

        # 6. Modulate omega_spinor phase by local incidence shear to excite Purple chiral band on edges
        shear_phase = torch.exp(1j * (0.65 * field_intensity).unsqueeze(-1).to(dtype=torch.complex128))
        omega_active = pi_spinor * shear_phase
        Z5 = torch.cat([omega_active, pi_spinor, curvature_horizon], dim=-1)
        Z5_hat = Z5 / torch.linalg.norm(Z5, dim=-1, keepdim=True).clamp(min=1e-14)

        gpo = cls.evaluate_robot_vision_gpo_rgb(Z5_hat)
        hud_frame = gpo["rgb_channels"].reshape(height, width, 3)

        return {
            "hud_frame": hud_frame,
            "field_intensity": intensity_2d,
            "c_green": gpo["c_green"].reshape(height, width),
            "c_purple": gpo["c_purple"].reshape(height, width),
            "c_orange": gpo["c_orange"].reshape(height, width),
            "triad_sum": gpo["triad_sum"].reshape(height, width),
            "Z5_hat": Z5_hat.reshape(height, width, 5),
        }

