import math
import torch
from twistor.incidence import sl2c_from_lie_algebra, sl2c_inverse_dagger
from twistor.penrose_core import PenroseTwistorCore
from twistor.harmonics import PenroseTwistorHarmonicEngine

class TwistorRenderer:
    """
    Implements the full 15-parameter Conformal Twistor Group SU(2,2) action and
    point-to-ray rendering with an explicit Euclidean baseline comparison:
      - 3 Spatial Rotations: theta_vec = (theta_x, theta_y, theta_z)
      - 3 Lorentz Boosts:    boost_vec = (eta_x, eta_y, eta_z)
      - 4 Translations:      trans_4d  = (T_t, T_x, T_y, T_z)
      - 4 Special Conformal: b_4d      = (b_t, b_x, b_y, b_z) via pi' = pi - i B omega
      - 1 Conformal Dilation: dilation_lambda via diag(e^{lambda/2} I_2, e^{-lambda/2} I_2)
    Total = 15 independent real parameters preserving G^dagger Sigma G = Sigma.
    """
    def __init__(self, width: int = 64, height: int = 64, fov_deg: float = 60.0, device=None):
        self.device = device or torch.device("cpu")
        self.width = width
        self.height = height
        self.fov_deg = fov_deg

    @staticmethod
    def build_dilation_su22_matrix(dilation_lambda: float, dtype=torch.complex128, device=None) -> torch.Tensor:
        """
        Constructs the 1-parameter conformal dilation matrix D(lambda) in SU(2,2):
            D(lambda) = [[ e^{lambda/2} I_2,  0                 ],
                         [ 0,                 e^{-lambda/2} I_2 ]]
        Satisfies D(lambda)^dagger Sigma D(lambda) = Sigma identically and scales X' = e^{lambda} X.
        """
        dev = device or torch.device("cpu")
        lam_t = torch.as_tensor(dilation_lambda, dtype=torch.float64, device=dev)
        s_pos = torch.exp(0.5 * lam_t).to(dtype)
        s_neg = torch.exp(-0.5 * lam_t).to(dtype)
        I2 = torch.eye(2, dtype=dtype, device=dev)
        Z2 = torch.zeros((2, 2), dtype=dtype, device=dev)
        top = torch.cat([s_pos * I2, Z2], dim=-1)
        bot = torch.cat([Z2, s_neg * I2], dim=-1)
        return torch.cat([top, bot], dim=-2)

    @staticmethod
    def build_full_15param_su22_matrix(
        theta_3d: torch.Tensor,
        boost_3d: torch.Tensor,
        trans_4d: torch.Tensor,
        special_conformal_4d: torch.Tensor,
        dilation_lambda: float = 0.0,
    ) -> torch.Tensor:
        """
        Composes all 15 real parameters of the conformal group into a single 4x4 matrix G in SU(2,2):
            G = K(b_4d) @ D(dilation_lambda) @ P(Lambda(theta_3d, boost_3d), trans_4d)
        Satisfies G^dagger Sigma G = Sigma identically.
        """
        Lambda = sl2c_from_lie_algebra(theta_3d, boost_3d)
        P = PenroseTwistorCore.build_poincare_su22_matrix(Lambda, trans_4d)
        K = PenroseTwistorCore.build_special_conformal_su22_matrix(special_conformal_4d)
        D = TwistorRenderer.build_dilation_su22_matrix(dilation_lambda, dtype=P.dtype, device=P.device)
        return K @ D @ P

    @staticmethod
    def lift_points_to_null_cone_twistors(pts_3d: torch.Tensor, camera_center: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Lifts 3D world points X_i relative to camera_center C onto the 4D future null cone:
            dx^a = (||X_i - C||_2, X_i - C)  =>  det(dX^{AA'}) == 0 identically.
        Also computes each point's celestial spinor pi_{A'} and incidence twistor Z^alpha = (i dX pi, pi).
        """
        rel_3d = pts_3d - camera_center.unsqueeze(0)
        dist = torch.norm(rel_3d, dim=-1, keepdim=True)
        dx_4d = torch.cat([dist, rel_3d], dim=-1)  # (N, 4) with t = ||x||
        dX_herm = PenroseTwistorCore.spacetime_to_hermitian(dx_4d)  # (N, 2, 2), det == 0
        pi = PenroseTwistorHarmonicEngine.points_to_celestial_spinors(pts_3d, camera_center=camera_center)
        omega = 1j * torch.einsum("nab,nb->na", dX_herm, pi)
        Z = torch.cat([omega, pi], dim=-1)  # (N, 4)
        return dX_herm, Z

    @staticmethod
    def spatial_traceless_hermitian(d_vec: torch.Tensor) -> torch.Tensor:
        """
        Maps a 3D spatial displacement d = (dx, dy, dz) to its 2x2 traceless Hermitian matrix D = d_k sigma^k.
        For any normalized spinor pi with Hopf unit ray n_k = <pi | sigma_k | pi>,
        the perpendicular spinor residual ||D pi||^2 - |<pi | D | pi>|^2 equals ||d||^2 - (d . n)^2.
        """
        dx, dy, dz = d_vec[..., 0], d_vec[..., 1], d_vec[..., 2]
        c_dtype = torch.complex128 if d_vec.dtype == torch.float64 else torch.complex64
        zero = torch.zeros_like(dx)
        D00 = torch.complex(dz, zero)
        D01 = torch.complex(dx, -dy)
        D10 = torch.complex(dx,  dy)
        D11 = torch.complex(-dz, zero)
        return torch.stack([
            torch.stack([D00, D01], dim=-1),
            torch.stack([D10, D11], dim=-1)
        ], dim=-2).to(c_dtype)

    @staticmethod
    def spinor_perpendicular_residual_sq(pts_3d: torch.Tensor, camera_center: torch.Tensor, ray_spinors: torch.Tensor) -> torch.Tensor:
        """
        Computes the perpendicular point-to-ray squared distance via spinor Pauli algebra:
            D = (X_i - C)_k sigma^k
            r_perp^2 = <pi | D^2 | pi> - <pi | D | pi>^2 = ||D pi||^2 - |pi^dagger D pi|^2
        Returns shape (M_rays, N_points).
        """
        d_vec = pts_3d - camera_center.unsqueeze(0)  # (N, 3)
        D_mat = TwistorRenderer.spatial_traceless_hermitian(d_vec)  # (N, 2, 2)

        # Normalize ray spinors pi: (M, 2)
        pi_norm = ray_spinors / torch.linalg.norm(ray_spinors, dim=-1, keepdim=True).clamp(min=1e-12)

        # Evaluate D_n @ pi_m -> shape (M, N, 2)
        D_pi = torch.einsum("nab,mb->mna", D_mat, pi_norm)
        norm_D_pi_sq = torch.sum(torch.abs(D_pi) ** 2, dim=-1).real  # (M, N)

        # Parallel projection <pi | D | pi> = d . n -> shape (M, N)
        proj_par = torch.einsum("ma,mna->mn", torch.conj(pi_norm), D_pi).real
        return torch.clamp(norm_D_pi_sq - proj_par ** 2, min=0.0)

    @staticmethod
    def euclidean_perpendicular_residual_sq(pts_3d: torch.Tensor, camera_center: torch.Tensor, ray_dirs: torch.Tensor) -> torch.Tensor:
        """
        Standard Euclidean point-to-ray squared distance baseline:
            r_perp^2 = ||d_i||^2 - (d_i . n_m)^2,  where d_i = X_i - C and ||n_m|| = 1.
        Returns shape (M_rays, N_points).
        """
        d_vec = pts_3d - camera_center.unsqueeze(0)  # (N, 3)
        d_norm_sq = torch.sum(d_vec ** 2, dim=-1).unsqueeze(0)  # (1, N)
        n_unit = ray_dirs / torch.linalg.norm(ray_dirs, dim=-1, keepdim=True).clamp(min=1e-12)  # (M, 3)
        dot_dn = torch.einsum("mc,nc->mn", n_unit, d_vec)  # (M, N)
        return torch.clamp(d_norm_sq - dot_dn ** 2, min=0.0)

    def generate_camera_rays(
        self,
        theta_3d: torch.Tensor = None,
        boost_3d: torch.Tensor = None,
        special_conformal_4d: torch.Tensor = None,
        dilation_lambda: float = 0.0,
        camera_center: torch.Tensor = None,
        dtype=torch.float64,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generates pixel ray spinors pi'_{A'} and unit directions n_k transformed by the
        full 15-parameter SU(2,2) conformal group (including 3-axis boost, dilation, and K(b)).
        """
        aspect = float(self.width) / float(self.height)
        fov_rad = math.radians(self.fov_deg)
        half_tan = math.tan(0.5 * fov_rad)

        ys = torch.linspace(-half_tan, half_tan, self.height, dtype=dtype, device=self.device)
        xs = torch.linspace(-half_tan * aspect, half_tan * aspect, self.width, dtype=dtype, device=self.device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        zz = torch.ones_like(xx)
        dirs_local = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
        dirs_local = dirs_local / torch.linalg.norm(dirs_local, dim=-1, keepdim=True)

        pi = PenroseTwistorHarmonicEngine.points_to_celestial_spinors(dirs_local)  # (M, 2)

        # Construct initial incidence spinor omega^A = i X_cam^{AA'} pi_{A'}
        if camera_center is not None:
            cam_4d = torch.cat([torch.norm(camera_center).unsqueeze(0), camera_center], dim=0)
            X_cam = PenroseTwistorCore.spacetime_to_hermitian(cam_4d)
            omega = 1j * torch.einsum("ab,mb->ma", X_cam, pi)
        else:
            omega = torch.zeros_like(pi)

        Z = torch.cat([omega, pi], dim=-1)  # (M, 4)

        # Apply full 15-parameter SU(2,2) transformation if any group parameters are active
        t3 = theta_3d if theta_3d is not None else torch.zeros(3, dtype=dtype, device=self.device)
        b3 = boost_3d if boost_3d is not None else torch.zeros(3, dtype=dtype, device=self.device)
        tr4 = torch.zeros(4, dtype=dtype, device=self.device)
        sc4 = special_conformal_4d if special_conformal_4d is not None else torch.zeros(4, dtype=dtype, device=self.device)

        G = self.build_full_15param_su22_matrix(t3, b3, tr4, sc4, dilation_lambda=dilation_lambda)
        Z_transformed = torch.einsum("ab,mb->ma", G, Z)
        pi_transformed = Z_transformed[:, 2:4]
        pi_transformed = pi_transformed / torch.linalg.norm(pi_transformed, dim=-1, keepdim=True).clamp(min=1e-12)
        ray_dirs = PenroseTwistorHarmonicEngine.spinor_to_unit_ray(pi_transformed)
        return pi_transformed, ray_dirs

    def render(
        self,
        pts_3d: torch.Tensor,
        camera_center: torch.Tensor,
        ray_dirs: torch.Tensor,
        sigma: float = 0.15,
    ) -> torch.Tensor:
        """
        Evaluates Gaussian splat intensity along forward-facing rays (d . n > 0):
            I(ray_m) = sum_{i: d_i . n_m > 0} exp(-r_perp^2(m, i) / (2 * sigma^2)) / (1 + ||d_i||^2)
        """
        d_vec = pts_3d - camera_center.unsqueeze(0)  # (N, 3)
        dist_sq = torch.sum(d_vec ** 2, dim=-1).unsqueeze(0)  # (1, N)
        n_unit = ray_dirs / torch.linalg.norm(ray_dirs, dim=-1, keepdim=True).clamp(min=1e-12)
        dot_dn = torch.einsum("mc,nc->mn", n_unit, d_vec)  # (M, N)

        forward_mask = dot_dn > 0.0
        r_perp_sq = torch.clamp(dist_sq - dot_dn ** 2, min=0.0)
        weights = torch.exp(-r_perp_sq / (2.0 * (sigma ** 2))) / (1.0 + dist_sq)
        weights = torch.where(forward_mask, weights, torch.zeros_like(weights))
        return torch.sum(weights, dim=-1)