import math
import torch
from twistor.incidence import sl2c_from_lie_algebra
from twistor.penrose_core import PenroseTwistorCore
from twistor.harmonics import PenroseTwistorHarmonicEngine

class TwistorRenderer:
    """
    Implements the full 15-parameter Conformal Twistor Group SU(2,2) action via exact
    fractional linear transformations on Hermitian spacetime matrices X' = (AX - iB)(iCX + D)^{-1}
    and consistent t = 0 hyperplane ray-origin reconstruction.
    """
    def __init__(self, width: int = 64, height: int = 64, fov_deg: float = 60.0, device=None):
        self.device = device or torch.device("cpu")
        self.width = width
        self.height = height
        self.fov_deg = fov_deg

    @staticmethod
    def build_dilation_su22_matrix(dilation_lambda: float, dtype=torch.complex128, device=None) -> torch.Tensor:
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
        Lambda = sl2c_from_lie_algebra(theta_3d, boost_3d)
        P = PenroseTwistorCore.build_poincare_su22_matrix(Lambda, trans_4d)
        K = PenroseTwistorCore.build_special_conformal_su22_matrix(special_conformal_4d)
        D = TwistorRenderer.build_dilation_su22_matrix(dilation_lambda, dtype=P.dtype, device=P.device)
        return K @ D @ P

    @staticmethod
    def transform_spacetime_event_su22(G: torch.Tensor, X_cam: torch.Tensor, min_sv_tol: float = 1e-10) -> torch.Tensor:
        """
        Evaluates exact conformal fractional linear transformation:
            X' = (A X - iB) (iC X + D)^{-1}
        Checks denominator conditioning via minimum singular value to guard against
        conformal infinity singularities det(iC X + D) == 0.
        """
        A = G[0:2, 0:2]
        B = G[0:2, 2:4]
        C = G[2:4, 0:2]
        D = G[2:4, 2:4]

        if X_cam.ndim == 2:
            num = A @ X_cam - 1j * B
            den = 1j * C @ X_cam + D
        else:
            num = torch.einsum("ab,mbc->mac", A, X_cam) - 1j * B.unsqueeze(0)
            den = 1j * torch.einsum("ab,mbc->mac", C, X_cam) + D.unsqueeze(0)

        sv = torch.linalg.svdvals(den)
        if torch.any(sv[..., -1] < min_sv_tol):
            raise ValueError("Singular conformal transformation: denominator (iC X + D) is ill-conditioned or non-invertible.")

        return num @ torch.linalg.inv(den)

    @staticmethod
    def reconstruct_ray_origins_from_twistors(Z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reconstructs the unique t = 0 hyperplane intersection X_rec of each null twistor line Z = (omega, pi).
        Because X_rec = XP + PX - (pi^dagger X pi)I has trace(X_rec) == 0 identically, X_rec represents
        the spatial event on the t = 0 slice: o_ray = r' - t' n', satisfying omega = i X_rec pi exactly.
        """
        omega = Z[:, 0:2]
        pi = Z[:, 2:4]

        pi_norm = torch.linalg.norm(pi, dim=-1, keepdim=True).clamp(min=1e-12)
        pi_unit = pi / pi_norm
        omega_unit = omega / pi_norm
        Z_norm = torch.cat([omega_unit, pi_unit], dim=-1)

        pi_dag_omega = torch.sum(torch.conj(pi_unit) * omega_unit, dim=-1, keepdim=True)
        tau = -pi_dag_omega.imag

        term1 = torch.einsum("mi,mj->mij", omega_unit, torch.conj(pi_unit))
        term2 = torch.einsum("mi,mj->mij", pi_unit, torch.conj(omega_unit))

        X_herm = -1j * (term1 - term2)
        I2 = torch.eye(2, dtype=X_herm.dtype, device=X_herm.device).unsqueeze(0)
        X_herm = 0.5 * (X_herm + torch.conj(X_herm.mT)) + tau.unsqueeze(-1) * I2

        X00 = X_herm[:, 0, 0]
        X01 = X_herm[:, 0, 1]
        X10 = X_herm[:, 1, 0]
        X11 = X_herm[:, 1, 1]

        z_val = 0.5 * (X00 - X11).real
        x_val = 0.5 * (X01 + X10).real
        y_val = 0.5 * (X10 - X01).imag

        ray_origins_3d = torch.stack([x_val, y_val, z_val], dim=-1)
        return ray_origins_3d, X_herm, Z_norm

    @staticmethod
    def lift_points_to_null_cone_twistors(pts_3d: torch.Tensor, camera_center: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rel_3d = pts_3d - camera_center.unsqueeze(0)
        dist = torch.norm(rel_3d, dim=-1, keepdim=True)
        dx_4d = torch.cat([dist, rel_3d], dim=-1)
        dX_herm = PenroseTwistorCore.spacetime_to_hermitian(dx_4d)
        pi = PenroseTwistorHarmonicEngine.points_to_celestial_spinors(pts_3d, camera_center=camera_center)
        omega = 1j * torch.einsum("nab,nb->na", dX_herm, pi)
        Z = torch.cat([omega, pi], dim=-1)
        return dX_herm, Z

    @staticmethod
    def spatial_traceless_hermitian(d_vec: torch.Tensor) -> torch.Tensor:
        dx = d_vec[..., 0].real if d_vec.is_complex() else d_vec[..., 0]
        dy = d_vec[..., 1].real if d_vec.is_complex() else d_vec[..., 1]
        dz = d_vec[..., 2].real if d_vec.is_complex() else d_vec[..., 2]

        c_dtype = torch.complex128 if d_vec.dtype in [torch.float64, torch.complex128] else torch.complex64
        zero = torch.zeros_like(dx, dtype=dx.dtype)
        D00 = torch.complex(dz, zero).to(c_dtype)
        D01 = torch.complex(dx, -dy).to(c_dtype)
        D10 = torch.complex(dx,  dy).to(c_dtype)
        D11 = torch.complex(-dz, zero).to(c_dtype)
        return torch.stack([
            torch.stack([D00, D01], dim=-1),
            torch.stack([D10, D11], dim=-1)
        ], dim=-2)

    @staticmethod
    def spinor_perpendicular_residual_sq(
        pts_3d: torch.Tensor,
        camera_center: torch.Tensor,
        ray_spinors: torch.Tensor,
        return_parallel: bool = False,
    ):
        pi_norm = ray_spinors / torch.linalg.norm(ray_spinors, dim=-1, keepdim=True).clamp(min=1e-12)
        if camera_center.ndim == 1:
            d_vec = pts_3d - camera_center.unsqueeze(0)
            D_mat = TwistorRenderer.spatial_traceless_hermitian(d_vec)
            D_pi = torch.einsum("nab,mb->mna", D_mat, pi_norm)
        else:
            d_vec = pts_3d.unsqueeze(0) - camera_center.unsqueeze(1)
            D_mat = TwistorRenderer.spatial_traceless_hermitian(d_vec)
            D_pi = torch.einsum("mnab,mb->mna", D_mat, pi_norm)

        norm_D_pi_sq = torch.sum(torch.abs(D_pi) ** 2, dim=-1).real
        proj_par = torch.einsum("ma,mna->mn", torch.conj(pi_norm), D_pi).real
        r_perp_sq = torch.clamp(norm_D_pi_sq - proj_par ** 2, min=0.0)
        if return_parallel:
            return r_perp_sq, proj_par, norm_D_pi_sq
        return r_perp_sq

    @staticmethod
    def euclidean_perpendicular_residual_sq(pts_3d: torch.Tensor, camera_center: torch.Tensor, ray_dirs: torch.Tensor) -> torch.Tensor:
        n_unit = ray_dirs / torch.linalg.norm(ray_dirs, dim=-1, keepdim=True).clamp(min=1e-12)
        if camera_center.ndim == 1:
            d_vec = pts_3d - camera_center.unsqueeze(0)
            d_norm_sq = torch.sum(d_vec ** 2, dim=-1).unsqueeze(0)
            dot_dn = torch.einsum("mc,nc->mn", n_unit, d_vec)
        else:
            d_vec = pts_3d.unsqueeze(0) - camera_center.unsqueeze(1)
            d_norm_sq = torch.sum(d_vec ** 2, dim=-1)
            dot_dn = torch.einsum("mc,mnc->mn", n_unit, d_vec)
        return torch.clamp(d_norm_sq - dot_dn ** 2, min=0.0)

    def generate_camera_rays(
        self,
        theta_3d: torch.Tensor = None,
        boost_3d: torch.Tensor = None,
        special_conformal_4d: torch.Tensor = None,
        dilation_lambda: float = 0.0,
        camera_center: torch.Tensor = None,
        dtype=torch.float64,
        trans_4d: torch.Tensor = None,
        return_ray_origins: bool = False,
    ):
        aspect = float(self.width) / float(self.height)
        fov_rad = math.radians(self.fov_deg)
        half_tan = math.tan(0.5 * fov_rad)

        ys = torch.linspace(-half_tan, half_tan, self.height, dtype=dtype, device=self.device)
        xs = torch.linspace(-half_tan * aspect, half_tan * aspect, self.width, dtype=dtype, device=self.device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        zz = torch.ones_like(xx)
        dirs_local = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
        dirs_local = dirs_local / torch.linalg.norm(dirs_local, dim=-1, keepdim=True)

        pi = PenroseTwistorHarmonicEngine.points_to_celestial_spinors(dirs_local)

        if camera_center is not None:
            # Anchor initial camera event on the t = 0 spatial hypersurface: X_cam = (0, c)
            zero_t = torch.zeros(1, dtype=dtype, device=self.device)
            cam_4d = torch.cat([zero_t, camera_center.to(dtype=dtype, device=self.device)], dim=0)
            X_cam = PenroseTwistorCore.spacetime_to_hermitian(cam_4d)
            omega = 1j * torch.einsum("ab,mb->ma", X_cam, pi)
        else:
            cam_4d = torch.zeros(4, dtype=dtype, device=self.device)
            X_cam = PenroseTwistorCore.spacetime_to_hermitian(cam_4d)
            omega = torch.zeros_like(pi)

        Z = torch.cat([omega, pi], dim=-1)

        t3 = theta_3d if theta_3d is not None else torch.zeros(3, dtype=dtype, device=self.device)
        b3 = boost_3d if boost_3d is not None else torch.zeros(3, dtype=dtype, device=self.device)
        tr4 = trans_4d if trans_4d is not None else torch.zeros(4, dtype=dtype, device=self.device)
        sc4 = special_conformal_4d if special_conformal_4d is not None else torch.zeros(4, dtype=dtype, device=self.device)

        G = self.build_full_15param_su22_matrix(t3, b3, tr4, sc4, dilation_lambda=dilation_lambda)
        Z_transformed = torch.einsum("ab,mb->ma", G, Z)

        ray_origins_3d, X_ray_herm, Z_norm = self.reconstruct_ray_origins_from_twistors(Z_transformed)
        pi_transformed = Z_norm[:, 2:4]
        ray_dirs = PenroseTwistorHarmonicEngine.spinor_to_unit_ray(pi_transformed)

        if return_ray_origins:
            return pi_transformed, ray_dirs, ray_origins_3d, X_ray_herm, Z_norm
        return pi_transformed, ray_dirs

    def render(
        self,
        pts_3d: torch.Tensor,
        camera_center: torch.Tensor,
        ray_dirs: torch.Tensor = None,
        sigma: float = 0.15,
        ray_spinors: torch.Tensor = None,
    ) -> torch.Tensor:
        if ray_spinors is None:
            if ray_dirs is None:
                raise ValueError("Either ray_dirs or ray_spinors must be provided to TwistorRenderer.render()")
            ray_spinors = PenroseTwistorHarmonicEngine.points_to_celestial_spinors(ray_dirs)

        r_perp_sq, proj_par, dist_sq = self.spinor_perpendicular_residual_sq(
            pts_3d, camera_center, ray_spinors, return_parallel=True
        )
        forward_mask = proj_par > 0.0
        weights = torch.exp(-r_perp_sq / (2.0 * (sigma ** 2))) / (1.0 + dist_sq)
        weights = torch.where(forward_mask, weights, torch.zeros_like(weights))
        return torch.sum(weights, dim=-1)