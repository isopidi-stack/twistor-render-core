import math
import torch

class CechCohomologyClassH1:
    """
    Explicit Cech 1-Cocycle Representative [f_{01}] in H^1(L_x, O(-2)) over the 2-patch open cover
    U = {U_0, U_1} of the celestial Riemann sphere L_x = CP^1:
      - Homogeneous coordinates: pi_{A'} = (pi_0, pi_1)^T, inhomogeneous zeta = pi_1 / pi_0 on U_0 = {pi_0 != 0}.
      - Given interior pole zeta_A(x) and exterior pole zeta_B(x) with |zeta_A(x)| < |zeta_B(x)|,
        the adaptive separating contour is Gamma_rho = { |zeta| = rho(x) }, where
            rho(x) = sqrt(|zeta_A(x)| * |zeta_B(x)|).
      - Patch U_0(x) = { |zeta| <= rho(x)*(1 + eps) } contains pole zeta_A(x).
      - Patch U_1(x) = { |zeta| >= rho(x)*(1 - eps) } union {infty} contains pole zeta_B(x).
      - A Cech 0-cochain pair (h_0, h_1) in C^0(U, O(-2)) is a valid coboundary delta(h)_{01} = h_0 - h_1
        if and only if:
          (a) h_0(Z) and h_1(Z) are strictly homogeneous of degree k = -2,
          (b) h_0(Z(zeta)) is holomorphic throughout U_0(x) (all poles of h_0 have |zeta| > rho(x)), and
          (c) h_1(Z(zeta)) is holomorphic throughout U_1(x) (all poles of h_1 have |zeta| < rho(x) and h_1 = O(zeta^{-2}) at infinity).
    """
    def __init__(self, A: torch.Tensor, B: torch.Tensor):
        self.A = A
        self.B = B

    @staticmethod
    def verify_patch_holomorphicity(x4: torch.Tensor, rho: torch.Tensor, C_u0: torch.Tensor, D_u1: torch.Tensor) -> dict:
        """
        Explicitly computes the pole locations of the candidate Cech 0-cochain sections:
            h_0(Z) = c_0 / (C_u0 . Z)^2   (must be holomorphic on U_0: |zeta_C(x)| > rho(x))
            h_1(Z) = d_1 / (D_u1 . Z)^2   (must be holomorphic on U_1: |zeta_D(x)| < rho(x))
        Raises ValueError if either section has a pole inside its claimed patch of holomorphicity.
        """
        _, p1_C, _, p1_D, zeta_C, zeta_D = TwistorCohomologyEngine.linear_forms_on_cp1(x4, C_u0, D_u1)
        mag_C = torch.abs(zeta_C)
        mag_D = torch.abs(zeta_D)

        # Check infinity regularity for U_1: p1_D must be non-zero so (D_u1 . Z)^{-2} ~ (p1_D * zeta)^{-2} = O(zeta^{-2}) at zeta -> infty
        if torch.any(torch.abs(p1_D) < 1e-10):
            raise ValueError("Invalid U_1 coboundary section h_1: pole at zeta = infty violates U_1 holomorphicity!")

        # U_0 holomorphicity: pole zeta_C of h_0 must lie strictly outside U_0 (|zeta_C| > rho)
        if not torch.all(mag_C > rho * 1.02):
            raise ValueError("Invalid U_0 coboundary section h_0: pole lies inside U_0 (|zeta_C| <= rho)!")

        # U_1 holomorphicity: pole zeta_D of h_1 must lie strictly outside U_1 (i.e. inside |zeta| < rho)
        if not torch.all(mag_D < rho * 0.98):
            raise ValueError("Invalid U_1 coboundary section h_1: pole lies inside U_1 (|zeta_D| >= rho)!")

        return {
            "u0_pole_mag": mag_C,
            "u1_pole_mag": mag_D,
            "contour_rho": rho,
            "u0_holomorphic": True,
            "u1_holomorphic": True,
        }


class TwistorCohomologyEngine:
    """
    Implements the Elementary Penrose Transform for f(Z) = 1 / ((A.Z)(B.Z)) in H^1(PT, O(-2))
    with Adaptive Geometric-Mean Contour Radius rho(x) = sqrt(|zeta_A(x)| * |zeta_B(x)|)
    and explicit U_0 / U_1 Cech open-cover patch holomorphicity verification.
    """
    # Dual twistors A_alpha = (alpha_A, mu^{A'}) and B_alpha = (beta_A, nu^{A'})
    # At x = 0:
    #   zeta_A(0) = -mu^0 / mu^1 = -(-0.25 + 0.05j) / 1.0 = +0.25 - 0.05j  (|zeta_A(0)| = 0.25495 < 1)
    #   zeta_B(0) = -nu^0 / nu^1 = -( 2.50 - 0.20j) / 1.0 = -2.50 + 0.20j  (|zeta_B(0)| = 2.50799 > 1)
    DEFAULT_A = torch.tensor([0.30 + 0.10j, -0.20 + 0.15j, -0.25 + 0.05j, 1.00 + 0.00j], dtype=torch.complex128)
    DEFAULT_B = torch.tensor([0.15 - 0.25j,  0.40 + 0.10j,  2.50 - 0.20j, 1.00 + 0.00j], dtype=torch.complex128)

    @staticmethod
    def elementary_twistor_function(Z: torch.Tensor, A: torch.Tensor = None, B: torch.Tensor = None) -> torch.Tensor:
        """
        Evaluates the exact homogeneity k = -2 twistor function:
            f(Z) = 1 / ((A_alpha Z^alpha) * (B_beta Z^beta))
        with zero artificial regularizers.
        """
        A = (A if A is not None else TwistorCohomologyEngine.DEFAULT_A).to(dtype=Z.dtype, device=Z.device)
        B = (B if B is not None else TwistorCohomologyEngine.DEFAULT_B).to(dtype=Z.dtype, device=Z.device)
        AZ = torch.einsum("a,...a->...", A, Z)
        BZ = torch.einsum("a,...a->...", B, Z)
        return 1.0 / (AZ * BZ)

    @staticmethod
    def check_homogeneity(func, Z: torch.Tensor, k: int, tol: float = 1e-13) -> tuple[bool, float]:
        """
        Checks f(lambda Z) == lambda^k f(Z) for complex scaling lambda.
        Uses `tol` explicitly and guards against division by zero if f(Z) == 0.
        Returns (is_homogeneous: bool, max_rel_err: float).
        """
        lambda_c = torch.tensor(1.68 + 0.75j, dtype=Z.dtype, device=Z.device)
        f_orig = func(Z)
        f_scaled = func(lambda_c * Z)
        expected = (lambda_c ** k) * f_orig
        denom = torch.abs(expected)
        if torch.any(denom < 1e-30):
            raise ValueError("check_homogeneity: f(Z) evaluated to zero; cannot normalize relative error.")
        rel_err = float(torch.max(torch.abs(f_scaled - expected) / denom).item())
        return (rel_err < tol), rel_err

    @staticmethod
    def linear_forms_on_cp1(x4: torch.Tensor, A: torch.Tensor = None, B: torch.Tensor = None):
        """
        Restricts A.Z and B.Z to the spacetime line L_x in CP^3 via omega^A = i X^{AA'} pi_{A'}
        for pi_{A'} = (1, zeta)^T:
            A.Z(zeta) = p0(x) + p1(x) * zeta
            B.Z(zeta) = q0(x) + q1(x) * zeta
        Returns (p0, p1, q0, q1, zeta_A, zeta_B).
        """
        dtype = torch.complex128 if x4.dtype == torch.float64 else torch.complex64
        device = x4.device
        A = (A if A is not None else TwistorCohomologyEngine.DEFAULT_A).to(dtype=dtype, device=device)
        B = (B if B is not None else TwistorCohomologyEngine.DEFAULT_B).to(dtype=dtype, device=device)

        t, x, y, z = x4[..., 0], x4[..., 1], x4[..., 2], x4[..., 3]
        X00 = t + z
        X01 = x - 1j * y
        X10 = x + 1j * y
        X11 = t - z

        p0 = 1j * (A[0] * X00 + A[1] * X10) + A[2]
        p1 = 1j * (A[0] * X01 + A[1] * X11) + A[3]

        q0 = 1j * (B[0] * X00 + B[1] * X10) + B[2]
        q1 = 1j * (B[0] * X01 + B[1] * X11) + B[3]

        zeta_A = -p0 / p1
        zeta_B = -q0 / q1
        return p0, p1, q0, q1, zeta_A, zeta_B

    @staticmethod
    def compute_adaptive_contour_radius(x4: torch.Tensor, A: torch.Tensor = None, B: torch.Tensor = None) -> torch.Tensor:
        """
        Computes the geometric-mean separating contour radius:
            rho(x) = sqrt(|zeta_A(x)| * |zeta_B(x)|)
        Verifies strictly across arbitrary batch dimensions that |zeta_A(x)| < |zeta_B(x)|
        so the circle |zeta| = rho(x) separates the interior pole zeta_A from the exterior pole zeta_B.
        Detached from autograd so contour deformation remains topologically locally constant during Hessian evaluation.
        """
        _, _, _, _, zeta_A, zeta_B = TwistorCohomologyEngine.linear_forms_on_cp1(x4.detach(), A, B)
        mag_A = torch.abs(zeta_A)
        mag_B = torch.abs(zeta_B)
        if not torch.all(mag_A < mag_B * (1.0 - 1e-7)):
            raise ValueError("Poles are not radially separated (|zeta_A(x)| >= |zeta_B(x)|); no concentric circle separates them!")
        return torch.sqrt(mag_A * mag_B)

    @staticmethod
    def analytical_residue_transform(x4: torch.Tensor, A: torch.Tensor = None, B: torch.Tensor = None) -> torch.Tensor:
        """
        Evaluates the exact analytical Cauchy residue at the interior pole zeta_A (|zeta_A| < rho < |zeta_B|):
            phi(x) = Res_{zeta = zeta_A} [ 1 / ((p0 + p1*zeta)(q0 + q1*zeta)) ] = 1 / (p1*q0 - p0*q1).
        """
        p0, p1, q0, q1, zeta_A, zeta_B = TwistorCohomologyEngine.linear_forms_on_cp1(x4, A, B)
        if not torch.all(torch.abs(zeta_A) < torch.abs(zeta_B) * (1.0 - 1e-7)):
            raise ValueError("Poles are not radially separated (|zeta_A(x)| >= |zeta_B(x)|)!")
        return 1.0 / (p1 * q0 - p0 * q1)

    @staticmethod
    def contour_integral_transform(
        x4: torch.Tensor,
        num_pts: int = 256,
        A: torch.Tensor = None,
        B: torch.Tensor = None,
        adaptive_radius: bool = True,
        cech_coboundary_pair: tuple[torch.Tensor, torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Evaluates the Penrose contour integral over Gamma_rho = { zeta = rho(x) * exp(i*phi) }:
            phi(x) = (1 / 2*pi*i) oint_{|zeta|=rho(x)} f(i X^{AA'} pi_{A'}, pi_{A'}) pi_{C'} d(pi^{C'})
        When `adaptive_radius=True` (default), uses rho(x) = sqrt(|zeta_A(x)| * |zeta_B(x)|),
        achieving ~1e-16 relative accuracy even when |zeta_A(x)| is within 1e-3 of the unit circle |zeta|=1.
        When `cech_coboundary_pair=(C_u0, D_u1)` is provided, explicitly verifies patch holomorphicity
        on U_0 and U_1 before adding delta(h)_{01} = h_0(Z) - h_1(Z).
        """
        if adaptive_radius:
            rho = TwistorCohomologyEngine.compute_adaptive_contour_radius(x4, A, B)  # (...)
        else:
            _, _, _, _, zeta_A, zeta_B = TwistorCohomologyEngine.linear_forms_on_cp1(x4.detach(), A, B)
            if not (torch.all(torch.abs(zeta_A) < 1.0) and torch.all(torch.abs(zeta_B) > 1.0)):
                raise ValueError("Fixed unit contour |zeta|=1 does not separate poles!")
            rho = torch.ones_like(x4[..., 0])

        device = x4.device
        phis = torch.linspace(0.0, 2.0 * math.pi, num_pts + 1, dtype=x4.dtype, device=device)[:-1]
        dphi = 2.0 * math.pi / num_pts
        unit_circle = torch.complex(torch.cos(phis), torch.sin(phis))  # (K,)
        zeta = rho.unsqueeze(-1) * unit_circle  # (..., K)

        t = x4[..., 0].unsqueeze(-1)
        x = x4[..., 1].unsqueeze(-1)
        y = x4[..., 2].unsqueeze(-1)
        z = x4[..., 3].unsqueeze(-1)
        X00, X01 = t + z, x - 1j * y
        X10, X11 = x + 1j * y, t - z

        pi_0 = torch.ones_like(zeta)
        pi_1 = zeta
        omega_0 = 1j * (X00 * pi_0 + X01 * pi_1)
        omega_1 = 1j * (X10 * pi_0 + X11 * pi_1)

        Z = torch.stack([omega_0, omega_1, pi_0, pi_1], dim=-1)  # (..., K, 4)
        f_val = TwistorCohomologyEngine.elementary_twistor_function(Z, A, B)

        if cech_coboundary_pair is not None:
            C_u0, D_u1 = cech_coboundary_pair
            # Strictly verify that h_0 is holomorphic on U_0 and h_1 is holomorphic on U_1!
            CechCohomologyClassH1.verify_patch_holomorphicity(x4.detach(), rho, C_u0, D_u1)
            C_u0 = C_u0.to(dtype=Z.dtype, device=device)
            D_u1 = D_u1.to(dtype=Z.dtype, device=device)
            CZ = torch.einsum("a,...a->...", C_u0, Z)
            DZ = torch.einsum("a,...a->...", D_u1, Z)
            h0_Z = 0.75 / (CZ ** 2)
            h1_Z = -0.45 / (DZ ** 2)
            f_val = f_val + (h0_Z - h1_Z)

        dzeta = 1j * zeta * dphi
        return torch.sum(f_val * dzeta / (2.0 * math.pi * 1.0j), dim=-1)