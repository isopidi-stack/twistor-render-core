import math
import torch

class TwistorCohomologyEngine:
    """
    Implements the Penrose Transform for elementary states f(Z) = 1 / ((A.Z)(B.Z)) in H^1(PT, O(-2))
    with explicit CP^1 pole tracking, analytical residue cross-checking, and U_0 / U_1 Cech coboundaries.
    No artificial epsilon regularizers (+1e-12) are used, preserving strict algebraic homogeneity.
    """
    # Dual twistors A_alpha = (alpha_A, mu^{A'}) and B_alpha = (beta_A, nu^{A'})
    # Chosen so that near x = 0:
    #   Polezeta_A(0) = -mu^0 / mu^1 = -(-0.25) / 1.0 = +0.25 (Strictly INSIDE |zeta| = 1, in patch U_1^c)
    #   Polezeta_B(0) = -nu^0 / nu^1 = -( 2.50) / 1.0 = -2.50 (Strictly OUTSIDE |zeta| = 1, in patch U_0^c)
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
    def check_homogeneity(func, Z: torch.Tensor, k: int, tol: float = 1e-13) -> float:
        """
        Returns the maximum relative homogeneity error |f(lambda Z) - lambda^k f(Z)| / |lambda^k f(Z)|.
        """
        lambda_c = torch.tensor(1.68 + 0.75j, dtype=Z.dtype, device=Z.device)
        f_orig = func(Z)
        f_scaled = func(lambda_c * Z)
        expected = (lambda_c ** k) * f_orig
        rel_err = torch.max(torch.abs(f_scaled - expected) / torch.abs(expected)).item()
        return rel_err

    @staticmethod
    def linear_forms_on_cp1(x4: torch.Tensor, A: torch.Tensor = None, B: torch.Tensor = None):
        """
        Restricts A.Z and B.Z to the spacetime line L_x in CP^3 via omega^A = i X^{AA'} pi_{A'}
        for pi_{A'} = (1, zeta)^T:
            A.Z(zeta) = p0(x) + p1(x) * zeta
            B.Z(zeta) = q0(x) + q1(x) * zeta
        Returns (p0, p1, q0, q1) and the two pole locations (zeta_A, zeta_B).
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

        # For any dual twistor W = (w0, w1, m0, m1):
        # W.Z(zeta) = w0 * i(X00 + X01*zeta) + w1 * i(X10 + X11*zeta) + m0 + m1*zeta
        p0 = 1j * (A[0] * X00 + A[1] * X10) + A[2]
        p1 = 1j * (A[0] * X01 + A[1] * X11) + A[3]

        q0 = 1j * (B[0] * X00 + B[1] * X10) + B[2]
        q1 = 1j * (B[0] * X01 + B[1] * X11) + B[3]

        zeta_A = -p0 / p1
        zeta_B = -q0 / q1
        return p0, p1, q0, q1, zeta_A, zeta_B

    @staticmethod
    def analytical_residue_transform(x4: torch.Tensor, A: torch.Tensor = None, B: torch.Tensor = None) -> torch.Tensor:
        """
        Evaluates the exact analytical Cauchy residue at the interior pole zeta_A (|zeta_A| < 1 < |zeta_B|):
            phi(x) = Res_{zeta = zeta_A} [ 1 / ((p0 + p1*zeta)(q0 + q1*zeta)) ] = 1 / (p1*q0 - p0*q1).
        """
        p0, p1, q0, q1, zeta_A, zeta_B = TwistorCohomologyEngine.linear_forms_on_cp1(x4, A, B)
        if not (torch.all(torch.abs(zeta_A) < 1.0) and torch.all(torch.abs(zeta_B) > 1.0)):
            raise ValueError(f"Contour |zeta|=1 does not separate poles! |zeta_A|={torch.abs(zeta_A).item():.3f}, |zeta_B|={torch.abs(zeta_B).item():.3f}")
        return 1.0 / (p1 * q0 - p0 * q1)

    @staticmethod
    def contour_integral_transform(x4: torch.Tensor, num_pts: int = 256, A: torch.Tensor = None, B: torch.Tensor = None, coboundary_fn=None) -> torch.Tensor:
        """
        Evaluates the contour integral over |zeta| = 1 directly from the homogeneous twistor function f(Z):
            phi(x) = (1 / 2*pi*i) oint_{|zeta|=1} f(i X^{AA'} pi_{A'}, pi_{A'}) pi_{C'} d(pi^{C'})
        Verifies strictly that the contour separates the poles |zeta_A(x)| < 1 < |zeta_B(x)|.
        """
        _, _, _, _, zeta_A, zeta_B = TwistorCohomologyEngine.linear_forms_on_cp1(x4, A, B)
        if not (torch.all(torch.abs(zeta_A) < 1.0) and torch.all(torch.abs(zeta_B) > 1.0)):
            raise ValueError(f"Contour |zeta|=1 does not separate poles! |zeta_A|={torch.abs(zeta_A).item():.3f}, |zeta_B|={torch.abs(zeta_B).item():.3f}")

        dtype = torch.complex128 if x4.dtype == torch.float64 else torch.complex64
        device = x4.device
        phis = torch.linspace(0.0, 2.0 * math.pi, num_pts + 1, dtype=x4.dtype, device=device)[:-1]
        dphi = 2.0 * math.pi / num_pts
        zeta = torch.complex(torch.cos(phis), torch.sin(phis))  # (K,)

        t, x, y, z = x4[..., 0].unsqueeze(-1), x4[..., 1].unsqueeze(-1), x4[..., 2].unsqueeze(-1), x4[..., 3].unsqueeze(-1)
        X00, X01 = t + z, x - 1j * y
        X10, X11 = x + 1j * y, t - z

        pi_0 = torch.ones_like(zeta)
        pi_1 = zeta
        omega_0 = 1j * (X00 * pi_0 + X01 * pi_1)
        omega_1 = 1j * (X10 * pi_0 + X11 * pi_1)

        Z = torch.stack([omega_0, omega_1, pi_0.expand_as(omega_0), pi_1.expand_as(omega_1)], dim=-1)
        f_val = TwistorCohomologyEngine.elementary_twistor_function(Z, A, B)

        if coboundary_fn is not None:
            f_val = f_val + coboundary_fn(Z)

        dzeta = 1j * zeta * dphi
        return torch.sum(f_val * dzeta / (2.0 * math.pi * 1.0j), dim=-1)