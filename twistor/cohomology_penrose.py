import math
import torch

class CechCohomologyClassH1:
    """
    Represents a Cech 1-cocycle [f_{01}] in H^1(L_x, O(-2)) induced by the global twistor cover
    U_A = {Z in PT : A.Z != 0}, U_B = {Z in PT : B.Z != 0} restricted to the incidence line L_x = CP^1.

    Contour & Open Cover Geometry on L_x:
      - In the inhomogeneous chart U_0 = {pi_0 != 0} with coordinate zeta = pi_1 / pi_0,
        the linear forms A.Z and B.Z vanish at simple poles zeta_A(x) and zeta_B(x).
      - Provided zeta_A(x) != zeta_B(x) (i.e. p1*q0 - p0*q1 != 0) and p1 != 0, q1 != 0,
        the pole-centered separating contour is:
            Gamma_x = { zeta(phi) = zeta_A(x) + R(x) * exp(i * phi) },  R(x) = 0.5 * |zeta_A(x) - zeta_B(x)|.
      - In the local coordinate w = zeta - zeta_A(x), the interior pole is at w = 0, the contour is |w| = R(x),
        and the exterior pole is at |w_B| = 2 * R(x). Thus the pole modulus ratio is IDENTICALLY r = 1/2
        everywhere in spacetime (regardless of whether |zeta_A| > |zeta_B| or |zeta_A| / |zeta_B| -> 1),
        giving spectral trapezoid convergence (1/2)^N ~ 1e-77 at N = 256.
      - Open cover of L_x:
          V_0(x) = { |zeta - zeta_A(x)| < 1.2 * R(x) }
          V_1(x) = { |zeta - zeta_A(x)| > 0.8 * R(x) } union {infty}
    """
    def __init__(self, A: torch.Tensor, B: torch.Tensor):
        self.A = A
        self.B = B

    @staticmethod
    def verify_patch_holomorphicity(
        x4: torch.Tensor,
        zeta_A: torch.Tensor,
        radius_R: torch.Tensor,
        C1: torch.Tensor,
        C2: torch.Tensor,
        patch: str = "V0",
    ) -> dict:
        """
        Verifies holomorphicity of a non-vacuous simple-pole rational section:
            h(Z) = c / ((C1 . Z) * (C2 . Z))
        Unlike double poles c / (C . Z)^2 (which have zero residue everywhere on CP^1),
        a two-pole section with C1 != C2 has non-zero residues at zeta_C1(x) and zeta_C2(x):
          - For h in O(V_0)(-2): BOTH poles zeta_C1, zeta_C2 must lie strictly outside Gamma_x
            (|zeta_Ci - zeta_A| > R(x)), so h has no poles inside the disk bounded by Gamma_x.
          - For h in O(V_1)(-2): BOTH poles zeta_C1, zeta_C2 must lie strictly inside Gamma_x
            (|zeta_Ci - zeta_A| < R(x)) and p1_C1, p1_C2 != 0 (regular of order O(zeta^{-2}) at infinity),
            so h is holomorphic throughout the exterior patch V_1(x) (and its two opposite residues
            inside Gamma_x cancel by Cauchy's theorem!).
          - Conversely, if one pole lies inside Gamma_x and the other outside Gamma_x, h(Z) is NOT
            a coboundary section on V_0 or V_1, and verify_patch_holomorphicity raises ValueError!
        """
        _, p1_1, _, p1_2, z1, z2 = TwistorCohomologyEngine.linear_forms_on_cp1(x4, C1, C2)
        if torch.any(torch.abs(p1_1) < 1e-10) or torch.any(torch.abs(p1_2) < 1e-10):
            raise ValueError("Degenerate linear form at infinity (p1 == 0) violates O(-2) regularity!")

        d1 = torch.abs(z1 - zeta_A)
        d2 = torch.abs(z2 - zeta_A)

        if patch == "V0":
            if not (torch.all(d1 > radius_R * 1.05) and torch.all(d2 > radius_R * 1.05)):
                raise ValueError("Invalid V_0 section: at least one simple pole lies inside V_0 (|zeta - zeta_A| <= R)!")
        elif patch == "V1":
            if not (torch.all(d1 < radius_R * 0.95) and torch.all(d2 < radius_R * 0.95)):
                raise ValueError("Invalid V_1 section: at least one simple pole lies inside V_1 (|zeta - zeta_A| >= R)!")
        else:
            raise ValueError(f"Unknown patch '{patch}'; expected 'V0' or 'V1'.")

        return {"d1": d1, "d2": d2, "radius_R": radius_R, "patch": patch, "valid": True}


class TwistorCohomologyEngine:
    """
    Implements the Elementary Scalar Penrose Transform for f(Z) = 1 / ((A.Z)(B.Z))
    using the pole-centered contour Gamma_x = { zeta_A(x) + 0.5 * |zeta_A(x) - zeta_B(x)| * e^{i*phi} }
    and non-vacuous simple-pole Cech cocycle / coboundary verification.
    """
    # At x = 0:
    #   zeta_A(0) = -(-0.25 + 0.05j) / 1.0 = +0.25 - 0.05j
    #   zeta_B(0) = -( 2.50 - 0.20j) / 1.0 = -2.50 + 0.20j
    DEFAULT_A = torch.tensor([0.30 + 0.10j, -0.20 + 0.15j, -0.25 + 0.05j, 1.00 + 0.00j], dtype=torch.complex128)
    DEFAULT_B = torch.tensor([0.15 - 0.25j,  0.40 + 0.10j,  2.50 - 0.20j, 1.00 + 0.00j], dtype=torch.complex128)

    @staticmethod
    def elementary_twistor_function(Z: torch.Tensor, A: torch.Tensor = None, B: torch.Tensor = None) -> torch.Tensor:
        A = (A if A is not None else TwistorCohomologyEngine.DEFAULT_A).to(dtype=Z.dtype, device=Z.device)
        B = (B if B is not None else TwistorCohomologyEngine.DEFAULT_B).to(dtype=Z.dtype, device=Z.device)
        AZ = torch.einsum("a,...a->...", A, Z)
        BZ = torch.einsum("a,...a->...", B, Z)
        return 1.0 / (AZ * BZ)

    @staticmethod
    def check_homogeneity(func, Z: torch.Tensor, k: int, tol: float = 1e-13) -> tuple[bool, float]:
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
    def cp1_chordal_pole_distance(p0: torch.Tensor, p1: torch.Tensor, q0: torch.Tensor, q1: torch.Tensor) -> torch.Tensor:
        """
        Chart-independent Fubini-Study chordal distance on CP^1 between poles [-p0:p1] and [-q0:q1].
        Valid even when a pole lies at zeta = infinity (p1 == 0 or q1 == 0).
        """
        det = torch.abs(p1 * q0 - p0 * q1)
        norm_p = torch.sqrt(torch.abs(p0) ** 2 + torch.abs(p1) ** 2).clamp(min=1e-30)
        norm_q = torch.sqrt(torch.abs(q0) ** 2 + torch.abs(q1) ** 2).clamp(min=1e-30)
        return det / (norm_p * norm_q)

    @staticmethod
    def linear_forms_homogeneous(x4: torch.Tensor, A: torch.Tensor = None, B: torch.Tensor = None):
        dtype = torch.complex128 if x4.dtype == torch.float64 else torch.complex64
        device = x4.device
        A = (A if A is not None else TwistorCohomologyEngine.DEFAULT_A).to(dtype=dtype, device=device)
        B = (B if B is not None else TwistorCohomologyEngine.DEFAULT_B).to(dtype=dtype, device=device)
        t, x, y, z = x4[..., 0], x4[..., 1], x4[..., 2], x4[..., 3]
        X00, X01, X10, X11 = t + z, x - 1j * y, x + 1j * y, t - z
        p0 = 1j * (A[0] * X00 + A[1] * X10) + A[2]
        p1 = 1j * (A[0] * X01 + A[1] * X11) + A[3]
        q0 = 1j * (B[0] * X00 + B[1] * X10) + B[2]
        q1 = 1j * (B[0] * X01 + B[1] * X11) + B[3]
        return p0, p1, q0, q1

    @staticmethod
    def linear_forms_on_cp1(x4: torch.Tensor, A: torch.Tensor = None, B: torch.Tensor = None):
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

        if torch.any(torch.abs(p1) < 1e-12) or torch.any(torch.abs(q1) < 1e-12):
            raise ValueError("Pole lies at zeta = infty (p1 == 0 or q1 == 0) in the pi_0 = 1 chart!")

        zeta_A = -p0 / p1
        zeta_B = -q0 / q1
        return p0, p1, q0, q1, zeta_A, zeta_B

    @staticmethod
    def compute_pole_centered_contour(x4: torch.Tensor, A: torch.Tensor = None, B: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the pole-centered contour parameters (center = zeta_A(x), radius = 0.5 * |zeta_A(x) - zeta_B(x)|).
        Valid everywhere in spacetime where the two poles do not coalesce (|zeta_A(x) - zeta_B(x)| > 1e-10),
        completely independent of whether |zeta_A| < |zeta_B| or |zeta_A| / |zeta_B| -> 1.
        """
        _, _, _, _, zeta_A, zeta_B = TwistorCohomologyEngine.linear_forms_on_cp1(x4.detach(), A, B)
        sep = torch.abs(zeta_A - zeta_B)
        if torch.any(sep < 1e-10):
            raise ValueError("Singular null cone: poles zeta_A(x) and zeta_B(x) coincide (p1*q0 - p0*q1 == 0)!")
        return zeta_A, 0.5 * sep

    @staticmethod
    def compute_adaptive_contour_radius(x4: torch.Tensor, A: torch.Tensor = None, B: torch.Tensor = None) -> torch.Tensor:
        """
        Computes the concentric geometric-mean radius rho(x) = sqrt(|zeta_A(x)| * |zeta_B(x)|)
        when 0 < |zeta_A(x)| < |zeta_B(x)|. Retained for concentric-annulus tests.
        """
        _, _, _, _, zeta_A, zeta_B = TwistorCohomologyEngine.linear_forms_on_cp1(x4.detach(), A, B)
        mag_A = torch.abs(zeta_A)
        mag_B = torch.abs(zeta_B)
        if torch.any(mag_A < 1e-12) or (not torch.all(mag_A < mag_B * (1.0 - 1e-7))):
            raise ValueError("Concentric circle requires 0 < |zeta_A(x)| < |zeta_B(x)|!")
        return torch.sqrt(mag_A * mag_B)

    @staticmethod
    def analytical_residue_transform(x4: torch.Tensor, A: torch.Tensor = None, B: torch.Tensor = None) -> torch.Tensor:
        p0, p1, q0, q1 = TwistorCohomologyEngine.linear_forms_homogeneous(x4, A, B)
        denom = p1 * q0 - p0 * q1
        if torch.any(torch.abs(denom) < 1e-12):
            raise ValueError("Point lies on singular null cone!")
        return 1.0 / denom

    @staticmethod
    def contour_integral_transform(
        x4: torch.Tensor,
        num_pts: int = 256,
        A: torch.Tensor = None,
        B: torch.Tensor = None,
        contour_mode: str = "pole_centered",
        adaptive_radius: bool = True,
        cech_coboundary_v0: tuple[torch.Tensor, torch.Tensor] = None,
        cech_coboundary_v1: tuple[torch.Tensor, torch.Tensor] = None,
        extra_cocycle_term: tuple[torch.Tensor, torch.Tensor, complex] = None,
    ) -> torch.Tensor:
        """
        Evaluates the Penrose contour integral:
            phi(x) = (1 / 2*pi*i) oint_{Gamma_x} f(i X^{AA'} pi_{A'}, pi_{A'}) pi_{C'} d(pi^{C'})
        Contour Modes:
          - `contour_mode="pole_centered"` (default): Integrates around zeta(phi) = zeta_A(x) + 0.5*|zeta_A - zeta_B|*e^{i*phi},
            giving machine-precision (~1e-16) agreement with the Cauchy residue at zeta_A(x) for ALL x4 off the singular
            cone (including x in [-10, 10]^4 and tight poles r = |zeta_A|/|zeta_B| -> 1).
          - `contour_mode="concentric"`: Uses origin-centered |zeta| = sqrt(|zeta_A||zeta_B|) (if adaptive_radius=True)
            or fixed |zeta| = 1 (if adaptive_radius=False).
        """
        device = x4.device
        phis = torch.linspace(0.0, 2.0 * math.pi, num_pts + 1, dtype=x4.dtype, device=device)[:-1]
        dphi = 2.0 * math.pi / num_pts
        unit_circle = torch.complex(torch.cos(phis), torch.sin(phis))  # (K,)

        if contour_mode == "pole_centered" and adaptive_radius:
            center_A, radius_R = TwistorCohomologyEngine.compute_pole_centered_contour(x4, A, B)
            zeta = center_A.unsqueeze(-1) + radius_R.unsqueeze(-1) * unit_circle
            dzeta = 1j * radius_R.unsqueeze(-1) * unit_circle * dphi
        elif contour_mode in ("origin_centered", "concentric"):
            center_A = torch.zeros_like(x4[..., 0], dtype=torch.complex128 if x4.dtype == torch.float64 else torch.complex64)
            if adaptive_radius:
                radius_R = TwistorCohomologyEngine.compute_adaptive_contour_radius(x4, A, B)
            else:
                _, _, _, _, zA, zB = TwistorCohomologyEngine.linear_forms_on_cp1(x4.detach(), A, B)
                if not (torch.all(torch.abs(zA) < 1.0) and torch.all(torch.abs(zB) > 1.0)):
                    raise ValueError("Fixed unit contour |zeta|=1 does not separate poles!")
                radius_R = torch.ones_like(x4[..., 0])
            zeta = radius_R.unsqueeze(-1) * unit_circle
            dzeta = 1j * zeta * dphi
        else:
            raise ValueError(f"Unknown contour_mode: {contour_mode}")

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

        # Non-vacuous simple-pole coboundary on V_0: h_0(Z) = c0 / ((C1.Z)(C2.Z)) with both poles outside Gamma_x
        if cech_coboundary_v0 is not None:
            C1, C2 = cech_coboundary_v0
            CechCohomologyClassH1.verify_patch_holomorphicity(x4.detach(), center_A, radius_R, C1, C2, patch="V0")
            C1 = C1.to(dtype=Z.dtype, device=device)
            C2 = C2.to(dtype=Z.dtype, device=device)
            h0_val = 0.65 / (torch.einsum("a,...a->...", C1, Z) * torch.einsum("a,...a->...", C2, Z))
            f_val = f_val + h0_val

        # Non-vacuous simple-pole coboundary on V_1: h_1(Z) = c1 / ((D1.Z)(D2.Z)) with both poles inside Gamma_x
        if cech_coboundary_v1 is not None:
            D1, D2 = cech_coboundary_v1
            CechCohomologyClassH1.verify_patch_holomorphicity(x4.detach(), center_A, radius_R, D1, D2, patch="V1")
            D1 = D1.to(dtype=Z.dtype, device=device)
            D2 = D2.to(dtype=Z.dtype, device=device)
            h1_val = -0.45 / (torch.einsum("a,...a->...", D1, Z) * torch.einsum("a,...a->...", D2, Z))
            f_val = f_val - h1_val

        # Optional negative-control cocycle g(Z) = weight / ((C_in.Z)(D_out.Z)) with poles on opposite sides of Gamma_x
        if extra_cocycle_term is not None:
            C_in, D_out, weight = extra_cocycle_term
            C_in = C_in.to(dtype=Z.dtype, device=device)
            D_out = D_out.to(dtype=Z.dtype, device=device)
            g_val = weight / (torch.einsum("a,...a->...", C_in, Z) * torch.einsum("a,...a->...", D_out, Z))
            f_val = f_val + g_val

        return torch.sum(f_val * dzeta / (2.0 * math.pi * 1.0j), dim=-1)