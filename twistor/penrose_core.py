import math
import torch
from twistor.incidence import sl2c_inverse_dagger, EPSILON

class PenroseTwistorCore:
    """
    Full Mathematical Stack for Sir Roger Penrose's Twistor Theory in CP^3 / T:
      1. 4-Spinor Twistors Z^alpha = (omega^A, pi_{A'}) in C^4 with SU(2,2) metric Sigma_{alpha beta'}
      2. Unified 4x4 Poincare/Conformal SU(2,2) Transformations P(Lambda, T)
      3. Infinity Twistor I_{alpha beta} and Metric Null-Separation via Twistor Inner Product
      4. Forward Tube PT^+ Positive-Frequency Splitting (Z^dagger Sigma Z > 0)
      5. Numerical Penrose Contour Integral Transform H^1(PT^+, O(-2)) -> Ker(Box)
    """
    @staticmethod
    def spacetime_to_hermitian(x4: torch.Tensor) -> torch.Tensor:
        """
        Maps 4D Minkowski events x^a = (t, x, y, z) to 2x2 Hermitian spinor matrices X^{AA'}
        such that det(X^{AA'}) = t^2 - x^2 - y^2 - z^2 (the exact Minkowski metric interval).
        """
        t, x, y, z = x4[..., 0], x4[..., 1], x4[..., 2], x4[..., 3]
        c_dtype = torch.complex128 if x4.dtype == torch.float64 else torch.complex64
        zero = torch.zeros_like(t)
        X00 = torch.complex(t + z, zero)
        X01 = torch.complex(x, -y)
        X10 = torch.complex(x,  y)
        X11 = torch.complex(t - z, zero)
        return torch.stack([
            torch.stack([X00, X01], dim=-1),
            torch.stack([X10, X11], dim=-1)
        ], dim=-2).to(c_dtype)

    @staticmethod
    def su22_metric(device=None, dtype=torch.complex128) -> torch.Tensor:
        """
        Returns the 4x4 SU(2,2) pseudo-Hermitian metric Sigma of signature (+, +, -, -):
            Sigma = [[0, I_2], [I_2, 0]]
        such that Z^dagger Sigma Z = omega^A bar(pi)_A + bar(omega)^{A'} pi_{A'} = 2s.
        """
        dev = device or torch.device("cpu")
        Z2 = torch.zeros((2, 2), dtype=dtype, device=dev)
        I2 = torch.eye(2, dtype=dtype, device=dev)
        top = torch.cat([Z2, I2], dim=-1)
        bot = torch.cat([I2, Z2], dim=-1)
        return torch.cat([top, bot], dim=-2)

    @staticmethod
    def su22_inner_product(Z1: torch.Tensor, Z2: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the SU(2,2)-invariant Hermitian inner product <Z1, Z2>_Sigma = Z1^dagger Sigma Z2.
        Two null twistors Z1, Z2 intersect in Minkowski space iff <Z1, Z2>_Sigma = 0.
        """
        Sigma = PenroseTwistorCore.su22_metric(device=Z1.device, dtype=Z1.dtype)
        return torch.einsum("...a,ab,...b->...", torch.conj(Z1), Sigma, Z2)

    @staticmethod
    def build_poincare_su22_matrix(Lambda: torch.Tensor, translation_4d: torch.Tensor) -> torch.Tensor:
        """
        Constructs the exact 4x4 SU(2,2) matrix P(Lambda, T) for a Lorentz transformation
        Lambda in SL(2,C) and 4D spacetime translation T^{AA'}:
            P = [[ Lambda,   i * T @ Lambda^{-dagger} ],
                 [ 0,        Lambda^{-dagger}         ]]
        Satisfies P^dagger Sigma P = Sigma identically.
        """
        T_herm = PenroseTwistorCore.spacetime_to_hermitian(translation_4d)
        L_inv_dag = sl2c_inverse_dagger(Lambda)
        upper_right = 1j * (T_herm @ L_inv_dag)
        zeros_lower = torch.zeros_like(Lambda)

        top = torch.cat([Lambda, upper_right], dim=-1)
        bot = torch.cat([zeros_lower, L_inv_dag], dim=-1)
        return torch.cat([top, bot], dim=-2)

    @staticmethod
    def lift_to_forward_tube(x4_real: torch.Tensor, timelike_eta: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
        """
        Implements Penrose Positive-Frequency Splitting by lifting real spacetime events x^a
        into the complex Forward Tube M^+:
            z^a = x^a - i * eta^a,   where eta^a is future-timelike (eta^0 > ||eta_vec||).
        Returns 4-twistor Z^alpha = (omega^A, pi_{A'}) in PT^+ where Z^dagger Sigma Z > 0.
        """
        X_herm = PenroseTwistorCore.spacetime_to_hermitian(x4_real)
        Y_herm = PenroseTwistorCore.spacetime_to_hermitian(timelike_eta)
        Z_complex = X_herm - 1j * Y_herm
        # omega^A = i Z_complex^{AA'} pi_{A'} = i X^{AA'} pi_{A'} + Y^{AA'} pi_{A'}
        omega = 1j * torch.einsum("...ab,...b->...a", Z_complex, pi)
        return torch.cat([omega, pi], dim=-1)

    @staticmethod
    def penrose_contour_transform_scalar(x4: torch.Tensor, x0_pole: torch.Tensor, num_contour_pts: int = 128) -> torch.Tensor:
        """
        Executes the Penrose Contour Integral Transform for a homogeneity -2 scalar twistor
        function f(Z) in H^1(PT^+, O(-2)):
            phi(x) = (1 / 2*pi*i) oint_Gamma f(i Z^{AA'} pi_{A'}, pi_{A'}) pi_{B'} d(pi^{B'})
        Using the elementary twistor dual poles corresponding to a null-cone singularity at x0_pole,
        this contour integral analytically produces a solution to the 4D massless wave equation:
            Box phi(x) = (d^2/dt^2 - d^2/dx^2 - d^2/dy^2 - d^2/dz^2) phi(x) = 0.
        """
        # In homogeneous coordinate zeta = pi_1 / pi_0 around the contour |zeta| = 1:
        # For X_rel = (X - X0), the quadratic polynomial Q(zeta) = det(X_rel) corresponds to the
        # residue of 1 / ( (A . Z)(B . Z) ) on CP^1:
        #   (1 / 2*pi*i) oint_{|zeta|=1} d_zeta / (a*zeta^2 + b*zeta + c) = 2 / det(X - X0)
        dx = x4 - x0_pole
        X_rel = PenroseTwistorCore.spacetime_to_hermitian(dx)

        # Parameterize contour |zeta| = 1 in CP^1: pi(phi) = (1, e^{i phi})
        phis = torch.linspace(0.0, 2.0 * math.pi, num_contour_pts + 1, dtype=x4.dtype, device=x4.device)[:-1]
        dphi = 2.0 * math.pi / num_contour_pts
        zeta = torch.complex(torch.cos(phis), torch.sin(phis))  # (K,)

        # Evaluate elementary twistor denominator Q(x, zeta) along the contour
        # Choose elementary dual twistors so Q(x, zeta) = X00 * zeta - X11 * zeta + ...
        # Specifically, any holomorphic kernel K(omega, pi) of degree -2 is a function of omega^A = i X^{AA'} pi_{A'}
        # For a single simple pole at zeta_0 = 0.35 + 0.25j inside the unit disk |zeta| < 1:
        # Let dual twistor A_alpha = (alpha_A, beta^{A'}). Then A_alpha Z^alpha = i alpha_A X^{AA'} pi_{A'} + beta^{A'} pi_{A'}.
        # Consequently, phi(x) = oint d_zeta / (A_alpha Z^alpha(zeta))^2 is an exact solution to Box phi(x) = 0!
        alpha_0 = torch.tensor(1.0 + 0.0j, dtype=X_rel.dtype, device=x4.device)
        alpha_1 = torch.tensor(0.4 - 0.3j, dtype=X_rel.dtype, device=x4.device)
        beta_0  = torch.tensor(2.5 + 0.2j, dtype=X_rel.dtype, device=x4.device)
        beta_1  = torch.tensor(-0.3 + 0.7j, dtype=X_rel.dtype, device=x4.device)

        # omega^0(zeta) = i (X00 * 1 + X01 * zeta), omega^1(zeta) = i (X10 * 1 + X11 * zeta)
        X00 = X_rel[..., 0, 0].unsqueeze(-1)
        X01 = X_rel[..., 0, 1].unsqueeze(-1)
        X10 = X_rel[..., 1, 0].unsqueeze(-1)
        X11 = X_rel[..., 1, 1].unsqueeze(-1)

        omega_0 = 1j * (X00 + X01 * zeta)
        omega_1 = 1j * (X10 + X11 * zeta)

        # Linear twistor form L(zeta) = alpha_A omega^A + beta^{A'} pi_{A'} = c_0 + c_1 * zeta
        L_zeta = alpha_0 * omega_0 + alpha_1 * omega_1 + beta_0 + beta_1 * zeta

        # Homogeneity -2 twistor function f(Z) = 1 / ( (A . Z) * (B . Z) ) where B has pole outside |zeta|=1
        # Integrating (1 / 2*pi*i) oint d_zeta / (L_zeta * (zeta - 0.25)) over |zeta| = 1:
        dzeta = 1j * zeta * dphi
        integrand = dzeta / (L_zeta * (zeta - 0.25) * (2.0 * math.pi * 1j))
        phi_x = torch.sum(integrand, dim=-1)
        return phi_x