import math
import torch
from twistor.incidence import sl2c_inverse_dagger, EPSILON
from twistor.cohomology_penrose import TwistorCohomologyEngine

class PenroseTwistorCore:
    """
    Core SL(2,C) / SU(2,2) Twistor Algebra and Penrose Contour Transform:
      1. 4-Spinor Twistors Z^alpha = (omega^A, pi_{A'}) in C^4 with SU(2,2) metric Sigma_{alpha beta'}
      2. 4x4 Poincare SU(2,2) Transformations P(Lambda, T) and 4x4 Special Conformal Transformations K(b)
      3. Forward Tube PT^+ Positive-Frequency Splitting (Z^dagger Sigma Z > 0)
      4. Penrose Contour Integral Transform with verified pole separation |zeta_A(x)| < 1 < |zeta_B(x)|
    """
    @staticmethod
    def spacetime_to_hermitian(x4: torch.Tensor) -> torch.Tensor:
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
        dev = device or torch.device("cpu")
        Z2 = torch.zeros((2, 2), dtype=dtype, device=dev)
        I2 = torch.eye(2, dtype=dtype, device=dev)
        top = torch.cat([Z2, I2], dim=-1)
        bot = torch.cat([I2, Z2], dim=-1)
        return torch.cat([top, bot], dim=-2)

    @staticmethod
    def su22_inner_product(Z1: torch.Tensor, Z2: torch.Tensor) -> torch.Tensor:
        Sigma = PenroseTwistorCore.su22_metric(device=Z1.device, dtype=Z1.dtype)
        return torch.einsum("...a,ab,...b->...", torch.conj(Z1), Sigma, Z2)

    @staticmethod
    def build_poincare_su22_matrix(Lambda: torch.Tensor, translation_4d: torch.Tensor) -> torch.Tensor:
        T_herm = PenroseTwistorCore.spacetime_to_hermitian(translation_4d)
        L_inv_dag = sl2c_inverse_dagger(Lambda)
        upper_right = 1j * (T_herm @ L_inv_dag)
        zeros_lower = torch.zeros_like(Lambda)
        top = torch.cat([Lambda, upper_right], dim=-1)
        bot = torch.cat([zeros_lower, L_inv_dag], dim=-1)
        return torch.cat([top, bot], dim=-2)

    @staticmethod
    def build_special_conformal_su22_matrix(b_4d: torch.Tensor) -> torch.Tensor:
        """
        Constructs the exact 4x4 Special Conformal matrix K(b) in SU(2,2) for a 4-vector b_a:
            omega'^A = omega^A
            pi'_{A'} = pi_{A'} - i B_{A'A} omega^A
            K(b) = [[ I_2,       0   ],
                    [ -i B_herm, I_2 ]]
        Satisfies K(b)^dagger Sigma K(b) = Sigma identically because B_herm is Hermitian.
        """
        B_herm = PenroseTwistorCore.spacetime_to_hermitian(b_4d)
        I2 = torch.eye(2, dtype=B_herm.dtype, device=B_herm.device).expand_as(B_herm)
        Z2 = torch.zeros_like(B_herm)
        top = torch.cat([I2, Z2], dim=-1)
        bot = torch.cat([-1j * B_herm, I2], dim=-1)
        return torch.cat([top, bot], dim=-2)

    @staticmethod
    def lift_to_forward_tube(x4_real: torch.Tensor, timelike_eta: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
        X_herm = PenroseTwistorCore.spacetime_to_hermitian(x4_real)
        Y_herm = PenroseTwistorCore.spacetime_to_hermitian(timelike_eta)
        Z_complex = X_herm - 1j * Y_herm
        omega = 1j * torch.einsum("...ab,...b->...a", Z_complex, pi)
        return torch.cat([omega, pi], dim=-1)

    @staticmethod
    def penrose_contour_transform_scalar(x4: torch.Tensor, x0_origin_shift: torch.Tensor = None, num_contour_pts: int = 256) -> torch.Tensor:
        """
        Delegates to TwistorCohomologyEngine.contour_integral_transform, which enforces
        strict pole separation |zeta_A(x)| < 1 < |zeta_B(x)| so phi(x) is genuinely non-zero.
        """
        dx = x4 - x0_origin_shift if x0_origin_shift is not None else x4
        return TwistorCohomologyEngine.contour_integral_transform(dx, num_pts=num_contour_pts)