import math
import torch

class PenroseTwistorHarmonicEngine:
    """
    Evaluates harmonic eigenmodes anchored to Penrose's positive-frequency splitting 
    and conformal field postulates, isolating points of lowest energy storage 
    (det(X^{AA'}) -> 0 null-cone constraints) across CP^3 spinor manifolds.
    """
    def __init__(self, device=None):
        self.device = device or torch.device("cpu")
        self.omega_0 = 2.0 * math.pi * 1.0
        self.ground_state_weight = 1.0 / math.sqrt(4.0 * math.pi)

    def compute_penrose_lowest_energy_modes(self, X_hermitian: torch.Tensor) -> torch.Tensor:
        X00 = X_hermitian[..., 0, 0]
        X11 = X_hermitian[..., 1, 1]
        X01 = X_hermitian[..., 0, 1]
        X10 = X_hermitian[..., 1, 0]

        trace_X = (X00 + X11).real
        det_X = (X00 * X11 - X01 * X10).real

        # Conformal energy storage functional minimized on the null cone
        energy_storage = torch.clamp(torch.abs(trace_X**2 - 4.0 * det_X), min=1e-12)
        harmonic_frequencies = self.omega_0 * torch.sqrt(energy_storage / (torch.mean(energy_storage) + 1e-6))
        return harmonic_frequencies

    def evaluate_penrose_harmonic_shading(self, normals: torch.Tensor, frequencies: torch.Tensor) -> torch.Tensor:
        nz = normals[..., 2]
        monopole = self.ground_state_weight * torch.ones_like(nz)
        dipole = 0.488603 * nz * torch.cos(frequencies)
        field_response = torch.clamp(monopole + dipole, 0.0, 1.0)
        return torch.stack([field_response, field_response * 0.95, field_response * 1.05], dim=-1)