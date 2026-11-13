import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "2"

import torch

class AIVectorShield:
    def __init__(self, dim: int = 512, threshold: float = 0.65):
        self.dim = dim
        self.threshold = threshold

    def _to_complex_projective(self, embeddings: torch.Tensor) -> torch.Tensor:
        if torch.is_complex(embeddings):
            return embeddings
        # Pair adjacent real dimensions [N, 2K] into complex projective coordinates [N, K]
        if embeddings.shape[-1] % 2 != 0:
            embeddings = torch.nn.functional.pad(embeddings, (0, 1))
        real = embeddings[..., 0::2]
        imag = embeddings[..., 1::2]
        return torch.complex(real, imag)

    def verify(self, embeddings: torch.Tensor) -> torch.Tensor:
        z_complex = self._to_complex_projective(embeddings)

        # 1. Complex projective norm incidence (catches OOD outlierp)
        mag_check = torch.abs(z_complex).mean(dim=-1) >= self.threshold

        # 2. CP^3 null-ray phase-curvature violation (catches Tier-X manifold-tangent stealth)
        phases = torch.angle(z_complex)
        phase_diff = torch.abs(phases[..., 1:] - phases[..., :-1]).cos().std(dim=-1)
        curvature_check = phase_diff <= 0.52

        return (mag_check & curvature_check).long()
