"""
Read-only compatibility diagnostic comparing the frozen legacy palatial_gpo_ray_optics.py
against the new twistor.congruence reference module. Does NOT modify any files.
"""

from __future__ import annotations

import hashlib
import math
import pathlib
import sys
import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import palatial_gpo_ray_optics as legacy_gpo
from twistor import congruence as cg

EXPECTED_LEGACY_SHA256 = "8809F3AF1D5BBB43202128C740306402D9BFF264911146CB5DB974EBF9716217"


def check_frozen_hash() -> str:
    gpo_path = REPO_ROOT / "palatial_gpo_ray_optics.py"
    digest = hashlib.sha256(gpo_path.read_bytes()).hexdigest().upper()
    assert digest == EXPECTED_LEGACY_SHA256, f"Frozen file hash changed: {digest}"
    return digest


def controlled_kz_sign_experiment(helicity_shift: float = 0.85) -> dict[str, float]:
    """
    Runs the exact same [0.25, 0.85]^2 x [0.20, 0.80] grid and 4th-order stencil as
    legacy_gpo.verify_kerr_theorem_shear_free_pde(), flipping ONLY kz to (1 - |zeta|^2)/(1 + |zeta|^2)
    with the matching Theorem 1 southern screen vector m_S(zeta).
    """
    b = float(helicity_shift)
    gx = torch.linspace(0.25, 0.85, 9, dtype=torch.float64)
    gy = torch.linspace(0.25, 0.85, 9, dtype=torch.float64)
    gz = torch.linspace(0.20, 0.80, 9, dtype=torch.float64)
    X, Y, Z = torch.meshgrid(gx, gy, gz, indexing="ij")

    def solve_zeta(x, y, z):
        a_coeff = (x - 1j * y).to(torch.complex128)
        b_coeff = (z + 1j * b).to(torch.complex128)
        c_coeff = -(x + 1j * y).to(torch.complex128)
        return (-b_coeff + torch.sqrt(b_coeff ** 2 - a_coeff * c_coeff)) / a_coeff

    def eval_k_fixed(x, y, z):
        zt = solve_zeta(x, y, z)
        z_abs2 = torch.abs(zt) ** 2
        denom = 1.0 + z_abs2
        return torch.stack([2.0 * torch.real(zt) / denom, 2.0 * torch.imag(zt) / denom, (1.0 - z_abs2) / denom], dim=-1)

    h = 1e-5

    def diff_4th(fp1, fm1, fp2, fm2):
        return (-fp2 + 8.0 * fp1 - 8.0 * fm1 + fm2) / (12.0 * h)

    zeta_0 = solve_zeta(X, Y, Z)
    dk_dx = diff_4th(eval_k_fixed(X + h, Y, Z), eval_k_fixed(X - h, Y, Z), eval_k_fixed(X + 2 * h, Y, Z), eval_k_fixed(X - 2 * h, Y, Z))
    dk_dy = diff_4th(eval_k_fixed(X, Y + h, Z), eval_k_fixed(X, Y - h, Z), eval_k_fixed(X, Y + 2 * h, Z), eval_k_fixed(X, Y - 2 * h, Z))
    dk_dz = diff_4th(eval_k_fixed(X, Y, Z + h), eval_k_fixed(X, Y, Z - h), eval_k_fixed(X, Y, Z + 2 * h), eval_k_fixed(X, Y, Z - 2 * h))
    B_jac = torch.stack([dk_dx, dk_dy, dk_dz], dim=-1).to(torch.complex128)

    m_s = cg.screen_vector_south(zeta_0)
    rho_fixed = torch.einsum("...a,...b,...ab->...", torch.conj(m_s), m_s, B_jac)
    sigma_fixed = torch.einsum("...a,...b,...ab->...", m_s, m_s, B_jac)

    pos_grid = torch.stack([X, Y, Z], dim=-1)
    w_grid = cg.principal_w(pos_grid, b, sheet_sign=+1.0)
    rho_analytic = cg.optical_scalar(w_grid)
    k_ref = cg.tangent_field(pos_grid, w_grid, b)
    k_fixed = eval_k_fixed(X, Y, Z)

    return {
        "max_sigma_after_kz_fix": float(torch.max(torch.abs(sigma_fixed)).item()),
        "max_rho_err_vs_1_over_conj_w": float(torch.max(torch.abs(rho_fixed - rho_analytic)).item()),
        "max_k_diff_vs_congruence_ref": float(torch.max(torch.linalg.norm(k_fixed - k_ref, dim=-1)).item()),
    }


def main() -> None:
    digest = check_frozen_hash()
    print(f"[OK] Frozen palatial_gpo_ray_optics.py SHA-256 intact: {digest}")

    legacy_res = legacy_gpo.verify_kerr_theorem_shear_free_pde(helicity_shift=0.85)
    print("\n=== 1. Legacy palatial_gpo_ray_optics.verify_kerr_theorem_shear_free_pde() ===")
    for k, v in legacy_res.items():
        print(f"  {k}: {v}")

    fixed_res = controlled_kz_sign_experiment(helicity_shift=0.85)
    print("\n=== 2. Controlled Sign-Correction Check (flipping ONLY kz & paired m_S on identical grid) ===")
    for k, v in fixed_res.items():
        print(f"  {k}: {v:.3e}")

    # 3. z-axis singularity comparison (x = y = 0)
    b = 0.85
    z_axis_pt = torch.tensor([0.0, 0.0, 0.5], dtype=torch.float64)
    a_c = torch.complex(z_axis_pt[0], -z_axis_pt[1])
    b_c = torch.complex(z_axis_pt[2], torch.tensor(b, dtype=torch.float64))
    c_c = -torch.complex(z_axis_pt[0], z_axis_pt[1])
    legacy_z_zeta = (-b_c + torch.sqrt(b_c ** 2 - a_c * c_c)) / a_c
    ref_w = cg.principal_w(z_axis_pt, b, sheet_sign=+1.0)
    ref_k = cg.tangent_field(z_axis_pt, ref_w, b)

    print("\n=== 3. z-Axis Regularity Comparison (x=0, y=0, z=0.5) ===")
    print(f"  Legacy solve_zeta(0, 0, 0.5): {complex(legacy_z_zeta.item())}")
    print(f"  twistor.congruence k(0, 0, 0.5): {ref_k.tolist()} (finite={bool(torch.isfinite(ref_k).all().item())})")


if __name__ == "__main__":
    main()