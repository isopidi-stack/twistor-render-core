"""
Unit Test Suite for Twistor Render Core - Project Argon v6.0
Verifies CP^4 Laplacian spectrum, SIGMA_5 signature (3,2), GPO projector completeness,
strict Penrose incidence preservation at zero and non-zero origins, and ray-ordered depth occlusion.
"""

import torch
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from palatial_gpo_ray_optics import (
    SIGMA_5,
    CP4_CASIMIR_SPECTRUM,
    build_gpo_projectors,
    compute_gpo_probabilities,
    spatial_to_hermitian,
    spinor_to_unit_direction,
    compute_penrose_incidence_state,
    render_gpo_hud_frame,
)


def test_sigma5_indefinite_signature():
    """Verifies SIGMA_5 is Hermitian with eigenvalues {1, 1, 1, -1, -1} (signature 3,2)."""
    assert torch.allclose(SIGMA_5, SIGMA_5.conj().T)
    evals = torch.linalg.eigvalsh(SIGMA_5)
    expected = torch.tensor([-1.0, -1.0, 1.0, 1.0, 1.0], dtype=torch.float64)
    assert torch.allclose(torch.sort(evals).values, expected, atol=1e-12)


def test_cp4_casimir_eigenvalues():
    """Verifies the scalar Laplacian spectrum on CP^4 is 4*l*(l+4) = (0, 20, 48)."""
    n = 4
    computed = tuple(float(4 * l * (l + n)) for l in range(3))
    assert computed == (0.0, 20.0, 48.0)
    assert CP4_CASIMIR_SPECTRUM == (0.0, 20.0, 48.0)


def test_gpo_projector_completeness_and_orthogonality():
    """Verifies P_G + P_P + P_O = I_5, idempotency, and probability partition sum to 1."""
    P_G, P_P, P_O = build_gpo_projectors()
    I5 = torch.eye(5, dtype=torch.complex128)
    
    assert torch.allclose(P_G + P_P + P_O, I5, atol=1e-12)
    assert torch.allclose(P_G @ P_G, P_G, atol=1e-12)
    assert torch.allclose(P_P @ P_P, P_P, atol=1e-12)
    assert torch.allclose(P_O @ P_O, P_O, atol=1e-12)
    assert torch.allclose(P_G @ P_P, torch.zeros_like(I5), atol=1e-12)

    # Random batch of states in C^5
    torch.manual_seed(42)
    Z_rand = torch.randn(16, 5, dtype=torch.complex128)
    probs = compute_gpo_probabilities(Z_rand)
    assert torch.all(probs >= -1e-12)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(16, dtype=torch.float64), atol=1e-12)


def test_penrose_incidence_zero_and_nonzero_origin():
    """Verifies strict Penrose incidence omega = i X pi, including omega_cam == 0 at origin."""
    zero_origin = torch.zeros(4, 3, dtype=torch.float64)
    pi_spinor = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.7071, 0.7071j], [0.6, -0.8]], dtype=torch.complex128)
    phi_mass = torch.ones(4, 1, dtype=torch.complex128)

    res_zero = compute_penrose_incidence_state(zero_origin, pi_spinor, phi_mass)
    assert torch.allclose(res_zero["omega_phys"], torch.zeros_like(res_zero["omega_phys"]), atol=1e-12)
    assert torch.allclose(res_zero["incidence_defect"], torch.zeros(4, dtype=torch.float64), atol=1e-12)

    # Non-zero displacement
    x_pos = torch.tensor([[1.5, -0.5, 2.0], [-1.0, 3.0, 0.2], [0.0, 0.0, 5.0], [2.2, 1.1, -1.5]], dtype=torch.float64)
    res_nonzero = compute_penrose_incidence_state(x_pos, pi_spinor, phi_mass)
    assert torch.allclose(res_nonzero["incidence_defect"], torch.zeros(4, dtype=torch.float64), atol=1e-12)


def test_hud_preserves_physical_incidence_and_occludes_depth():
    """
    Verifies HUD pipeline preserves exact camera incidence in Z5_physical and correctly
    occludes a distant object behind a near object along the same ray without a Z-buffer.
    """
    cam_origin = torch.zeros(1, 3, dtype=torch.float64)
    # Spinor [1, 0] corresponds to unit direction +Z: (0, 0, 1)
    pi_spinor = torch.tensor([[1.0, 0.0]], dtype=torch.complex128)
    phi_mass = torch.tensor([[1.0]], dtype=torch.complex128)

    # Two primitives directly on the +Z ray at z = 2.0 (near, high opacity) and z = 8.0 (far)
    field_centers = torch.tensor([
        [0.0, 0.0, 8.0],  # Far object listed first to test longitudinal sorting
        [0.0, 0.0, 2.0],  # Near occluder listed second
    ], dtype=torch.float64)
    field_opacities = torch.tensor([0.9, 0.95], dtype=torch.float64)

    out = render_gpo_hud_frame(cam_origin, pi_spinor, field_centers, field_opacities, phi_mass)

    # 1. Physical omega at zero origin must still be identically zero
    assert torch.allclose(out["omega_physical"], torch.zeros(1, 2, dtype=torch.complex128), atol=1e-12)
    assert torch.allclose(out["incidence_defect"], torch.zeros(1, dtype=torch.float64), atol=1e-12)

    # 2. Near object (sorted index 0 at z=2.0) must dominate weight over occluded far object (z=8.0)
    weights = out["primitive_weights"][0]
    assert weights[0] > 0.90  # Near object receives full unoccluded transmittance
    assert weights[1] < 0.05  # Far object is attenuated by (1 - 0.95) = 0.05
    assert out["visible_depth"][0] < 2.5

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    for name, fn in sorted( list(globals().items()) ):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"[PASS] {name}")

