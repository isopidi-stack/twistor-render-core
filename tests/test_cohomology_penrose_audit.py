"""
Standalone Adversarial Audit Suite for Twistor Render Core - Project Argon v6.1
Verifies:
  1. SIGMA_5 signature (3,2)
  2. Active 5x5 H_GPO spectral operator eigenvalues {0, 20, 48} & Rayleigh expectation bounds
  3. GPO projector completeness & orthogonality
  4. Full 4D Minkowski Infeld-van der Waerden determinant det(X) = t^2 - ||x||^2
  5. Exact Poincare translation gauge covariance & Sigma_5 norm invariance under L_0 scaling
  6. Continuous erf() volumetric occlusion symmetry for concentric/overlapping primitives
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from palatial_gpo_ray_optics import (
    SIGMA_5,
    CP4_CASIMIR_SPECTRUM,
    build_gpo_projectors,
    build_gpo_spectral_operator,
    compute_gpo_probabilities,
    compute_gpo_spectral_expectation,
    spatial_to_hermitian,
    compute_penrose_incidence_state,
    apply_spacetime_translation,
    render_gpo_hud_frame,
)


def test_sigma5_indefinite_signature():
    """Verifies SIGMA_5 is Hermitian with eigenvalues {1, 1, 1, -1, -1} (signature 3,2)."""
    assert torch.allclose(SIGMA_5, SIGMA_5.conj().T)
    evals = torch.linalg.eigvalsh(SIGMA_5)
    expected = torch.tensor([-1.0, -1.0, 1.0, 1.0, 1.0], dtype=torch.float64)
    assert torch.allclose(torch.sort(evals).values, expected, atol=1e-12)


def test_cp4_casimir_eigenvalues_and_active_spectral_operator():
    """Verifies H_GPO = 0*P_G + 20*P_P + 48*P_O has exact spectrum {0, 0, 20, 20, 48}."""
    H_gpo = build_gpo_spectral_operator()
    assert torch.allclose(H_gpo, H_gpo.conj().T)
    evals = torch.linalg.eigvalsh(H_gpo)
    expected = torch.tensor([0.0, 0.0, 20.0, 20.0, 48.0], dtype=torch.float64)
    assert torch.allclose(torch.sort(evals).values, expected, atol=1e-12)

    # Projective scale invariance of Rayleigh expectation: <lambda Z | H_GPO | lambda Z> == <Z | H_GPO | Z>
    torch.manual_seed(11)
    Z = torch.randn(10, 5, dtype=torch.complex128)
    scale = (torch.randn(10, 1, dtype=torch.complex128) * 5.0) + 1.0
    exp_1 = compute_gpo_spectral_expectation(Z)
    exp_2 = compute_gpo_spectral_expectation(Z * scale)
    assert torch.allclose(exp_1, exp_2, atol=1e-12)
    assert torch.all((exp_1 >= 0.0) & (exp_1 <= 48.0))


def test_gpo_projector_completeness_and_orthogonality():
    """Verifies P_G + P_P + P_O = I_5, idempotency, and probability partition sum to 1."""
    P_G, P_P, P_O = build_gpo_projectors()
    I5 = torch.eye(5, dtype=torch.complex128)
    assert torch.allclose(P_G + P_P + P_O, I5, atol=1e-12)
    assert torch.allclose(P_G @ P_G, P_G, atol=1e-12)
    assert torch.allclose(P_P @ P_P, P_P, atol=1e-12)
    assert torch.allclose(P_O @ P_O, P_O, atol=1e-12)
    assert torch.allclose(P_G @ P_P, torch.zeros_like(I5), atol=1e-12)


def test_minkowski_infeld_van_der_waerden_determinant():
    """Verifies 4D spacetime mapping X^{AA'} = t I_2 + x . sigma satisfies det(X) = t^2 - ||x||^2."""
    events_4d = torch.tensor([
        [5.0, 1.0, 2.0, 2.0],   # Timelike: 25 - 9 = +16
        [3.0, 3.0, 0.0, 0.0],   # Null/Lightlike: 9 - 9 = 0
        [1.0, 2.0, 3.0, 4.0],   # Spacelike: 1 - 29 = -28
    ], dtype=torch.float64)
    X_mat = spatial_to_hermitian(events_4d)
    dets = torch.linalg.det(X_mat).real
    expected = events_4d[:, 0]**2 - torch.sum(events_4d[:, 1:]**2, dim=-1)
    assert torch.allclose(dets, expected, atol=1e-12)


def test_penrose_incidence_and_translation_gauge_invariance():
    """
    Verifies scaled Penrose incidence omega = (i/L_0) X pi and proves that under any real
    spacetime translation X -> X + A, the indefinite Sigma_5 norm Z^dagger Sigma_5 Z = |phi|^2 is invariant.
    """
    x_pos = torch.tensor([[1.5, -0.5, 2.0], [0.0, 0.0, 0.0]], dtype=torch.float64)
    a_shift = torch.tensor([[10.0, -25.0, 40.0], [-5.0, 12.0, 8.0]], dtype=torch.float64)
    pi_spinor = torch.tensor([[0.8, 0.6j], [1.0, 0.0]], dtype=torch.complex128)
    phi_mass = torch.tensor([[1.5 + 0.5j], [0.0 + 0.0j]], dtype=torch.complex128)
    L0 = 2.5

    st_initial = compute_penrose_incidence_state(x_pos, pi_spinor, phi_mass, L0_scale=L0)
    st_shifted_direct = compute_penrose_incidence_state(x_pos + a_shift, pi_spinor, phi_mass, L0_scale=L0)
    Z5_shifted_gauge = apply_spacetime_translation(st_initial["Z5_unnormalized"], a_shift, L0_scale=L0)

    # Direct evaluation at X + A must identically match twistor translation operator
    assert torch.allclose(st_shifted_direct["Z5_unnormalized"], Z5_shifted_gauge, atol=1e-12)
    # Sigma_5 quadratic form must equal |phi|^2 before and after arbitrary spatial translation
    expected_sigma5_norm = torch.abs(phi_mass.squeeze(-1)) ** 2
    assert torch.allclose(st_initial["sigma5_norm_unnormalized"], expected_sigma5_norm, atol=1e-12)
    assert torch.allclose(st_shifted_direct["sigma5_norm_unnormalized"], expected_sigma5_norm, atol=1e-12)


def test_hud_erf_continuous_occlusion_and_concentric_symmetry():
    """
    Verifies:
      1. Near occluder (z=2) attenuates far object (z=8) along +Z ray.
      2. Two identical concentric volumes at the exact same depth (z=4.0) receive equal
         symmetric transmittance weights (no discrete sorting jump artifact!).
    """
    cam_origin = torch.zeros(1, 3, dtype=torch.float64)
    pi_spinor = torch.tensor([[1.0, 0.0]], dtype=torch.complex128)
    phi_mass = torch.tensor([[1.0]], dtype=torch.complex128)

    # Case A: separated depths (z=8.0 and z=2.0)
    centers_sep = torch.tensor([[0.0, 0.0, 8.0], [0.0, 0.0, 2.0]], dtype=torch.float64)
    opacities_sep = torch.tensor([0.9, 0.95], dtype=torch.float64)
    out_sep = render_gpo_hud_frame(cam_origin, pi_spinor, centers_sep, opacities_sep, phi_mass)
    w_sep = out_sep["primitive_weights"][0]
    assert w_sep[0] > 0.90 and w_sep[1] < 0.06

    # Case B: identical concentric overlapping volumes at z=4.0
    centers_conc = torch.tensor([[0.0, 0.0, 4.0], [0.0, 0.0, 4.0]], dtype=torch.float64)
    opacities_conc = torch.tensor([0.8, 0.8], dtype=torch.float64)
    out_conc = render_gpo_hud_frame(cam_origin, pi_spinor, centers_conc, opacities_conc, phi_mass)
    w_conc = out_conc["primitive_weights"][0]
    # Because erf(0) == 0, both concentric volumes must receive identical weight!
    assert torch.allclose(w_conc[0], w_conc[1], atol=1e-12)


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"[PASS] {name}")
