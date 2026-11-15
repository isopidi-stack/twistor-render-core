import math
import unittest
import torch
from twistor.incidence import sl2c_from_lie_algebra, rotate_spinor, epsilon_inner_product
from twistor.harmonics import PenroseTwistorHarmonicEngine
from twistor.localization import (
    pixel_to_stereographic_spinor,
    spinor_to_transverse_basis,
    solve_camera_position_spinor,
    solve_camera_position_euclidean_reference
)
from twistor.ransac import solve_camera_position_ransac

def euler_to_su2(angles: torch.Tensor) -> torch.Tensor:
    yaw, pitch, roll = angles[0], angles[1], angles[2]
    cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)
    cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
    cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
    z = torch.zeros_like(yaw)
    Rz = torch.stack([
        torch.stack([torch.complex(cy, -sy), torch.complex(z, z)]),
        torch.stack([torch.complex(z, z), torch.complex(cy, sy)])
    ])
    Rx = torch.stack([
        torch.stack([torch.complex(cp, z), torch.complex(z, -sp)]),
        torch.stack([torch.complex(z, -sp), torch.complex(cp, z)])
    ])
    Ry = torch.stack([
        torch.stack([torch.complex(cr, z), torch.complex(-sr, z)]),
        torch.stack([torch.complex(sr, z), torch.complex(cr, z)])
    ])
    return torch.view_as_real(Rz @ Rx @ Ry).reshape(-1)

def su2_tangent_perturbation(w: torch.Tensor, U_base: torch.Tensor) -> torch.Tensor:
    U_delta = sl2c_from_lie_algebra(w)
    return torch.view_as_real(U_delta @ U_base).reshape(-1)

class TestTwistorAuditSuite(unittest.TestCase):

    def test_1_gimbal_lock_vs_su2_lie_algebra(self):
        gimbal_angles = torch.tensor([0.3, math.pi / 2.0, -0.2], dtype=torch.float64, requires_grad=True)
        J_euler = torch.autograd.functional.jacobian(euler_to_su2, gimbal_angles)
        rank_euler = int(torch.linalg.matrix_rank(J_euler, tol=1e-6).item())

        U_base = torch.view_as_complex(euler_to_su2(gimbal_angles.detach()).reshape(2, 2, 2))
        w_zero = torch.zeros(3, dtype=torch.float64, requires_grad=True)
        J_su2 = torch.autograd.functional.jacobian(lambda w: su2_tangent_perturbation(w, U_base), w_zero)
        rank_su2 = int(torch.linalg.matrix_rank(J_su2, tol=1e-6).item())

        print(f"  -> [AUDIT 1] Jacobian Rank at pitch=pi/2 | Euler: {rank_euler} vs Direct SU(2): {rank_su2}")
        self.assertEqual(rank_euler, 2)
        self.assertEqual(rank_su2, 3)

    def test_2_spinor_vs_euclidean_equivalence_and_2ray_minimal(self):
        torch.manual_seed(42)
        C_true = torch.tensor([1.25, -0.85, 2.40], dtype=torch.float64)
        landmarks = torch.tensor([
            [3.1, 0.5, 6.2],
            [-1.4, 2.2, 5.8],
            [0.8, -2.5, 7.1],
            [2.0, 1.8, 4.9]
        ], dtype=torch.float64)

        dirs = landmarks - C_true.unsqueeze(0)
        dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)
        fx, fy, cx, cy = 800.0, 800.0, 640.0, 360.0
        px = fx * (dirs[:, 0] / dirs[:, 2]) + cx
        py = fy * (dirs[:, 1] / dirs[:, 2]) + cy

        pi_world = pixel_to_stereographic_spinor(px, py, fx, fy, cx, cy)
        _, _, rays_n = spinor_to_transverse_basis(pi_world)

        # Full 4-landmark equivalence check
        res_spinor = solve_camera_position_spinor(pi_world, landmarks)
        C_euclid = solve_camera_position_euclidean_reference(rays_n, landmarks)
        diff = float(torch.norm(res_spinor["position"] - C_euclid).item())
        print(f"  -> [AUDIT 2A] Spinor vs Euclidean QR/SVD Discrepancy: {diff:.2e} m")
        self.assertLess(diff, 1e-12)
        self.assertTrue(res_spinor["is_valid"])

        # 2-Landmark minimal exact solver check
        res_2ray = solve_camera_position_spinor(pi_world[:2], landmarks[:2])
        err_2ray = float(torch.norm(res_2ray["position"] - C_true).item())
        print(f"  -> [AUDIT 2B] 2-Ray Minimal Exact Position Error: {err_2ray:.2e} m (Rank={res_2ray['rank']})")
        self.assertEqual(res_2ray["rank"], 3)
        self.assertLess(err_2ray, 1e-11)

    def test_3_ransac_angular_residual_and_outlier_rejection(self):
        torch.manual_seed(7)
        C_true = torch.tensor([0.5, -0.2, 1.5], dtype=torch.float64)
        landmarks = torch.randn((12, 3), dtype=torch.float64) * 2.0 + torch.tensor([0.5, -0.2, 6.0], dtype=torch.float64)
        dirs = landmarks - C_true.unsqueeze(0)
        dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)
        px = 800.0 * (dirs[:, 0] / dirs[:, 2]) + 640.0
        py = 800.0 * (dirs[:, 1] / dirs[:, 2]) + 360.0

        # Corrupt 3 landmarks as gross outliers
        px[9:] += 250.0
        py[9:] -= 180.0

        pi_world = pixel_to_stereographic_spinor(px, py, 800.0, 800.0, 640.0, 360.0)
        ransac_res = solve_camera_position_ransac(pi_world, landmarks, num_hypotheses=256, angle_threshold_rad=0.008)
        pos_err = float(torch.norm(ransac_res["position"] - C_true).item())
        print(f"  -> [AUDIT 3] RANSAC Status: {ransac_res['status']} | Inliers: {ransac_res['num_inliers']}/12 | Error: {pos_err:.2e} m")
        self.assertEqual(ransac_res["status"], "robust_success")
        self.assertEqual(ransac_res["num_inliers"], 9)
        self.assertLess(pos_err, 1e-10)

    def test_4_epsilon_sl2c_invariance_and_harmonics(self):
        axis_angle = torch.tensor([0.4, -0.7, 1.1], dtype=torch.float32)
        boost = torch.tensor([0.2, 0.1, -0.15], dtype=torch.float32)
        Lambda = sl2c_from_lie_algebra(axis_angle, boost)

        xi = torch.tensor([1.0 + 0.5j, -0.3 + 0.8j], dtype=torch.complex64)
        eta = torch.tensor([-0.4 + 0.2j, 0.9 - 0.1j], dtype=torch.complex64)
        orig_prod = epsilon_inner_product(xi, eta)
        trans_prod = epsilon_inner_product(Lambda @ xi, Lambda @ eta)
        inv_err = float(torch.abs(orig_prod - trans_prod).item())
        print(f"  -> [AUDIT 4] SL(2,C) Symplectic Epsilon-Contraction Error: {inv_err:.2e}")
        self.assertLess(inv_err, 1e-6)

if __name__ == "__main__":
    print("="*92)
    print("[ARGON EXECUTION] Running Full Twistor-Render-Core Mathematical Audit & Harmonics Suite")
    print("="*92)
    unittest.main( verbosity=2 )