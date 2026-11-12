"""
SU(2,2) AutoDiff Inverse Rendering Demo
=======================================
Optimizes unknown camera position and yaw orientation via gradient descent 
by minimizing pixel-wise Mean Squared Error through the Twistor incidence kernel.
"""

import sys
import os
import torch
import torch.optim as optim

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import twistor.renderer as twistor_r


def render_with_pose(points, width, height, threshold, cam_pos, yaw):
    """Helper to render frame given a dynamic camera position and yaw with correct broadcasting."""
    device = points.device
    R_A, I_A, R_D, I_D = twistor_r.generate_su22_camera_matrix(yaw=yaw, pitch=0.0, boost_v=0.0, zoom=1.0, device=device)
    
    c_tensor = cam_pos
    R_X, I_X = twistor_r.points_to_real_hermitian(points) # Shape: [P, 2, 2]
    r_omega, i_omega, r_pi, i_pi = twistor_r.generate_transformed_ray_spinors(
        width, height, c_tensor, R_A, I_A, R_D, I_D, device=device
    ) # Shapes: [R, 2]

    # Expand dimensions for broadcasted batch matrix multiplication: [R, P, 2]
    R_X_exp = R_X.unsqueeze(0)  # [1, P, 2, 2]
    I_X_exp = I_X.unsqueeze(0)  # [1, P, 2, 2]
    
    r_pi_exp = r_pi.unsqueeze(1).unsqueeze(-1)  # [R, 1, 2, 1]
    i_pi_exp = i_pi.unsqueeze(1).unsqueeze(-1)  # [R, 1, 2, 1]

    pred_r_omega = (-torch.matmul(I_X_exp, r_pi_exp) - torch.matmul(R_X_exp, i_pi_exp)).squeeze(-1) # [R, P, 2]
    pred_i_omega = (torch.matmul(R_X_exp, r_pi_exp) - torch.matmul(I_X_exp, i_pi_exp)).squeeze(-1) # [R, P, 2]

    r_omega_exp = r_omega.unsqueeze(1) # [R, 1, 2]
    i_omega_exp = i_omega.unsqueeze(1) # [R, 1, 2]

    diff_r = r_omega_exp - pred_r_omega
    diff_i = i_omega_exp - pred_i_omega

    sq_residual = torch.sum(diff_r ** 2 + diff_i ** 2, dim=-1) # [R, P] sum over spinor components
    min_sq_res, _ = torch.min(sq_residual, dim=-1)             # [R] min over all points
    
    intensity = torch.clamp(1.0 - (torch.sqrt(min_sq_res) / threshold), min=0.0, max=1.0)
    return intensity.reshape(height, width)


def main():
    device = "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu"
    print(f"Running AutoDiff Inverse Rendering Demo on {device.upper()}...\n")

    # 1. Create a target spherical point cloud
    phi = torch.linspace(0.2, 2.8, 30)
    theta = torch.linspace(0, 2 * 3.14159, 35)
    grid_phi, grid_theta = torch.meshgrid(phi, theta, indexing="ij")
    
    x = 0.4 * torch.sin(grid_phi) * torch.cos(grid_theta)
    y = 0.4 * torch.sin(grid_phi) * torch.sin(grid_theta)
    z = 0.4 * torch.cos(grid_phi) + 1.2
    pts = torch.stack([x, y, z], dim=-1).reshape(-1, 3).to(device)
    threshold = 2.4

    # 2. Ground Truth Camera Pose
    true_cam_pos = torch.tensor([0.2, -0.1, -2.0], dtype=torch.float32, device=device)
    true_yaw = 0.35  # ~0.35 radians rotation
    
    print(f"Target Ground Truth Camera Position: {true_cam_pos.tolist()}, Yaw: {true_yaw:.3f}")
    with torch.no_grad():
        target_image = render_with_pose(pts, 16, 16, threshold, true_cam_pos, true_yaw)

    # 3. Initial Guess Camera Pose (perturbed away from ground truth)
    est_cam_pos = torch.tensor([0.0, 0.0, -2.2], dtype=torch.float32, device=device, requires_grad=True)
    est_yaw = torch.tensor(0.0, dtype=torch.float32, device=device, requires_grad=True)

    optimizer = optim.Adam([est_cam_pos, est_yaw], lr=0.05)

    print("\nStarting Autodiff Optimization Loop...")
    for epoch in range(61):
        optimizer.zero_grad()
        
        pred_image = render_with_pose(pts, 16, 16, threshold, est_cam_pos, est_yaw)
        
        # Mean Squared Error loss between target and estimated frames
        loss = torch.mean((pred_image - target_image) ** 2)
        
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch:02d} | Loss: {loss.item():.6f} | Est Pos: [{est_cam_pos[0].item():.3f}, {est_cam_pos[1].item():.3f}, {est_cam_pos[2].item():.3f}] | Est Yaw: {est_yaw.item():.3f}")

    print("\nOptimization Complete!")
    print(f"Target -- Pos: {true_cam_pos.tolist()}, Yaw: {true_yaw:.3f}")
    print(f"Learned -- Pos: {[round(p, 3) for p in est_cam_pos.tolist()]} , Yaw: {est_yaw.item():.3f}")
    print("AutoDiff Camera Optimization Test Successful!")


if __name__ == "__main__":
    main()