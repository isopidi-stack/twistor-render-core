import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

device = torch.device("xpu" if (hasattr(torch, "xpu") and torch.xpu.is_available()) else "cpu")
gpu_name = torch.xpu.get_device_name(0) if device.type == "xpu" else "CPU"

def sync_device():
    if device.type == "xpu":
        torch.xpu.synchronize()

print(f"TwistorCore Zero-Sync XMX Engine initialized on: {device.type.upper()} ({gpu_name})", flush=True)

class TwistorZeroSyncEngine:
    def __init__(self, num_particles=8192, domain_size=110.0, radius=0.40, max_neighbors=12):
        self.N = num_particles
        self.L = domain_size
        self.radius = radius
        self.inv_L = 1.0 / domain_size
        self.K = max_neighbors

        self.gravity = torch.tensor([0.0, -34.0], device=device, dtype=torch.float32)
        self.obs_center = torch.tensor([0.0, -14.0], device=device, dtype=torch.float32)
        self.obs_radius = 13.0
        self.bounds = self.L * 0.46
        self.min_dist = 2.0 * self.radius
        self.cos_thresh = float(np.sqrt(1.0 - (self.min_dist / self.L) ** 2))

        # Twin colliding streams
        half = self.N // 2
        x1 = torch.empty(half, device=device).uniform_(-40.0, -8.0)
        y1 = torch.empty(half, device=device).uniform_(4.0, 48.0)
        vx1 = torch.full((half,), 14.0, device=device)
        vy1 = torch.full((half,), -12.0, device=device)

        rem = self.N - half
        x2 = torch.empty(rem, device=device).uniform_(8.0, 40.0)
        y2 = torch.empty(rem, device=device).uniform_(4.0, 48.0)
        vx2 = torch.full((rem,), -14.0, device=device)
        vy2 = torch.full((rem,), -12.0, device=device)

        self.pos = torch.cat([torch.stack([x1, y1], dim=1), torch.stack([x2, y2], dim=1)], dim=0)
        self.vel = torch.cat([torch.stack([vx1, vy1], dim=1), torch.stack([vx2, vy2], dim=1)], dim=0)

    def lift_to_cp2_fp16(self, pos: torch.Tensor) -> torch.Tensor:
        scaled = pos * self.inv_L
        norm_sq = torch.sum(scaled * scaled, dim=1, keepdim=True)
        inv_norm = torch.rsqrt(1.0 + norm_sq)
        return torch.cat([inv_norm, scaled * inv_norm], dim=1).to(torch.float16)

    @torch.no_grad()
    def step(self, dt=0.008):
        self.vel.add_(self.gravity, alpha=dt)
        self.pos.add_(self.vel, alpha=dt)

        # 1. Lift coordinates onto CP^2 chart in FP16
        Z = self.lift_to_cp2_fp16(self.pos)

        # 2. Fused XMX Matrix Multiply (Z @ Z.T) -> all-pairs Fubini-Study inner product
        cos_theta = torch.matmul(Z, Z.T)
        cos_theta.fill_diagonal_(-1.0)

        # 3. Zero-Sync Static Top-K Manifold Neighborhood (eliminates torch.nonzero CPU stall!)
        top_cos, top_idx = torch.topk(cos_theta, k=self.K, dim=1, largest=True, sorted=False)
        active = (top_cos.float() > self.cos_thresh).unsqueeze(2)  # (N, K, 1)

        # 4. Pure register-level PBD positional projection (no index_add_ atomics!)
        nbr_pos = self.pos[top_idx]  # (N, K, 2)
        diff = self.pos.unsqueeze(1) - nbr_pos
        dist = torch.norm(diff, dim=2, keepdim=True) + 1e-6
        overlap = torch.clamp(self.min_dist - dist, min=0.0) * active
        normals = diff / dist

        impulse = torch.sum(normals * overlap, dim=1) * 0.45
        self.pos.add_(impulse)
        self.vel.add_(impulse, alpha=(0.35 / dt))

        # 5. Central manifold obstacle
        rel_obs = self.pos - self.obs_center
        dist_obs = torch.norm(rel_obs, dim=1, keepdim=True) + 1e-6
        obs_mask = (dist_obs < (self.obs_radius + self.radius)).squeeze(1)
        if obs_mask.any():
            n_obs = rel_obs[obs_mask] / dist_obs[obs_mask]
            self.pos[obs_mask] = self.obs_center + n_obs * (self.obs_radius + self.radius)
            v_dot_n = torch.sum(self.vel[obs_mask] * n_obs, dim=1, keepdim=True)
            self.vel[obs_mask] -= 1.55 * torch.clamp(v_dot_n, max=0.0) * n_obs

        # 6. Domain walls & floor
        self.vel.mul_(0.9985)

        floor_mask = self.pos[:, 1] < -self.bounds
        self.pos[floor_mask, 1] = -self.bounds
        self.vel[floor_mask, 1] *= -0.45
        self.vel[floor_mask, 0] *= 0.94

        left_mask = self.pos[:, 0] < -self.bounds
        self.pos[left_mask, 0] = -self.bounds
        self.vel[left_mask, 0] *= -0.55

        right_mask = self.pos[:, 0] > self.bounds
        self.pos[right_mask, 0] = self.bounds
        self.vel[right_mask, 0] *= -0.55

def benchmark(n_particles, domain_size, radius):
    eng = TwistorZeroSyncEngine(num_particles=n_particles, domain_size=domain_size, radius=radius)
    for _ in range(25):
        eng.step()
    sync_device()

    n_iters = 300
    t0 = time.perf_counter()
    for _ in range(n_iters):
        eng.step()
    sync_device()
    ms = ((time.perf_counter() - t0) * 1000.0) / n_iters
    fps = 1000.0 / ms
    pairs_sec = (n_particles ** 2 * fps) / 1e9
    print(f"  [{n_particles:,} particles] -> {ms:.2f} ms/frame | {fps:.1f} FPS | {pairs_sec:.2f} B pairs/sec", flush=True)
    return fps, ms

print("\nINTEL ARC B580 ZERO-SYNC BENCHMARKS:", flush=True)
fps_8k, ms_8k = benchmark(8192, 110.0, 0.40)
fps_16k, ms_16k = benchmark(16384, 120.0, 0.35)

# Render showcase GIF using the 8,192-particle direct comparison run
os.makedirs("assets", exist_ok=True)
gif_path = os.path.join("assets", "twistor_b580_120fps.gif")
engine = TwistorZeroSyncEngine(num_particles=8192, domain_size=110.0, radius=0.40)

fig, ax = plt.subplots(figsize=(7, 7), dpi=110, facecolor="#090c10")
ax.set_facecolor("#090c10")
bounds = engine.L * 0.47
ax.set_xlim(-bounds, bounds)
ax.set_ylim(-bounds, bounds)
ax.set_xticks([])
ax.set_yticks([])

circle = plt.Circle((0, -14.0), 13.0, color="#1f2937", ec="#38bdf8", lw=1.5, zorder=2)
ax.add_patch(circle)

ax.set_title(
    f"TwistorCore CP2 Engine | 8,192 Particles @ {fps_8k:.0f} FPS (Intel Arc B580)",
    color="#f8fafc", fontsize=11, fontweight="bold", pad=10
)

pts = engine.pos.cpu().numpy()
speeds = torch.norm(engine.vel, dim=1).cpu().numpy()
scat = ax.scatter(pts[:, 0], pts[:, 1], c=speeds, cmap="turbo", s=4.5, alpha=0.92, vmin=0, vmax=45, zorder=3)

hud = ax.text(
    0.03, 0.96, "", transform=ax.transAxes, color="#38bdf8",
    fontsize=9, family="monospace", verticalalignment="top",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#0f172a", edgecolor="#1e293b", alpha=0.85)
)

def animate(frame_idx):
    for _ in range(3):
        engine.step()
    pts = engine.pos.cpu().numpy()
    speeds = torch.norm(engine.vel, dim=1).cpu().numpy()
    scat.set_offsets(pts)
    scat.set_array(speeds)
    hud.set_text(
        f"GPU: INTEL ARC B580 (XMX)\n"
        f"MANIFOLD: CP2 (FP16 ZERO-SYNC)\n"
        f"PARTICLES: 8,192 (vs his 8,000)\n"
        f"PHYSICS: {fps_8k:.1f} FPS ({ms_8k:.2f}ms)\n"
        f"16K SCALE: {fps_16k:.1f} FPS | GRID: NONE"
    )
    return scat, hud

print(f"\nRendering showcase GIF to {gif_path}...", flush=True)
ani = FuncAnimation(fig, animate, frames=90, interval=33, blit=False)
ani.save(gif_path, writer=PillowWriter(fps=30))
plt.close(fig)
print(f"Saved! Open {gif_path} to inspect the animation.", flush=True)
