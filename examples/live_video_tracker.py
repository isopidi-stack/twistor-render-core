"""
Real-Time 6-DoF Live Webcam / FPV Drone Video 3D Tracker for twistor-vision
===========================================================================
Combines Essential-matrix visual attitude estimation (SO(3) -> SU(2) spinor R_A)
with GPU-batched twistor.localize_ransac() to track full 6-DoF drone flight
(roll, pitch, yaw + 3D X, Y, Z position) from:
  - A real FPV drone video file (--source path/to/fpv_flight.mp4)
  - A live webcam (--source 0 or --source 1)
  - Built-in 6-DoF banked 3D drone FPV simulator (--source demo)

Controls:
  R - Reset 3D map & origin at current camera view
  Q / ESC - Quit (and finalize MP4 recording if --save-video is set)
"""

import argparse
import os
import time
import cv2
import numpy as np
import torch
import twistor


def quat_wxyz_to_su2(q_wxyz: torch.Tensor) -> torch.Tensor:
    q = q_wxyz / torch.linalg.norm(q_wxyz, dim=-1, keepdim=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    u00 = torch.complex(w, z)
    u01 = torch.complex(y, x)
    u10 = torch.complex(-y, x)
    u11 = torch.complex(w, -z)
    return torch.stack([
        torch.stack([u00, u01], dim=-1),
        torch.stack([u10, u11], dim=-1),
    ], dim=-2).to(torch.complex64)


def rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Converts a 3x3 SO(3) rotation matrix into a unit quaternion [w, x, y, z]."""
    tr = float(np.trace(R))
    if tr > 0.0:
        s = 0.5 / np.sqrt(tr + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float32)
    return q / np.linalg.norm(q)


def rotmat_to_rpy_deg(R_wc: np.ndarray) -> tuple[float, float, float]:
    """Extracts camera roll, pitch, yaw in degrees from world-to-camera rotation."""
    R_cw = R_wc.T
    sy = np.sqrt(R_cw[0, 0] ** 2 + R_cw[1, 0] ** 2)
    if sy > 1e-6:
        roll = np.arctan2(R_cw[2, 1], R_cw[2, 2])
        pitch = np.arctan2(-R_cw[2, 0], sy)
        yaw = np.arctan2(R_cw[1, 0], R_cw[0, 0])
    else:
        roll = np.arctan2(-R_cw[1, 2], R_cw[1, 1])
        pitch = np.arctan2(-R_cw[2, 0], sy)
        yaw = 0.0
    return float(np.degrees(roll)), float(np.degrees(pitch)), float(np.degrees(yaw))


class SyntheticDroneFPVFeed:
    """Generates a real-time textured 6-DoF banked flight video feed with moving outlier clutter."""

    def __init__(self, width: int = 960, height: int = 600):
        self.w = width
        self.h = height
        self.t = 0.0
        rng = np.random.default_rng(42)
        canvas = np.full((2200, 2200, 3), (26, 32, 40), dtype=np.uint8)
        for gx in range(0, 2200, 80):
            cv2.line(canvas, (gx, 0), (gx, 2200), (55, 68, 82), 2)
        for gy in range(0, 2200, 80):
            cv2.line(canvas, (0, gy), (2200, gy), (55, 68, 82), 2)
        for _ in range(520):
            cx, cy = rng.integers(120, 2080, size=2)
            rad = int(rng.integers(8, 28))
            col = tuple(int(c) for c in rng.integers(90, 240, size=3))
            if rng.random() > 0.5:
                cv2.circle(canvas, (int(cx), int(cy)), rad, col, -1)
            else:
                cv2.rectangle(canvas, (int(cx - rad), int(cy - rad)), (int(cx + rad), int(cy + rad)), col, 2)
        self.canvas = canvas

    def read(self):
        self.t += 0.028
        dx = 150.0 * np.sin(self.t)
        dy = 100.0 * np.sin(2.0 * self.t)
        bank_deg = 12.0 * np.sin(self.t * 1.1)
        scale = 1.0 + 0.14 * np.cos(self.t * 1.3)

        center = (1100.0 + dx, 1100.0 + dy)
        M_rot = cv2.getRotationMatrix2D(center, bank_deg, scale)
        M_rot[0, 2] += (self.w * 0.5) - center[0]
        M_rot[1, 2] += (self.h * 0.5) - center[1]
        frame = cv2.warpAffine(self.canvas, M_rot, (self.w, self.h), flags=cv2.INTER_LINEAR)

        # Inject moving foreground obstacle (dynamic outlier generator)
        ox = int(self.w * 0.5 + 260 * np.cos(self.t * 2.4))
        oy = int(self.h * 0.5 + 140 * np.sin(self.t * 3.1))
        cv2.rectangle(frame, (ox - 55, oy - 40), (ox + 55, oy + 40), (40, 50, 210), -1)
        cv2.putText(frame, "MOVING OBSTACLE", (ox - 48, oy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        return True, frame

    def release(self):
        pass


def open_video_source(source: str):
    if source.lower() == "demo":
        return SyntheticDroneFPVFeed(), "Synthetic 6-DoF Drone FPV Stream"
    if source.isdigit():
        idx = int(source)
        for backend in (cv2.CAP_DSHOW, cv2.CAP_ANY):
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    return cap, f"Webcam #{idx}"
                cap.release()
        print(f"[Info] No physical webcam detected at index {idx}. Falling back to 6-DoF Drone FPV Demo stream!")
        return SyntheticDroneFPVFeed(), "Synthetic 6-DoF Drone FPV Stream"
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {source}")
    return cap, os.path.basename(source)


def run_live_tracker(source: str = "0", save_video: str | None = None, max_points: int = 450, scale_width: int = 960):
    device = twistor.get_best_device()
    cap, source_label = open_video_source(source)

    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to grab initial frame from video source.")

    # Resize high-res 1080p/4K drone videos smoothly while preserving aspect ratio
    orig_h, orig_w = frame.shape[:2]
    if scale_width > 0 and orig_w > scale_width:
        scale_factor = scale_width / float(orig_w)
        frame = cv2.resize(frame, (scale_width, int(orig_h * scale_factor)), interpolation=cv2.INTER_AREA)

    h, w = frame.shape[:2]
    fx = fy = 0.85 * max(w, h)
    cx, cy = w * 0.5, h * 0.5
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

    writer = None
    if save_video:
        os.makedirs(os.path.dirname(save_video) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_video, fourcc, 30.0, (w, h))

    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )

    def init_anchor_map(gray_img, R_wc_curr: np.ndarray, pos_curr: np.ndarray):
        pts = cv2.goodFeaturesToTrack(
            gray_img, maxCorners=max_points, qualityLevel=0.01, minDistance=14, blockSize=7
        )
        if pts is None or len(pts) < 16:
            return None, None
        pts_2d = pts.reshape(-1, 2).astype(np.float32)
        xn = (pts_2d[:, 0] - cx) / fx
        yn = (pts_2d[:, 1] - cy) / fy
        z0 = np.full_like(xn, 2.5)
        rays_cam = np.stack([xn * z0, yn * z0, z0], axis=-1).astype(np.float32)
        # Rotate camera-frame rays into world frame and offset by current camera position
        xyz_world = (rays_cam @ R_wc_curr) + pos_curr.reshape(1, 3)
        return pts, torch.tensor(xyz_world, dtype=torch.float32, device=device)

    R_wc = np.eye(3, dtype=np.float64)
    pos_np = np.zeros(3, dtype=np.float32)
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    p0, landmarks_3d = init_anchor_map(prev_gray, R_wc, pos_np)
    trail = []

    print(f"=== twistor-vision v{twistor.__version__} 6-DoF Live Video Tracker [{device}] ===")
    print(f"Source: {source_label} ({w}x{h}) | Press 'R' to reset origin, 'Q' or ESC to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if scale_width > 0 and frame.shape[1] > scale_width:
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hud = frame.copy()

        if p0 is None or len(p0) < 45:
            p0, landmarks_3d = init_anchor_map(gray, R_wc, pos_np)
            prev_gray = gray
            continue

        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **lk_params)
        good_mask = (st.reshape(-1) == 1)
        p0_good = p0[good_mask].reshape(-1, 2)
        p1_good = p1[good_mask].reshape(-1, 2)
        landmarks_3d = landmarks_3d[torch.tensor(good_mask, device=device)]

        if len(p1_good) < 20:
            p0, landmarks_3d = init_anchor_map(gray, R_wc, pos_np)
            prev_gray = gray
            continue

        # 1. Estimate incremental 3D camera rotation dR via Essential Matrix RANSAC
        E, e_mask = cv2.findEssentialMat(p0_good, p1_good, K, method=cv2.RANSAC, prob=0.999, threshold=1.2)
        if E is not None and E.shape == (3, 3):
            _, dR, _, _ = cv2.recoverPose(E, p0_good, p1_good, K, mask=e_mask)
            # Only integrate clean small-angle inter-frame rotations (reject degenerate planar flips)
            angle = np.arccos(np.clip((np.trace(dR) - 1.0) * 0.5, -1.0, 1.0))
            if angle < 0.25:
                R_wc = dR @ R_wc
                # Re-orthogonalize via SVD to prevent numerical drift
                U_svd, _, Vt_svd = np.linalg.svd(R_wc)
                R_wc = U_svd @ Vt_svd

        q_wxyz = torch.tensor(rotmat_to_quat_wxyz(R_wc), dtype=torch.float32, device=device)
        R_A = quat_wxyz_to_su2(q_wxyz)

        # 2. Convert pixels to celestial-sphere stereographic rays & solve 3D position via Spinor RANSAC
        px_t = torch.tensor(p1_good[:, 0], dtype=torch.float32, device=device)
        py_t = torch.tensor(p1_good[:, 1], dtype=torch.float32, device=device)
        uv_t = twistor.pixel_to_stereographic_uv(px_t, py_t, fx, fy, cx, cy)

        t0 = time.perf_counter()
        pos_est, inliers = twistor.localize_ransac(
            landmarks_3d,
            uv_t,
            R_A=R_A,
            num_hypotheses=256,
            inlier_threshold=0.014,
        )
        if str(device) == "xpu":
            torch.xpu.synchronize()
        elif str(device) == "cuda":
            torch.cuda.synchronize()
        solve_ms = (time.perf_counter() - t0) * 1000.0

        inliers_np = inliers.cpu().numpy()
        pos_np = pos_est.cpu().numpy()
        roll_deg, pitch_deg, yaw_deg = rotmat_to_rpy_deg(R_wc)

        trail.append(pos_np.copy())
        if len(trail) > 180:
            trail.pop(0)

        # Draw Green Inliers & Red RANSAC-Rejected Outliers
        for (x_pt, y_pt), is_in in zip(p1_good, inliers_np):
            color = (60, 235, 110) if is_in else (50, 60, 255)
            radius = 3 if is_in else 5
            cv2.circle(hud, (int(x_pt), int(y_pt)), radius, color, -1)

        # Top-left 6-DoF telemetry HUD
        cv2.rectangle(hud, (12, 12), (540, 148), (18, 22, 28), -1)
        cv2.rectangle(hud, (12, 12), (540, 148), (0, 180, 255), 1)
        cv2.putText(
            hud,
            f"twistor-vision v{twistor.__version__} 6-DoF [{str(device).upper()}] | {solve_ms:.2f} ms ({1000.0/max(solve_ms,1e-3):.0f} Hz)",
            (22, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.53,
            (0, 210, 255),
            2,
        )
        cv2.putText(
            hud,
            f"Pos (m): X={pos_np[0]:+.3f}  Y={pos_np[1]:+.3f}  Z={pos_np[2]:+.3f}",
            (22, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.53,
            (240, 245, 250),
            1,
        )
        cv2.putText(
            hud,
            f"Att (deg): Roll={roll_deg:+.1f}  Pitch={pitch_deg:+.1f}  Yaw={yaw_deg:+.1f}",
            (22, 96),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.53,
            (180, 225, 255),
            1,
        )
        cv2.putText(
            hud,
            f"Spinor RANSAC Inliers: {int(inliers_np.sum())}/{len(inliers_np)} (Green=Inlier, Red=Rejected)",
            (22, 126),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.49,
            (90, 240, 130),
            1,
        )

        # Top-right X-Y / X-Z mini-map trajectory trail
        map_size = 148
        mx0, my0 = w - map_size - 16, 16
        cv2.rectangle(hud, (mx0, my0), (mx0 + map_size, my0 + map_size), (18, 22, 28), -1)
        cv2.rectangle(hud, (mx0, my0), (mx0 + map_size, my0 + map_size), (100, 120, 140), 1)
        center_px = (mx0 + map_size // 2, my0 + map_size // 2)
        cv2.circle(hud, center_px, 3, (120, 120, 120), -1)
        for k in range(1, len(trail)):
            pt_a = (int(center_px[0] + trail[k - 1][0] * 55), int(center_px[1] + trail[k - 1][1] * 55))
            pt_b = (int(center_px[0] + trail[k][0] * 55), int(center_px[1] + trail[k][1] * 55))
            cv2.line(hud, pt_a, pt_b, (0, 210, 255), 2)

        if writer is not None:
            writer.write(hud)

        cv2.imshow("twistor-vision 6-DoF Live Spinor Tracker", hud)
        key = cv2.waitKey(16) & 0xFF
        if key in (27, ord("q"), ord("Q")):
            break
        elif key in (ord("r"), ord("R")):
            R_wc = np.eye(3, dtype=np.float64)
            pos_np = np.zeros(3, dtype=np.float32)
            p0, landmarks_3d = init_anchor_map(gray, R_wc, pos_np)
            prev_gray = gray
            trail.clear()
            continue

        prev_gray = gray
        p0 = p1_good.reshape(-1, 1, 2)

    cap.release()
    if writer is not None:
        writer.release()
        print(f"Saved recorded demo video to {save_video}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="0", help="Webcam index (0), video file (.mp4), or 'demo'")
    parser.add_argument("--save-video", type=str, default=None, help="Optional output path to record HUD video (.mp4)")
    parser.add_argument("--scale-width", type=int, default=960, help="Resize width for high-res videos (0 to disable)")
    args = parser.parse_args()
    run_live_tracker(source=args.source, save_video=args.save_video, scale_width=args.scale_width)