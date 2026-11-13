"""
Real-Time Live Webcam / Video / Synthetic Drone FPV 3D Tracker for twistor-vision
=================================================================================
Tracks optical flow keypoints from:
  - A live webcam (--source 0 or --source 1)
  - A recorded video file (--source path/to/drone_flight.mp4)
  - A built-in real-time 3D textured drone FPV simulator (--source demo)

Converts pinhole pixels to stereographic spinors and runs twistor.localize_ransac()
on GPU/XPU in real time with inlier/outlier HUD and X-Z trajectory mini-map.

Controls:
  R - Re-anchor 3D reference map at current camera view
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


class SyntheticDroneFPVFeed:
    """Generates a real-time textured 3D indoor flight video feed with moving outlier clutter."""

    def __init__(self, width: int = 960, height: int = 600):
        self.w = width
        self.h = height
        self.t = 0.0
        # Build rich high-contrast textured floor/wall canvas (2000x2000)
        rng = np.random.default_rng(42)
        canvas = np.full((2000, 2000, 3), (28, 34, 42), dtype=np.uint8)
        for gx in range(0, 2000, 80):
            cv2.line(canvas, (gx, 0), (gx, 2000), (55, 68, 82), 2)
        for gy in range(0, 2000, 80):
            cv2.line(canvas, (0, gy), (2000, gy), (55, 68, 82), 2)
        for _ in range(450):
            cx, cy = rng.integers(100, 1900, size=2)
            rad = int(rng.integers(8, 28))
            col = tuple(int(c) for c in rng.integers(90, 240, size=3))
            if rng.random() > 0.5:
                cv2.circle(canvas, (int(cx), int(cy)), rad, col, -1)
            else:
                cv2.rectangle(canvas, (int(cx - rad), int(cy - rad)), (int(cx + rad), int(cy + rad)), col, 2)
        self.canvas = canvas

    def read(self):
        self.t += 0.03
        # Smooth 3D figure-8 camera motion
        dx = 140.0 * np.sin(self.t)
        dy = 95.0 * np.sin(2.0 * self.t)
        scale = 1.0 + 0.14 * np.cos(self.t * 1.3)

        src_pts = np.float32([
            [520 + dx, 700 + dy],
            [1480 + dx, 700 + dy],
            [1480 + dx, 1300 + dy],
            [520 + dx, 1300 + dy],
        ])
        pad_x = (1.0 - scale) * 180.0
        pad_y = (1.0 - scale) * 120.0
        dst_pts = np.float32([
            [pad_x, pad_y],
            [self.w - pad_x, pad_y],
            [self.w - pad_x, self.h - pad_y],
            [pad_x, self.h - pad_y],
        ])
        H = cv2.getPerspectiveTransform(src_pts, dst_pts)
        frame = cv2.warpPerspective(self.canvas, H, (self.w, self.h))

        # Inject a moving foreground object (dynamic outlier generator)
        ox = int(self.w * 0.5 + 260 * np.cos(self.t * 2.4))
        oy = int(self.h * 0.5 + 140 * np.sin(self.t * 3.1))
        cv2.rectangle(frame, (ox - 55, oy - 40), (ox + 55, oy + 40), (40, 50, 210), -1)
        cv2.putText(frame, "MOVING OBSTACLE", (ox - 48, oy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        return True, frame

    def release(self):
        pass


def open_video_source(source: str):
    if source.lower() == "demo":
        return SyntheticDroneFPVFeed(), "Synthetic 3D Drone FPV Stream"
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
        print(f"[Info] No physical webcam detected at index {idx}. Falling back to built-in 3D Drone FPV Demo stream!")
        return SyntheticDroneFPVFeed(), "Synthetic 3D Drone FPV Stream"
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {source}")
    return cap, os.path.basename(source)


def run_live_tracker(source: str = "0", save_video: str | None = None, max_points: int = 400):
    device = twistor.get_best_device()
    cap, source_label = open_video_source(source)

    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to grab initial frame from video source.")

    h, w = frame.shape[:2]
    fx = fy = 0.85 * max(w, h)
    cx, cy = w * 0.5, h * 0.5

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

    def init_anchor_map(gray_img):
        pts = cv2.goodFeaturesToTrack(
            gray_img, maxCorners=max_points, qualityLevel=0.01, minDistance=14, blockSize=7
        )
        if pts is None or len(pts) < 12:
            return None, None
        pts_2d = pts.reshape(-1, 2).astype(np.float32)
        xn = (pts_2d[:, 0] - cx) / fx
        yn = (pts_2d[:, 1] - cy) / fy
        z0 = np.full_like(xn, 2.0)
        xyz = np.stack([xn * z0, yn * z0, z0], axis=-1).astype(np.float32)
        return pts, torch.tensor(xyz, device=device)

    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    p0, landmarks_3d = init_anchor_map(prev_gray)
    trail = []
    R_identity = quat_wxyz_to_su2(torch.tensor([1.0, 0.0, 0.0, 0.0], device=device))

    print(f"=== twistor-vision v{twistor.__version__} Live Video Tracker [{device}] ===")
    print(f"Source: {source_label} ({w}x{h}) | Press 'R' to re-anchor map, 'Q' or ESC to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hud = frame.copy()

        if p0 is None or len(p0) < 45:
            offset = trail[-1] if len(trail) > 0 else np.zeros(3, dtype=np.float32)
            p0, landmarks_3d = init_anchor_map(gray)
            if landmarks_3d is not None:
                landmarks_3d = landmarks_3d + torch.tensor(offset, device=device)
            prev_gray = gray
            continue

        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **lk_params)
        good_mask = (st.reshape(-1) == 1)
        p1_good = p1[good_mask].reshape(-1, 2)
        landmarks_3d = landmarks_3d[torch.tensor(good_mask, device=device)]

        if len(p1_good) < 16:
            p0, landmarks_3d = init_anchor_map(gray)
            prev_gray = gray
            trail.clear()
            continue

        px_t = torch.tensor(p1_good[:, 0], dtype=torch.float32, device=device)
        py_t = torch.tensor(p1_good[:, 1], dtype=torch.float32, device=device)
        uv_t = twistor.pixel_to_stereographic_uv(px_t, py_t, fx, fy, cx, cy)

        t0 = time.perf_counter()
        pos_est, inliers = twistor.localize_ransac(
            landmarks_3d,
            uv_t,
            R_A=R_identity,
            num_hypotheses=256,
            inlier_threshold=0.012,
        )
        if str(device) == "xpu":
            torch.xpu.synchronize()
        elif str(device) == "cuda":
            torch.cuda.synchronize()
        solve_ms = (time.perf_counter() - t0) * 1000.0

        inliers_np = inliers.cpu().numpy()
        pos_np = pos_est.cpu().numpy()
        trail.append(pos_np.copy())
        if len(trail) > 150:
            trail.pop(0)

        for (x_pt, y_pt), is_in in zip(p1_good, inliers_np):
            color = (60, 235, 110) if is_in else (50, 60, 255)
            radius = 3 if is_in else 5
            cv2.circle(hud, (int(x_pt), int(y_pt)), radius, color, -1)

        cv2.rectangle(hud, (12, 12), (510, 122), (18, 22, 28), -1)
        cv2.rectangle(hud, (12, 12), (510, 122), (0, 180, 255), 1)
        cv2.putText(
            hud,
            f"twistor-vision v{twistor.__version__} [{str(device).upper()}] | {solve_ms:.2f} ms ({1000.0/max(solve_ms,1e-3):.0f} Hz)",
            (22, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 210, 255),
            2,
        )
        cv2.putText(
            hud,
            f"Pos (m): X={pos_np[0]:+.3f}  Y={pos_np[1]:+.3f}  Z={pos_np[2]:+.3f}",
            (22, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (240, 245, 250),
            1,
        )
        cv2.putText(
            hud,
            f"RANSAC Inliers: {int(inliers_np.sum())}/{len(inliers_np)} (Green=Inlier, Red=Rejected)",
            (22, 98),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (90, 240, 130),
            1,
        )

        map_size = 140
        mx0, my0 = w - map_size - 16, 16
        cv2.rectangle(hud, (mx0, my0), (mx0 + map_size, my0 + map_size), (18, 22, 28), -1)
        cv2.rectangle(hud, (mx0, my0), (mx0 + map_size, my0 + map_size), (100, 120, 140), 1)
        center_px = (mx0 + map_size // 2, my0 + map_size // 2)
        cv2.circle(hud, center_px, 3, (120, 120, 120), -1)
        for k in range(1, len(trail)):
            pt_a = (int(center_px[0] + trail[k - 1][0] * 60), int(center_px[1] + trail[k - 1][1] * 60))
            pt_b = (int(center_px[0] + trail[k][0] * 60), int(center_px[1] + trail[k][1] * 60))
            cv2.line(hud, pt_a, pt_b, (0, 210, 255), 2)

        if writer is not None:
            writer.write(hud)

        cv2.imshow("twistor-vision Live 3D Spinor Tracker", hud)
        key = cv2.waitKey(16) & 0xFF
        if key in (27, ord("q"), ord("Q")):
            break
        elif key in (ord("r"), ord("R")):
            p0, landmarks_3d = init_anchor_map(gray)
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
    args = parser.parse_args()
    run_live_tracker(source=args.source, save_video=args.save_video)