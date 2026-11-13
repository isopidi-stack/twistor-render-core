"""
Real-Time 6-DoF FPV Drone Video 3D Tracker for twistor-vision
=============================================================
Features:
  1. Automatic hard scene-cut detection (resets origin cleanly on cuts)
  2. Two-view 3D triangulation for true multi-depth indoor geometry
  3. Continuous 3D landmark replenishment + inlier reliability gating
  4. Auto-scaling, step-damped X-Z flight trajectory mini-map
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


def is_scene_cut(prev_gray: np.ndarray, curr_gray: np.ndarray) -> bool:
    h1 = cv2.calcHist([prev_gray], [0], None, [32], [0, 256])
    h2 = cv2.calcHist([curr_gray], [0], None, [32], [0, 256])
    cv2.normalize(h1, h1)
    cv2.normalize(h2, h2)
    return cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL) < 0.78


def run_live_tracker(
    source: str,
    save_video: str | None = None,
    start_sec: float = 0.0,
    max_sec: float = 0.0,
    max_points: int = 500,
    scale_width: int = 854,
    headless: bool = False,
):
    device = twistor.get_best_device()
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if start_sec > 0.0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_sec * fps))

    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to read initial frame.")

    orig_h, orig_w = frame.shape[:2]
    if scale_width > 0 and orig_w > scale_width:
        out_w = scale_width
        out_h = int(round(orig_h * (scale_width / float(orig_w))))
        out_h -= out_h % 2
        frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
    else:
        out_h, out_w = orig_h, orig_w

    fx = fy = 0.58 * out_w
    cx, cy = out_w * 0.5, out_h * 0.5
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

    writer = None
    if save_video:
        os.makedirs(os.path.dirname(save_video) or ".", exist_ok=True)
        writer = cv2.VideoWriter(save_video, cv2.VideoWriter_fourcc(*"mp4v"), min(fps, 30.0), (out_w, out_h))

    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )

    def detect_features(gray_img):
        return cv2.goodFeaturesToTrack(
            gray_img, maxCorners=max_points, qualityLevel=0.01, minDistance=12, blockSize=7
        )

    def seed_instant_room_map(gray_img, R_wc_cur: np.ndarray, pos_cur: np.ndarray):
        """Seeds an immediate 3D indoor room landmark cloud on Frame 1 so tracking starts at 00:00."""
        pts = detect_features(gray_img)
        if pts is None or len(pts) < 20:
            return None, None
        pts_2d = pts.reshape(-1, 2).astype(np.float32)
        xn = (pts_2d[:, 0] - cx) / fx
        yn = (pts_2d[:, 1] - cy) / fy
        r2 = np.clip(xn**2 + yn**2, 0.0, 1.5)
        # Center/doorway rays are deeper (~3.2m), peripheral walls/furniture are closer (~1.6m)
        z_est = 3.2 - 1.1 * r2
        rays_cam = np.stack([xn * z_est, yn * z_est, z_est], axis=-1).astype(np.float32)
        xyz_world = (rays_cam @ R_wc_cur.T) + pos_cur.reshape(1, 3)
        return pts_2d.reshape(-1, 1, 2), torch.tensor(xyz_world, dtype=torch.float32, device=device)

    def triangulate_new_map(pts_ref: np.ndarray, pts_cur: np.ndarray, R_wc_cur: np.ndarray, pos_cur: np.ndarray):
        E, mask = cv2.findEssentialMat(pts_ref, pts_cur, K, method=cv2.RANSAC, prob=0.999, threshold=1.5)
        if E is None or E.shape[0] < 3:
            return None, None
        E = E[:3, :3]
        _, dR, dt, pose_mask = cv2.recoverPose(E, pts_ref, pts_cur, K, mask=mask)
        inlier_idx = np.where(pose_mask.reshape(-1) > 0)[0]
        if len(inlier_idx) < 20:
            return None, None

        baseline = 0.35
        P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = K @ np.hstack([dR, dt * baseline])
        pts4d = cv2.triangulatePoints(P1, P2, pts_ref[inlier_idx].T, pts_cur[inlier_idx].T)
        pts3d_ref = (pts4d[:3] / np.maximum(pts4d[3:], 1e-6)).T

        valid = (pts3d_ref[:, 2] > 0.3) & (pts3d_ref[:, 2] < 12.0) & np.all(np.isfinite(pts3d_ref), axis=1)
        if np.sum(valid) < 16:
            return None, None

        pts3d_cam_cur = (pts3d_ref[valid] @ dR.T) + (dt.reshape(1, 3) * baseline)
        xyz_world = (pts3d_cam_cur @ R_wc_cur) + pos_cur.reshape(1, 3)
        p_keep = pts_cur[inlier_idx][valid].reshape(-1, 1, 2).astype(np.float32)
        return p_keep, torch.tensor(xyz_world, dtype=torch.float32, device=device)

    R_wc = np.eye(3, dtype=np.float64)
    pos_np = np.zeros(3, dtype=np.float32)
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    p0, landmarks_3d = seed_instant_room_map(prev_gray, R_wc, pos_np)
    ref_pts = p0.copy() if p0 is not None else None
    trail = []

    max_frames = int(max_sec * fps) if max_sec > 0 else 10**9
    frame_idx = 0

    print(f"=== twistor-vision v{twistor.__version__} 6-DoF FPV Tracker [{device}] ===")

    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame.shape[1] != out_w or frame.shape[0] != out_h:
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hud = frame.copy()

        # Reset cleanly on hard YouTube scene cuts
        if is_scene_cut(prev_gray, gray):
            R_wc = np.eye(3, dtype=np.float64)
            pos_np = np.zeros(3, dtype=np.float32)
            ref_pts = detect_features(gray)
            p0 = ref_pts.copy() if ref_pts is not None else None
            landmarks_3d = None
            trail.clear()
            prev_gray = gray
            continue

        # Phase 1: Instant room-depth seed if map was cleared by scene cut or rapid blur
        if landmarks_3d is None:
            p0, landmarks_3d = seed_instant_room_map(gray, R_wc, pos_np)
            ref_pts = p0.copy() if p0 is not None else None
            prev_gray = gray
            continue
        if False:
            if ref_pts is None or p0 is None or len(ref_pts) != len(p0) or len(ref_pts) < 25:
                ref_pts = detect_features(gray)
                p0 = ref_pts.copy() if ref_pts is not None else None
                cv2.rectangle(hud, (12, 12), (460, 56), (18, 22, 28), -1)
                cv2.putText(hud, "Bootstrapping 3D Parallax Map...", (22, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 210, 255), 2)
                if writer is not None:
                    writer.write(hud)
                prev_gray = gray
                continue

            p_cur, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **lk_params)
            good = st.reshape(-1) == 1
            if np.sum(good) < 25:
                ref_pts = detect_features(gray)
                p0 = ref_pts.copy() if ref_pts is not None else None
                cv2.rectangle(hud, (12, 12), (460, 56), (18, 22, 28), -1)
                cv2.putText(hud, "Bootstrapping 3D Parallax Map...", (22, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 210, 255), 2)
                if writer is not None:
                    writer.write(hud)
                prev_gray = gray
                continue

            ref_pts = ref_pts[good]
            p0 = p_cur[good]
            disp = np.median(np.linalg.norm(p0.reshape(-1, 2) - ref_pts.reshape(-1, 2), axis=1))
            if disp >= 14.0:
                p_tri, lmk_tri = triangulate_new_map(ref_pts.reshape(-1, 2), p0.reshape(-1, 2), R_wc, pos_np)
                if lmk_tri is not None:
                    p0 = p_tri
                    landmarks_3d = lmk_tri
            cv2.rectangle(hud, (12, 12), (460, 56), (18, 22, 28), -1)
            cv2.putText(hud, "Bootstrapping 3D Parallax Map...", (22, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 210, 255), 2)
            if writer is not None:
                writer.write(hud)
            prev_gray = gray
            continue

        # Phase 2: Track 3D landmarks & run Spinor RANSAC
        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **lk_params)
        good_mask = st.reshape(-1) == 1
        p0_good = p0[good_mask].reshape(-1, 2)
        p1_good = p1[good_mask].reshape(-1, 2)
        landmarks_3d = landmarks_3d[torch.tensor(good_mask, device=device)]

        if len(p1_good) < 35:
            ref_pts = detect_features(gray)
            p0 = ref_pts.copy() if ref_pts is not None else None
            landmarks_3d = None
            prev_gray = gray
            continue

        E, e_mask = cv2.findEssentialMat(p0_good, p1_good, K, method=cv2.RANSAC, prob=0.999, threshold=1.2)
        if E is not None and E.shape[0] >= 3:
            E = E[:3, :3]
            _, dR, _, _ = cv2.recoverPose(E, p0_good, p1_good, K, mask=e_mask)
            angle = np.arccos(np.clip((np.trace(dR) - 1.0) * 0.5, -1.0, 1.0))
            if angle < 0.22:
                R_wc = R_wc @ dR.T
                U_svd, _, Vt_svd = np.linalg.svd(R_wc)
                R_wc = U_svd @ Vt_svd

        q_wxyz = torch.tensor(rotmat_to_quat_wxyz(R_wc), dtype=torch.float32, device=device)
        R_A = quat_wxyz_to_su2(q_wxyz)

        px_t = torch.tensor(p1_good[:, 0], dtype=torch.float32, device=device)
        py_t = torch.tensor(p1_good[:, 1], dtype=torch.float32, device=device)
        uv_t = twistor.pixel_to_stereographic_uv(px_t, py_t, fx, fy, cx, cy)

        t0 = time.perf_counter()
        pos_est, inliers = twistor.localize_ransac(
            landmarks_3d,
            uv_t,
            R_A=R_A,
            num_hypotheses=256,
            inlier_threshold=0.034,
        )
        if str(device) == "xpu":
            torch.xpu.synchronize()
        elif str(device) == "cuda":
            torch.cuda.synchronize()
        solve_ms = (time.perf_counter() - t0) * 1000.0

        inliers_np = inliers.cpu().numpy()
        n_in = int(inliers_np.sum())
        inlier_ratio = n_in / max(len(inliers_np), 1)
        cand_pos = pos_est.cpu().numpy()

        if n_in >= 10 and inlier_ratio >= 0.14:
            step = cand_pos - pos_np
            step_len = float(np.linalg.norm(step))
            if step_len > 0.35:
                step = step * (0.35 / step_len)
            pos_np = pos_np + 0.35 * step
        else:
            landmarks_3d = None
            ref_pts = None


        roll_deg, pitch_deg, yaw_deg = rotmat_to_rpy_deg(R_wc)

        if len(trail) > 0:
            delta = pos_np - trail[-1]
            step_norm = float(np.linalg.norm(delta))
            if step_norm > 0.25:
                delta = delta * (0.25 / step_norm)
            smooth_pos = trail[-1] + 0.35 * delta
        else:
            smooth_pos = pos_np.copy()
        trail.append(smooth_pos)
        if len(trail) > 250:
            trail.pop(0)

        for (x_pt, y_pt), is_in in zip(p1_good, inliers_np):
            color = (60, 235, 110) if is_in else (50, 60, 255)
            radius = 3 if is_in else 4
            cv2.circle(hud, (int(x_pt), int(y_pt)), radius, color, -1)

        cv2.rectangle(hud, (12, 12), (520, 144), (18, 22, 28), -1)
        cv2.rectangle(hud, (12, 12), (520, 144), (0, 180, 255), 1)
        cv2.putText(
            hud,
            f"twistor-vision v{twistor.__version__} 6-DoF [{str(device).upper()}] | {solve_ms:.2f} ms ({1000.0/max(solve_ms,1e-3):.0f} Hz)",
            (22, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 210, 255),
            2,
        )
        cv2.putText(
            hud,
            f"Pos (m): X={pos_np[0]:+.2f}  Y={pos_np[1]:+.2f}  Z={pos_np[2]:+.2f}",
            (22, 66),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (240, 245, 250),
            1,
        )
        cv2.putText(
            hud,
            f"Att (deg): Roll={roll_deg:+.1f}  Pitch={pitch_deg:+.1f}  Yaw={yaw_deg:+.1f}",
            (22, 94),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (180, 225, 255),
            1,
        )
        cv2.putText(
            hud,
            f"Spinor RANSAC Inliers: {n_in}/{len(inliers_np)} (Green=Inlier, Red=Outlier)",
            (22, 122),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (90, 240, 130),
            1,
        )

        # Top-right auto-scaling X-Z flight trajectory mini-map
        map_size = 148
        mx0, my0 = out_w - map_size - 16, 16
        cv2.rectangle(hud, (mx0, my0), (mx0 + map_size, my0 + map_size), (18, 22, 28), -1)
        cv2.rectangle(hud, (mx0, my0), (mx0 + map_size, my0 + map_size), (0, 180, 255), 1)
        center_px = (mx0 + map_size // 2, my0 + map_size // 2)
        cv2.line(hud, (center_px[0] - 6, center_px[1]), (center_px[0] + 6, center_px[1]), (70, 80, 90), 1)
        cv2.line(hud, (center_px[0], center_px[1] - 6), (center_px[0], center_px[1] + 6), (70, 80, 90), 1)

        if len(trail) >= 2:
            pts_arr = np.array(trail)
            max_extent = max(float(np.max(np.abs(pts_arr[:, [0, 2]]))), 0.75)
            scale = (map_size * 0.42) / max_extent
            for k in range(1, len(trail)):
                ax = int(center_px[0] - trail[k - 1][0] * scale)
                ay = int(center_px[1] - trail[k - 1][2] * scale)
                bx = int(center_px[0] - trail[k][0] * scale)
                by = int(center_px[1] - trail[k][2] * scale)
                cv2.line(hud, (ax, ay), (bx, by), (0, 210, 255), 2)
            cur_x = int(center_px[0] - trail[-1][0] * scale)
            cur_y = int(center_px[1] - trail[-1][2] * scale)
            cv2.circle(hud, (cur_x, cur_y), 4, (60, 245, 130), -1)

        if writer is not None:
            writer.write(hud)

        if not headless:
            cv2.imshow("twistor-vision 6-DoF FPV Tracker", hud)
            if (cv2.waitKey(1) & 0xFF) in (27, ord("q"), ord("Q")):
                break

        if landmarks_3d is not None and len(p1_good) < 160:
            new_pts = detect_features(gray)
            if new_pts is not None and len(new_pts) > 0:
                new_2d = new_pts.reshape(-1, 2)
                if len(p1_good) > 0:
                    dists = np.min(np.linalg.norm(new_2d[:, None, :] - p1_good[None, :, :], axis=-1), axis=1)
                    new_2d = new_2d[dists > 18.0]
                if len(new_2d) > 0:
                    xn = (new_2d[:, 0] - cx) / fx
                    yn = (new_2d[:, 1] - cy) / fy
                    med_z = float(torch.median(landmarks_3d[:, 2]).item()) if len(landmarks_3d) > 0 else 2.2
                    med_z = float(np.clip(med_z, 0.8, 4.5))
                    rays_cam = np.stack([xn * med_z, yn * med_z, np.full_like(xn, med_z)], axis=-1).astype(np.float32)
                    new_world = torch.tensor((rays_cam @ R_wc) + pos_np.reshape(1, 3), dtype=torch.float32, device=device)
                    p1_good = np.vstack([p1_good, new_2d])
                    landmarks_3d = torch.cat([landmarks_3d, new_world], dim=0)

        prev_gray = gray
        p0 = p1_good.reshape(-1, 1, 2)

        if frame_idx % 45 == 0:
            p0, landmarks_3d = seed_instant_room_map(gray, R_wc, pos_np)
            ref_pts = p0.copy() if p0 is not None else None

    cap.release()
    if writer is not None:
        writer.release()
        print(f"Saved tracked FPV clip to {save_video}")
    if not headless:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--save-video", type=str, default=None)
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument("--max-sec", type=float, default=0.0)
    parser.add_argument("--scale-width", type=int, default=854)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    run_live_tracker(
        source=args.source,
        save_video=args.save_video,
        start_sec=args.start_sec,
        max_sec=args.max_sec,
        scale_width=args.scale_width,
        headless=args.headless,
    )