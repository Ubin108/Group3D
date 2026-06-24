import numpy as np
import cv2


def to4x4(M: np.ndarray) -> np.ndarray:
    M = np.asarray(M, dtype=np.float32)
    if M.shape == (4, 4):
        return M
    if M.shape == (3, 4):
        H = np.eye(4, dtype=np.float32)
        H[:3, :4] = M
        return H
    raise ValueError(f"Unexpected pose shape: {M.shape}")


def load_axis_alignment(scene_txt_path: str) -> np.ndarray:
    with open(scene_txt_path, "r") as f:
        for line in f:
            if "axisAlignment" in line:
                vals = list(map(float, line.strip().split("=")[1].split()))
                return np.array(vals).reshape(4, 4)
    raise ValueError(f"axisAlignment not found in {scene_txt_path}")


def backproject_depth(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    depth (H, W) + intrinsics K → camera-space points (H, W, 3).
    """
    H, W = depth.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)
    z = depth.astype(np.float32)
    return np.stack([(uu - cx) * z / fx, (vv - cy) * z / fy, z], axis=-1)


def cam_to_world(points_cam: np.ndarray, C_c2w: np.ndarray) -> np.ndarray:
    """
    points_cam (H, W, 3) + c2w pose (4, 4) → world points (H, W, 3).
    """
    H, W, _ = points_cam.shape
    pts = points_cam.reshape(-1, 3)
    pts_h = np.concatenate([pts, np.ones((len(pts), 1), dtype=np.float32)], axis=1)
    return (pts_h @ C_c2w.T)[:, :3].reshape(H, W, 3)

def _parse_scannet_txt(scene_txt_path: str) -> dict:
    vals = {}
    with open(scene_txt_path, "r") as f:
        for line in f:
            if "=" not in line:
                continue
            k, v = line.strip().split("=", 1)
            vals[k.strip()] = v.strip()
    return vals


def load_color_K_and_size(scene_txt_path: str):
    vals = _parse_scannet_txt(scene_txt_path)
    cw, ch = int(vals["colorWidth"]), int(vals["colorHeight"])
    Kc = np.array([
        [float(vals["fx_color"]), 0,                     float(vals["mx_color"])],
        [0,                      float(vals["fy_color"]), float(vals["my_color"])],
        [0,                      0,                      1],
    ], dtype=np.float32)
    return Kc, (cw, ch)


def load_scannet_calib(scene_txt_path: str):
    vals = _parse_scannet_txt(scene_txt_path)
    cw, ch = int(vals["colorWidth"]),  int(vals["colorHeight"])
    dw, dh = int(vals["depthWidth"]),  int(vals["depthHeight"])

    Kc = np.array([
        [float(vals["fx_color"]), 0,                     float(vals["mx_color"])],
        [0,                      float(vals["fy_color"]), float(vals["my_color"])],
        [0,                      0,                      1],
    ], dtype=np.float32)

    Kd = np.array([
        [float(vals["fx_depth"]), 0,                     float(vals["mx_depth"])],
        [0,                      float(vals["fy_depth"]), float(vals["my_depth"])],
        [0,                      0,                      1],
    ], dtype=np.float32)

    T_d_c = (
        np.array(list(map(float, vals["colorToDepthExtrinsics"].split())),
                 dtype=np.float32).reshape(4, 4)
        if "colorToDepthExtrinsics" in vals
        else np.eye(4, dtype=np.float32)
    )

    return Kc, Kd, T_d_c, (cw, ch, dw, dh)


def scale_intrinsics(K: np.ndarray, src_wh: tuple, dst_wh: tuple) -> np.ndarray:
    W0, H0 = src_wh
    W1, H1 = dst_wh
    K2 = K.copy().astype(np.float32)
    K2[0, 0] *= W1 / W0;  K2[0, 2] *= W1 / W0
    K2[1, 1] *= H1 / H0;  K2[1, 2] *= H1 / H0
    return K2

def load_gt_depth_png(path: str) -> np.ndarray:
    """ScanNet GT depth PNG (mm, uint16) → float32 meters."""
    depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise ValueError(f"Cannot read depth: {path}")
    if depth.dtype == np.uint16:
        depth = depth.astype(np.float32) / 1000.0
    return depth


def estimate_scale_from_depth(pred_depth: np.ndarray, gt_depth: np.ndarray) -> float:
    mask = (pred_depth > 0) & (gt_depth > 0)
    if mask.sum() < 100:
        print("Warning: too few valid depth pixels for scale estimation")
        return 1.0
    ratios = gt_depth[mask] / pred_depth[mask]
    ratios = ratios[np.isfinite(ratios) & (ratios > 0.1) & (ratios < 10)]
    return float(np.median(ratios))


def warp_depth_to_pred_frame(
    depth_raw_m: np.ndarray,
    scene_txt_path: str,
    pred_shape_hw: tuple,
) -> np.ndarray:
    Kc, Kd, T_d_c, (cw, ch, dw, dh) = load_scannet_calib(scene_txt_path)
    Hp, Wp = pred_shape_hw
    T_c_d  = np.linalg.inv(T_d_c).astype(np.float32)
    sx, sy = Wp / float(cw), Hp / float(ch)

    v, u = np.indices((dh, dw))
    z    = depth_raw_m.astype(np.float32)
    valid = (z > 0) & np.isfinite(z)
    u, v, z = u[valid].astype(np.float32), v[valid].astype(np.float32), z[valid]

    fx_d, fy_d, cx_d, cy_d = Kd[0,0], Kd[1,1], Kd[0,2], Kd[1,2]
    fx_c, fy_c, cx_c, cy_c = Kc[0,0], Kc[1,1], Kc[0,2], Kc[1,2]

    x_d  = (u - cx_d) * z / fx_d
    y_d  = (v - cy_d) * z / fy_d
    pts_d = np.stack([x_d, y_d, z, np.ones_like(z)], axis=0)
    pts_c = T_c_d @ pts_d
    Xc, Yc, Zc = pts_c[0], pts_c[1], pts_c[2]

    ok = (Zc > 1e-6) & np.isfinite(Zc)
    Xc, Yc, Zc = Xc[ok], Yc[ok], Zc[ok]

    ui = np.round((fx_c * (Xc / Zc) + cx_c) * sx).astype(np.int32)
    vi = np.round((fy_c * (Yc / Zc) + cy_c) * sy).astype(np.int32)
    inb = (ui >= 0) & (ui < Wp) & (vi >= 0) & (vi < Hp)
    ui, vi, Zc = ui[inb], vi[inb], Zc[inb]

    depth_warp = np.full((Hp, Wp), np.inf, dtype=np.float32)
    np.minimum.at(depth_warp.reshape(-1), vi * Wp + ui, Zc)
    depth_warp[~np.isfinite(depth_warp)] = 0.0
    return depth_warp

def save_ply(points: np.ndarray, colors: np.ndarray, save_path: str):
    N = len(points)
    with open(save_path, "w") as f:
        f.write(
            f"ply\nformat ascii 1.0\n"
            f"element vertex {N}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            "end_header\n"
        )
        for p, c in zip(points, colors):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                    f"{int(c[0])} {int(c[1])} {int(c[2])}\n")
