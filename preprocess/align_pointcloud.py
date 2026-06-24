import argparse
import glob
import os
import shutil
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

from _preprocess_utils import (
    to4x4,
    load_axis_alignment,
    load_color_K_and_size,
    scale_intrinsics,
    load_gt_depth_png,
    estimate_scale_from_depth,
    warp_depth_to_pred_frame,
    backproject_depth,
    cam_to_world,
    save_ply,
)

_OUT_SUBDIR = {
    "pose_free":  "point_cloud_aligned_pose_free",
    "pose_known": "point_cloud_aligned_pose_known",
}

def _compute_alignment_transform(
    gt_pose_txt_path: str,
    scene_txt_path: str,
    pred_pose_3x4: np.ndarray,
    gt_convention: str = "c2w",
    pred_convention: str = "w2c",
) -> np.ndarray:
    """
    Compute the 4x4 rigid transform T that maps the predicted coordinate
    frame to the GT coordinate frame, using the first-frame poses.
    """
    C_gt   = np.loadtxt(gt_pose_txt_path)
    E_pred = to4x4(pred_pose_3x4)
    A      = load_axis_alignment(scene_txt_path)

    E_gt = np.linalg.inv(A @ C_gt) if gt_convention == "c2w" else C_gt @ np.linalg.inv(A)
    if pred_convention == "c2w":
        E_pred = np.linalg.inv(E_pred)

    return np.linalg.inv(E_gt) @ E_pred


def _build_pose_free(scene_id: str, dataset_root: str):
    root_dir  = Path(dataset_root) / scene_id
    src_dir   = root_dir / "point_cloud"
    final_dir = root_dir / _OUT_SUBDIR["pose_free"]
    final_dir.mkdir(parents=True, exist_ok=True)

    scene_txt_path = f"{dataset_root}/{scene_id}/{scene_id}.txt"
    gt_pose_path   = f"{dataset_root}/{scene_id}/pose/0.txt"
    gt_depth_path  = f"{dataset_root}/{scene_id}/depth/0.png"

    depth_raw     = np.load(str(src_dir / "depth.npy"))
    gt_depth_raw  = load_gt_depth_png(gt_depth_path)
    gt_depth_warp = warp_depth_to_pred_frame(gt_depth_raw, scene_txt_path,
                                              pred_shape_hw=depth_raw[0].shape)
    scale = estimate_scale_from_depth(depth_raw[0], gt_depth_warp)
    print(f"  [{scene_id}] scale = {scale:.4f}")

    pts = np.load(str(src_dir / "points.npy")) * scale
    ext = np.load(str(src_dir / "extrinsics.npy")).copy()
    ext[:, :3, 3] *= scale

    T    = _compute_alignment_transform(gt_pose_path, scene_txt_path, ext[0])
    R, t = T[:3, :3], T[:3, 3]
    pts_aligned = (pts.reshape(-1, 3) @ R.T + t).reshape(pts.shape)
    E_new       = ext @ np.linalg.inv(T)

    np.save(str(final_dir / "points.npy"), pts_aligned.astype(np.float32))
    np.save(str(final_dir / "extrinsics.npy"),   E_new.astype(np.float32))
    np.save(str(final_dir / "depth.npy"),        (depth_raw * scale).astype(np.float32))

    conf = np.load(str(src_dir / "conf.npy"))
    np.save(str(final_dir / "conf.npy"), conf)

    intrinsic_files = list(src_dir.glob("intrinsics*.npy"))
    if intrinsic_files:
        shutil.copy(intrinsic_files[0], final_dir / "intrinsics.npy")

    _save_vis_ply(final_dir, pts_aligned, root_dir, conf)
    print(f"  [{scene_id}] Saved -> {final_dir}")


def _build_pose_known(scene_id: str, dataset_root: str):
    root_dir  = Path(dataset_root) / scene_id
    src_dir   = root_dir / "point_cloud"
    final_dir = root_dir / _OUT_SUBDIR["pose_known"]
    final_dir.mkdir(parents=True, exist_ok=True)

    scene_txt_path = f"{dataset_root}/{scene_id}/{scene_id}.txt"
    pose_dir       = Path(f"{dataset_root}/{scene_id}/pose")

    depth = np.load(str(src_dir / "depth.npy")).astype(np.float32)
    gt_depth_raw  = load_gt_depth_png(f"{dataset_root}/{scene_id}/depth/0.png")
    gt_depth_warp = warp_depth_to_pred_frame(gt_depth_raw, scene_txt_path,
                                              pred_shape_hw=depth[0].shape)
    scale = estimate_scale_from_depth(depth[0], gt_depth_warp)
    print(f"  [{scene_id}] scale = {scale:.4f}")
    depth *= scale

    Tn, Hp, Wp = depth.shape
    Kc_native, (cw, ch) = load_color_K_and_size(scene_txt_path)
    K_pred = scale_intrinsics(Kc_native, src_wh=(cw, ch), dst_wh=(Wp, Hp))
    np.save(str(final_dir / "intrinsics.npy"), K_pred.astype(np.float32))

    video_idx = np.loadtxt(str(root_dir / "video_idx_128.txt"), dtype=int)
    assert len(video_idx) == Tn

    A = load_axis_alignment(scene_txt_path).astype(np.float32)

    pts_world      = np.full((Tn, Hp, Wp, 3), np.nan, dtype=np.float32)
    extrinsics_w2c = np.zeros((Tn, 4, 4), dtype=np.float32)

    for i in range(Tn):
        frame_id  = int(video_idx[i])
        pose_path = pose_dir / f"{frame_id}.txt"
        if not pose_path.exists():
            pose_path = pose_dir / f"{frame_id:05d}.txt"

        if not pose_path.exists():
            print(f"  [WARN] missing pose: {scene_id} frame {frame_id}")
            extrinsics_w2c[i] = np.eye(4, dtype=np.float32)
            continue

        C_gt_raw = to4x4(np.loadtxt(str(pose_path))).astype(np.float32)
        if not np.isfinite(C_gt_raw).all():
            print(f"  [WARN] non-finite pose: {scene_id} frame {frame_id}")
            extrinsics_w2c[i] = np.eye(4, dtype=np.float32)
            continue

        C = A @ C_gt_raw
        E = np.linalg.inv(C).astype(np.float32)
        if not np.isfinite(E).all():
            print(f"  [WARN] non-finite w2c: {scene_id} frame {frame_id}")
            extrinsics_w2c[i] = np.eye(4, dtype=np.float32)
            continue

        extrinsics_w2c[i] = E

        z       = depth[i].astype(np.float32)
        pts_cam = backproject_depth(z, K_pred)
        pw      = cam_to_world(pts_cam, C)
        pw[~np.isfinite(z)]                   = np.nan
        pw[~np.isfinite(pw).all(axis=-1)]     = np.nan
        pts_world[i] = pw

    np.save(str(final_dir / "points.npy"), pts_world)
    np.save(str(final_dir / "extrinsics.npy"),   extrinsics_w2c)
    np.save(str(final_dir / "depth.npy"),        depth)

    conf = np.load(str(src_dir / "conf.npy"))
    np.save(str(final_dir / "conf.npy"), conf)

    _save_vis_ply(final_dir, pts_world, root_dir, conf,
                  finite_only=True)
    print(f"  [{scene_id}] Saved -> {final_dir}")


def _save_vis_ply(final_dir: Path, pts: np.ndarray,
                  root_dir: Path, conf: np.ndarray,
                  finite_only: bool = False):
    color_dir = root_dir / "video_color_128"
    rgb_stack = np.stack([
        np.array(Image.open(p).convert("RGB"))
        for p in sorted(glob.glob(str(color_dir / "*.jpg")))
    ], axis=0)

    pts_flat  = pts.reshape(-1, 3)
    cols_flat = rgb_stack.reshape(-1, 3)
    conf_flat = conf.reshape(-1)

    mask = conf_flat > np.quantile(conf_flat, 0.2)
    if finite_only:
        mask &= np.isfinite(pts_flat).all(axis=-1)

    pts_m, cols_m = pts_flat[mask], cols_flat[mask]
    if len(pts_m) > 500_000:
        idx   = np.random.choice(len(pts_m), 500_000, replace=False)
        pts_m = pts_m[idx]; cols_m = cols_m[idx]

    save_ply(pts_m, cols_m, str(final_dir / "pointcloud_vis.ply"))

def build_scene(scene_id: str, pose_mode: str, dataset_root: str):
    """Build point cloud for a single scene according to pose_mode."""
    final_dir = Path(dataset_root) / scene_id / _OUT_SUBDIR[pose_mode]
    if (final_dir / "points.npy").exists():
        print(f"  [{scene_id}] Already processed, skipping.")
        return

    if pose_mode == "pose_free":
        _build_pose_free(scene_id, dataset_root)
    else:
        _build_pose_known(scene_id, dataset_root)

def main():
    parser = argparse.ArgumentParser(
        description="Build point clouds from DA3 outputs (pose-free or pose-known)"
    )
    parser.add_argument("--pose_mode", type=str, default="pose_free",
                        choices=["pose_free", "pose_known"],
                        help="pose_free: align DA3 pred pose to GT first frame | "
                             "pose_known: back-project with GT pose")
    parser.add_argument("--dataset_root", type=str,
                        default="/hdd/ubin108/ECCV/Group3D_release/data/ScanNetv2/scans")
    parser.add_argument("--scene", type=str, default=None,
                        help="Process a single scene (e.g. scene0000_00)")
    parser.add_argument("--start_scene", type=str, default=None,
                        help="Resume from this scene name (inclusive)")
    args = parser.parse_args()

    use_dist = "LOCAL_RANK" in os.environ
    rank, world_size = 0, 1

    if use_dist:
        import torch
        import torch.distributed as dist
        dist.init_process_group(backend="nccl")
        rank       = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    if args.scene:
        build_scene(args.scene, args.pose_mode, args.dataset_root)
    else:
        root   = Path(args.dataset_root)
        scenes = sorted(d.name for d in root.iterdir() if d.is_dir())

        if args.start_scene:
            start_idx = next(
                (i for i, s in enumerate(scenes) if s == args.start_scene), 0
            )
            scenes = scenes[start_idx:]

        scenes_for_rank = scenes[rank::world_size]

        for scene in tqdm(scenes_for_rank, disable=(rank != 0),
                          desc=f"Building ({args.pose_mode})"):
            try:
                build_scene(scene, args.pose_mode, args.dataset_root)
            except Exception as e:
                print(f"[ERROR] {scene}: {e}")

    if use_dist:
        import torch.distributed as dist
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
