import argparse
import os
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm


def unproject(depth_map: np.ndarray, intrinsic: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    h, w  = depth_map.shape
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    zs    = depth_map.flatten()

    xs = (xs.flatten() - intrinsic[0, 2]) * zs / intrinsic[0, 0]
    ys = (ys.flatten() - intrinsic[1, 2]) * zs / intrinsic[1, 1]
    pts_cam = np.stack([xs, ys, zs], axis=1)

    R_wc = extrinsic[:3, :3]
    t_wc = extrinsic[:3,  3]
    R_cw = R_wc.T
    t_cw = -R_cw @ t_wc

    pts_world = pts_cam @ R_cw.T + t_cw
    return pts_world.reshape(h, w, 3)


def run_scene(model, scene_dir: Path, target_n: int, process_res: int):
    img_dir = scene_dir / f"video_color_{target_n}"
    out_dir = scene_dir / "point_cloud"

    if (out_dir / "points.npy").exists():
        print(f"[{scene_dir.name}] Already processed, skipping.")
        return

    frames = sorted(img_dir.glob("*.jpg"))
    if not frames:
        print(f"[SKIP] No frames found: {img_dir}")
        return

    imgs = []
    for p in frames:
        try:
            imgs.append(Image.open(p).convert("RGB"))
        except Exception as e:
            print(f"  [WARN] Cannot open {p}: {e}")

    if not imgs:
        print(f"  [WARN] No valid frames: {scene_dir.name}")
        return

    print(f"[{scene_dir.name}] Running DA3 on {len(imgs)} frames ...")

    import torch
    with torch.no_grad():
        pred = model.inference(
            imgs,
            export_dir=None,
            process_res=process_res,
            export_format="npz",
        )

    all_points, all_conf, all_depth, all_ext = [], [], [], []

    for i in range(len(imgs)):
        depth = pred.depth[i]       # (H, W)
        conf  = pred.conf[i]        # (H, W)
        K     = pred.intrinsics[i]  # (3, 3)
        T     = pred.extrinsics[i]  # (3, 4) or (4, 4)

        pc = unproject(depth, K, T)

        all_points.append(pc)
        all_conf.append(conf)
        all_depth.append(depth)
        all_ext.append(T if T.shape == (4, 4) else np.vstack([T, [0, 0, 0, 1]]))

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(out_dir / "points.npy"),
            np.stack(all_points, axis=0).astype(np.float32))
    np.save(str(out_dir / "extrinsics.npy"),
            np.stack(all_ext,    axis=0).astype(np.float32))
    np.save(str(out_dir / "depth.npy"),
            np.stack(all_depth,  axis=0).astype(np.float32))
    np.save(str(out_dir / "conf.npy"),
            np.stack(all_conf,   axis=0).astype(np.float32))

    print(f"[{scene_dir.name}] Saved -> {out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Build point clouds with Depth Anything 3 (DA3)"
    )
    parser.add_argument("--root", type=str,
                        default="/hdd/ubin108/ECCV/Group3D_release/data/ScanNetv2/scans")
    parser.add_argument("--target_n",    type=int, default=128,
                        help="Number of frames per scene (must match sample_and_extract.py)")
    parser.add_argument("--process_res", type=int, default=504,
                        help="DA3 inference resolution")
    parser.add_argument("--model_name",  type=str,
                        default="depth-anything/da3nested-giant-large")
    parser.add_argument("--scene", type=str, default=None,
                        help="Process a single scene (e.g. scene0000_00)")
    parser.add_argument("--start_scene", type=str, default=None,
                        help="Resume from this scene name (inclusive)")
    args = parser.parse_args()

    import torch
    import torch.distributed as dist
    from depth_anything_3.api import DepthAnything3

    use_dist = "LOCAL_RANK" in os.environ
    rank, world_size = 0, 1

    if use_dist:
        dist.init_process_group(backend="nccl")
        rank       = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")
    model  = DepthAnything3.from_pretrained(
        args.model_name, torch_dtype=torch.float32
    ).to(device)

    root = Path(args.root)

    if args.scene:
        run_scene(model, root / args.scene, args.target_n, args.process_res)
    else:
        scenes = sorted(d for d in root.iterdir() if d.is_dir())

        if args.start_scene:
            start_idx = next(
                (i for i, s in enumerate(scenes) if s.name == args.start_scene), 0
            )
            scenes = scenes[start_idx:]

        scenes_for_rank = scenes[rank::world_size]

        for scene in tqdm(scenes_for_rank, disable=(rank != 0), desc="Building DA3 points"):
            try:
                run_scene(model, scene, args.target_n, args.process_res)
            except Exception as e:
                print(f"[ERROR] {scene.name}: {e}")

    if use_dist:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
