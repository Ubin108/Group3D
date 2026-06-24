import argparse
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm


def process_scene(
    scene: Path,
    out_root: Path,
    target_n: int,
    width: int,
    height: int,
    img_subdir: str = "color",
):
    img_dir = scene / img_subdir
    frames  = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))

    if len(frames) == 0:
        print(f"[SKIP] No frames found: {img_dir}")
        return

    idxs = (
        np.linspace(0, len(frames) - 1, target_n, dtype=int)
        if len(frames) > target_n
        else np.arange(len(frames), dtype=int)
    )

    out_scene = out_root / scene.name
    out_dir   = out_scene / f"video_color_{target_n}"

    if out_dir.exists() and len(list(out_dir.glob("*.jpg"))) == len(idxs):
        print(f"[{scene.name}] Already exists, skipping -> {out_dir}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_scene / f"video_idx_{target_n}.txt", idxs, fmt="%d")

    for new_idx, original_idx in enumerate(tqdm(idxs, desc=scene.name, leave=False)):
        img_path = img_dir / f"{original_idx:05d}.jpg"
        if not img_path.exists():
            img_path = img_dir / f"{original_idx}.jpg"
        if not img_path.exists():
            print(f"  [WARN] Not found: {img_path}")
            continue

        Image.open(img_path).convert("RGB") \
             .resize((width, height), Image.BILINEAR) \
             .save(out_dir / f"{new_idx:05d}.jpg")

    print(f"[{scene.name}] {len(idxs)} frames -> {out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Uniformly sample N frames and resize them"
    )
    parser.add_argument("--root", type=str,
                        default="/hdd/ubin108/ECCV/Group3D_release/data/ScanNetv2/scans")
    parser.add_argument("--out_root", type=str, default=None,
                        help="Output root directory (default: same as --root)")
    parser.add_argument("--target_n",   type=int, default=128)
    parser.add_argument("--width",      type=int, default=504)
    parser.add_argument("--height",     type=int, default=378)
    parser.add_argument("--img_subdir", type=str, default="color")
    parser.add_argument("--scene", type=str, default=None,
                        help="Process a single scene (e.g. scene0000_00)")
    args = parser.parse_args()

    root     = Path(args.root)
    out_root = Path(args.out_root) if args.out_root else root

    if args.scene:
        process_scene(root / args.scene, out_root,
                      args.target_n, args.width, args.height, args.img_subdir)
    else:
        scenes = sorted(d for d in root.iterdir() if d.is_dir())
        for scene in tqdm(scenes, desc="Processing scenes"):
            try:
                process_scene(scene, out_root,
                              args.target_n, args.width, args.height, args.img_subdir)
            except Exception as e:
                print(f"[ERROR] {scene.name}: {e}")


if __name__ == "__main__":
    main()
