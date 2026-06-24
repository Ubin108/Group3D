"""
Group3D entry point.

Usage:
    # Run full pipeline in order
    python run.py

    # Select a specific stage
    python run.py --stage vocab
    python run.py --stage mask
    python run.py --stage detect

    # Single scene only
    python run.py --scene scene0011_00 --stage vocab
    python run.py --scene scene0011_00 --stage mask
    python run.py --scene scene0011_00

    # Specify scene range for mask generation
    python run.py --stage mask --start_scene scene0633_00 --end_scene scene0704_01
"""

import argparse

from group3d import (
    load_config,
    run_scene, run_all,
    build_scene_vocab, build_all_vocabs,
    build_scene_masks, build_all_masks,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Group3D: MLLM-Driven 3D Instance Detection")
    parser.add_argument("--config", type=str, default="configs/pose_free.yaml",
                        help="Path to YAML config file (default: config.yaml)")
    parser.add_argument("--stage", type=str, default="all",
                        choices=["all", "vocab", "mask", "detect"],
                        help="Pipeline stage to run (default: all)")
    parser.add_argument("--scene", type=str, default=None,
                        help="Run a single scene instead of all scenes")
    parser.add_argument("--start_scene", type=str, default=None,
                        help="(mask stage) Start scene name, inclusive (e.g. scene0633_00)")
    parser.add_argument("--end_scene", type=str, default=None,
                        help="(mask stage) End scene name, inclusive (e.g. scene0704_01)")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg  = load_config(args.config)

    run_vocab  = args.stage in ("all", "vocab")
    run_mask   = args.stage in ("all", "mask")
    run_detect = args.stage in ("all", "detect")

    if args.scene:
        if run_vocab:
            build_scene_vocab(args.scene, cfg)
        if run_mask:
            from group3d.scene.mask import _load_sam3, SAM3_ROOT
            processor = _load_sam3(SAM3_ROOT)
            build_scene_masks(args.scene, cfg, processor)
        if run_detect:
            run_scene(args.scene, cfg)
    else:
        if run_vocab:
            build_all_vocabs(cfg)
        if run_mask:
            build_all_masks(cfg,
                            start_scene=args.start_scene,
                            end_scene=args.end_scene)
        if run_detect:
            run_all(cfg)


if __name__ == "__main__":
    main()
