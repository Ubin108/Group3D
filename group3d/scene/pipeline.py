import os
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from ..config import Group3DConfig
from ..utils.label_utils import normalize_label
from ..utils.mask_utils import erode_mask, suppress_duplicate_masks_2d, load_saved_masks
from ..core.instance import build_3d_instance, choose_final_label, clean_instance
from ..core.merge import merge_loop_voxel, post_merge_instances
from .io import save_instances_json


def run_scene(scene: str, cfg: Group3DConfig):
    ds = cfg.dataset

    image_dir   = ds.image_dir(scene)
    points_path = os.path.join(ds.points_dir(scene), "points.npy")
    conf_path   = os.path.join(ds.points_dir(scene), "conf.npy")
    depth_path  = os.path.join(ds.points_dir(scene), "depth.npy")
    
    label_path  = ds.label_path(scene)
    mask_root   = ds.mask_root(scene)

    points_world = np.load(points_path)
    points_conf  = np.load(conf_path)
    pixel_depth  = np.load(depth_path)

    conf_thresh     = np.percentile(points_conf, cfg.conf_percentile)
    valid_conf_mask = points_conf >= conf_thresh

    frame_objects = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            frame_objects.append(json.loads(line))
    
    instances_3d = {i: [] for i in range(cfg.num_frames)}

    for i in range(cfg.num_frames):
        img_path = os.path.join(image_dir, f"{i:05d}.jpg")
        if not os.path.exists(img_path):
            continue

        image_pil = Image.open(img_path).convert("RGB")
        objects = [normalize_label(o) for o in frame_objects[i]["objects"]]

        frame_dir = os.path.join(mask_root, f"{i:05d}")
        masks = load_saved_masks(frame_dir, wanted_labels=objects,
                                 sam_score_thresh=cfg.sam_score_thresh)
        masks = [(erode_mask(m, 2), lbl, score) for m, lbl, score in masks]
        masks = suppress_duplicate_masks_2d(masks)

        for mid, (mask, label, score) in enumerate(masks):
            valid = mask & valid_conf_mask[i]
            pts   = points_world[i][valid]
            depth = pixel_depth[i][valid]

            if depth.size == 0:
                continue

            finite_mask = np.isfinite(pts).all(axis=-1)
            pts         = pts[finite_mask]
            depth       = depth[finite_mask]
            colors_raw  = np.array(image_pil)[valid][finite_mask].astype(np.float32) / 255.0

            if len(pts) == 0:
                continue

            depth_far  = np.percentile(depth, 95)
            depth_near = np.percentile(depth, 5)
            depth_mask = (depth < depth_far) & (depth > depth_near)

            pts    = pts[depth_mask]
            colors = colors_raw[depth_mask]

            if len(pts) < 100:
                continue

            inst = build_3d_instance(mid, pts, colors, label, i,
                                     voxel_size=cfg.voxel_size, score=score)
            if inst is not None:
                instances_3d[i].append(inst)

    global_instances, label_group = merge_loop_voxel(
        instances_3d,
        force_merge_labels=set(cfg.merge.force_merge_labels),
        gpt_model=cfg.vlm.gpt_model,
        voxel_iou_thresh=cfg.merge.voxel_iou_thresh,
        voxel_contain_thresh=cfg.merge.voxel_contain_thresh,
    )

    for inst in global_instances:
        label, score = choose_final_label(inst, mode="score", return_score=True)
        inst["name"]  = label
        inst["score"] = float(score)

    cleaned = []
    for inst in global_instances:
        if len(inst.get("frame_ids", [])) <= 3:
            continue
        new_inst = clean_instance(inst, voxel_size=cfg.voxel_size)
        if new_inst is not None:
            cleaned.append(new_inst)

    instances_sorted = sorted(cleaned, key=lambda x: x["min"][2])

    os.makedirs(cfg.output.dir, exist_ok=True)
    out_path = os.path.join(cfg.output.dir, f"scene_instances_{scene}{cfg.output.json_suffix}")
    save_instances_json(instances_sorted, out_path)

    print(f"[{scene}] Saved {len(instances_sorted)} instances → {out_path}")
    return instances_sorted


def run_all(cfg: Group3DConfig):
    root = Path(cfg.dataset.root)
    scenes = sorted(d.name for d in root.iterdir() if d.is_dir())

    for scene in tqdm(scenes, desc="Scenes"):
        print("=" * 40)
        print(f"SCENE: {scene}")
        try:
            run_scene(scene, cfg)
        except Exception as e:
            print(f"[ERROR] {scene}: {e}")
