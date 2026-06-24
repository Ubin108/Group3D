import os
import re
import sys
import json
import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict
from typing import Optional
from PIL import Image
from tqdm import tqdm
import torch

from ..config import Group3DConfig, SAM3_ROOT
from ..utils.label_utils import normalize_label


def _safe_dirname(label: str) -> str:
    s = re.sub(r"[^\w\s\-\.]", "", label)
    s = s.strip().replace(" ", "_")
    return s if s else "unknown"


def _save_mask_png(save_path: str, mask_bool: np.ndarray):
    cv2.imwrite(save_path, mask_bool.astype(np.uint8) * 255)


def _normalize_mask(mask) -> np.ndarray:
    if hasattr(mask, "is_cuda"):
        mask = mask.detach().cpu().numpy()
    if mask.ndim == 3:
        mask = mask[0]
    return mask.astype(bool)


def _load_sam3(sam3_root: str):
    if sam3_root not in sys.path:
        sys.path.insert(0, sam3_root)

    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    model     = build_sam3_image_model().to("cuda").eval()
    processor = Sam3Processor(model)
    return processor


def build_scene_masks(
    scene: str,
    cfg: Group3DConfig,
    processor,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
):
    if end_frame is None:
        end_frame = cfg.num_frames

    ds         = cfg.dataset
    image_dir  = ds.image_dir(scene)
    label_path = ds.label_path(scene)
    mask_root  = ds.mask_root(scene)

    if os.path.isdir(mask_root) and len(os.listdir(mask_root)) > 0:
        print(f"[{scene}] Masks already exist, skipping → {mask_root}")
        return
    
    os.makedirs(mask_root, exist_ok=True)

    frame_objects = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            frame_objects.append(json.loads(line))

    for i in tqdm(range(start_frame, end_frame), desc=scene, leave=False):
        img_path = os.path.join(image_dir, f"{i:05d}.jpg")
        if not os.path.exists(img_path):
            continue

        image_pil = Image.open(img_path).convert("RGB")
        objects   = [normalize_label(o) for o in frame_objects[i]["objects"]]

        with torch.autocast("cuda", dtype=torch.bfloat16):
            state = processor.set_image(image_pil)

        label_to_masks = defaultdict(list)
        for obj in objects:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = processor.set_text_prompt(state=state, prompt=obj)
            for m, score in zip(out["masks"], out["scores"]):
                score = float(score)
                if score < cfg.sam_score_thresh:
                    continue
                label_to_masks[obj].append((_normalize_mask(m), score))

        frame_dir = os.path.join(mask_root, f"{i:05d}")
        os.makedirs(frame_dir, exist_ok=True)

        for label, ms in label_to_masks.items():
            label_dir = os.path.join(frame_dir, _safe_dirname(label))
            os.makedirs(label_dir, exist_ok=True)
            for k, (mask_bool, score) in enumerate(ms):
                save_path = os.path.join(label_dir, f"{k:02d}_s{score:.2f}.png")
                _save_mask_png(save_path, mask_bool)


def build_all_masks(
    cfg: Group3DConfig,
    start_scene: Optional[str] = None,
    end_scene: Optional[str] = None,
):
    processor = _load_sam3(SAM3_ROOT)

    root   = Path(cfg.dataset.root)
    scenes = sorted(d.name for d in root.iterdir() if d.is_dir())

    if start_scene:
        scenes = [s for s in scenes if s >= start_scene]
    if end_scene:
        scenes = [s for s in scenes if s <= end_scene]

    print(f"Processing {len(scenes)} scenes"
          + (f" from {start_scene}" if start_scene else "")
          + (f" to {end_scene}" if end_scene else ""))

    for scene in tqdm(scenes, desc="Building masks"):
        print("=" * 40)
        print(f"SCENE: {scene}")
        try:
            build_scene_masks(scene, cfg, processor)
        except Exception as e:
            print(f"[ERROR] {scene}: {e}")
