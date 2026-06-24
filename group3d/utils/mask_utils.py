import os
import re
import cv2
import random
import numpy as np
from PIL import Image
from typing import List, Tuple


_score_pat = re.compile(r"(?:_s|^s)(\d+(?:\.\d+)?)")


def parse_score_from_name(fname: str, default: float = 1.0) -> float:
    m = _score_pat.search(fname)
    return float(m.group(1)) if m else default


def erode_mask(mask_np: np.ndarray, iterations: int = 1) -> np.ndarray:
    if iterations <= 0:
        return mask_np
    kernel = np.ones((3, 3), np.uint8)
    return cv2.erode(mask_np.astype(np.uint8), kernel, iterations=iterations) > 0


def suppress_duplicate_masks_2d(
    masks: List[Tuple],
    iou_thresh: float = 0.1,
    contain_thresh: float = 0.4,
) -> List[Tuple]:
    if len(masks) <= 1:
        return masks

    masks = sorted(masks, key=lambda x: x[0].sum(), reverse=True)
    kept = []

    for mask, label, score in masks:
        area = mask.sum()
        discard = False
        for kept_mask, _, _ in kept:
            inter = np.logical_and(mask, kept_mask).sum()
            if inter == 0:
                continue
            union = np.logical_or(mask, kept_mask).sum()
            iou = inter / union
            contain = inter / area
            if iou >= iou_thresh or contain >= contain_thresh:
                discard = True
                break
        if not discard:
            kept.append((mask, label, score))

    return kept


def load_saved_masks(
    frame_dir: str,
    wanted_labels=None,
    sam_score_thresh: float = 0.5,
) -> List[Tuple]:
    masks = []
    if not os.path.isdir(frame_dir):
        return masks

    wanted = set(wanted_labels) if wanted_labels is not None else None

    for label in sorted(os.listdir(frame_dir)):
        label_dir = os.path.join(frame_dir, label)
        if not os.path.isdir(label_dir):
            continue

        norm_label = label.replace("_", " ").strip()
        if wanted is not None and norm_label not in wanted:
            continue

        for fname in sorted(os.listdir(label_dir)):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            fpath = os.path.join(label_dir, fname)
            m = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue

            mask_bool = m > 127
            score = parse_score_from_name(fname, default=1.0)
            if score < sam_score_thresh:
                continue
            if not mask_bool.any():
                continue

            masks.append((mask_bool, norm_label, score))

    return masks
