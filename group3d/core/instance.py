import math
import numpy as np
from collections import Counter, defaultdict
from sklearn.neighbors import NearestNeighbors
from typing import Optional

from .voxel import voxelize_points


def build_3d_instance(
    mask_id: int,
    points: np.ndarray,
    colors: np.ndarray,
    label: str,
    frame: int,
    voxel_size: float,
    score: float = None,
) -> Optional[dict]:
    if len(points) == 0:
        return None

    inst = {
        "mask_id": mask_id,
        "name": label,
        "points": points,
        "colors": colors,
        "voxels": voxelize_points(points, voxel_size),
        "min": points.min(axis=0),
        "max": points.max(axis=0),
        "frame_ids": {frame},
        "frame_mask_ids": {frame: {mask_id}},
        "label_counts": Counter([label]),
        "label_scores": defaultdict(float),
    }
    if score is not None:
        inst["label_scores"][label] += float(score)

    inst["centroid"] = points.mean(axis=0)
    inst["size"] = inst["max"] - inst["min"]
    inst["num_points"] = len(points)
    return inst


def merge_instances(a: dict, b: dict) -> dict:
    pts = np.concatenate([a["points"], b["points"]], axis=0)
    colors = np.concatenate([a["colors"], b["colors"]], axis=0)

    new_frame_mask_ids = {}
    for f, masks in a["frame_mask_ids"].items():
        new_frame_mask_ids[f] = set(masks)
    for f, masks in b["frame_mask_ids"].items():
        if f not in new_frame_mask_ids:
            new_frame_mask_ids[f] = set(masks)
        else:
            new_frame_mask_ids[f].update(masks)

    merged = {
        "name": a["name"],
        "points": pts,
        "colors": colors,
        "voxels": a["voxels"] | b["voxels"],
        "min": np.minimum(a["min"], b["min"]),
        "max": np.maximum(a["max"], b["max"]),
        "frame_ids": a["frame_ids"] | b["frame_ids"],
        "frame_mask_ids": new_frame_mask_ids,
        "label_counts": a.get("label_counts", Counter()) + b.get("label_counts", Counter()),
        "label_scores": defaultdict(float),
    }
    for k, v in a.get("label_scores", {}).items():
        merged["label_scores"][k] += float(v)
    for k, v in b.get("label_scores", {}).items():
        merged["label_scores"][k] += float(v)

    merged["centroid"] = pts.mean(axis=0)
    merged["size"] = merged["max"] - merged["min"]
    merged["num_points"] = len(pts)
    return merged


def label_conf(lc: Counter, ls: dict, label: str, tau: float = 3.0) -> float:
    c = int(lc.get(label, 0))
    if c <= 0:
        return 0.0
    mean_conf = float(ls.get(label, 0.0)) / c
    satur = 1.0 - math.exp(-c / tau)
    return mean_conf * satur


def choose_final_label(inst: dict, mode: str = "score", return_score: bool = False):
    lc = inst.get("label_counts", Counter())
    ls = inst.get("label_scores", {})

    if not lc:
        return (inst["name"], 0.0) if return_score else inst["name"]

    if mode == "count":
        best_label = lc.most_common(1)[0][0]
        best_score = float(lc[best_label])
    elif mode == "hybrid":
        best_label = max(lc.keys(), key=lambda l: (lc[l], label_conf(lc, ls, l)))
        best_score = label_conf(lc, ls, best_label)
    else:  # score (default)
        best_label = max(lc.keys(), key=lambda l: label_conf(lc, ls, l))
        best_score = label_conf(lc, ls, best_label)

    return (best_label, best_score) if return_score else best_label


def remove_statistical_outliers(points: np.ndarray, k: int = 10, std_ratio: float = 1.5) -> np.ndarray:
    if len(points) < k:
        return np.ones(len(points), dtype=bool)
    nbrs = NearestNeighbors(n_neighbors=k).fit(points)
    distances, _ = nbrs.kneighbors(points)
    mean_dist = distances.mean(axis=1)
    threshold = mean_dist.mean() + std_ratio * mean_dist.std()
    return mean_dist < threshold


def clean_instance(inst: dict, voxel_size: float, k: int = 10, std_ratio: float = 1.5) -> Optional[dict]:
    points = inst["points"]
    if len(points) == 0:
        return inst

    mask = remove_statistical_outliers(points, k=k, std_ratio=std_ratio)
    points = points[mask]
    if len(points) == 0:
        return None

    inst["points"] = points
    inst["min"] = points.min(axis=0)
    inst["max"] = points.max(axis=0)
    inst["voxels"] = voxelize_points(points, voxel_size)
    inst["centroid"] = points.mean(axis=0)
    inst["size"] = inst["max"] - inst["min"]
    inst["num_points"] = len(points)
    return inst
