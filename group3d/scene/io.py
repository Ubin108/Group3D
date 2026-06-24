import json
import os
import random
from typing import List

import cv2
import numpy as np


def save_instances_json(instances: List[dict], filename: str):
    data = []
    for i, inst in enumerate(instances):
        data.append({
            "instance_id": i,
            "name": inst["name"],
            "score": float(inst.get("score", 1.0)),
            "centroid": inst["centroid"].tolist(),
            "bbox": {"min": inst["min"].tolist(), "max": inst["max"].tolist()},
            "size": inst["size"].tolist(),
            "num_points": int(inst["num_points"]),
            "frame": list(inst["frame_ids"]),
            "label_counts": dict(inst.get("label_counts", {})),
            "label_scores": dict(inst.get("label_scores", {})),
        })
    if os.path.dirname(filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)


def save_instances_to_ply(instances: List[dict], filename: str, ratio: float = 0.1):
    pts_all, col_all = [], []
    for inst in instances:
        pts = inst["points"]
        n = int(len(pts) * ratio)
        if n == 0:
            continue
        idx = np.random.choice(len(pts), n, replace=False)
        pts_all.append(pts[idx])
        col_all.append(np.tile(np.random.randint(0, 255, 3), (n, 1)))

    if not pts_all:
        print("No points to save:", filename)
        return

    pts_all = np.vstack(pts_all)
    col_all = np.vstack(col_all)

    with open(filename, "w") as f:
        f.write(
            "ply\nformat ascii 1.0\n"
            f"element vertex {len(pts_all)}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            "end_header\n"
        )
        for p, c in zip(pts_all, col_all):
            f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def color_from_id(idx: int) -> np.ndarray:
    random.seed(idx)
    return np.array([random.randint(50, 255) for _ in range(3)], dtype=np.uint8)


def draw_xy_bboxes(canvas: np.ndarray, instances: List[dict], to_pixel) -> np.ndarray:
    img = canvas.copy()
    placed_labels = []

    for inst_id, inst in enumerate(instances):
        min_xy, max_xy = inst["min"][:2], inst["max"][:2]
        corners = np.array([
            [min_xy[0], min_xy[1]], [max_xy[0], min_xy[1]],
            [max_xy[0], max_xy[1]], [min_xy[0], max_xy[1]],
        ])
        px, py = to_pixel(corners)
        color = color_from_id(inst_id).tolist()

        for i in range(4):
            cv2.line(img, (px[i], py[i]), (px[(i + 1) % 4], py[(i + 1) % 4]), color, 2)

        label = f"{inst_id}:{inst['name']}"
        font, font_scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        x, y = int(px.mean() - tw / 2), max(th + 2, int(py.min() - 5))

        def overlaps(b1, b2):
            x1, y1, w1, h1 = b1
            x2, y2, w2, h2 = b2
            return not (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1)

        label_box = (x, y - th, tw, th)
        while any(overlaps(label_box, b) for b in placed_labels):
            y += th + 4
            label_box = (x, y - th, tw, th)
        placed_labels.append(label_box)

        cv2.rectangle(img, (x, y - th - baseline), (x + tw, y + baseline), color, -1)
        cv2.putText(img, label, (x, y), font, font_scale, (0, 0, 0), thickness)

    return img
