import os
import re
import json
import argparse

import numpy as np
from plyfile import PlyData
from tqdm import tqdm

from scannet20_classes import OV3DET_CLASSES, NAME_TO_ID_20, NYU40_TO_OV3DET
from scannet200_classes import VALID_CLASS_IDS_200, LABEL_TO_ID_200, ID_TO_LABEL_200

_VALID_CLASS_IDS_200_SET = set(VALID_CLASS_IDS_200)

IOU_THRESHOLDS = [0.25, 0.50]

def normalize_name(name):
    s = name.strip().lower().replace("_", " ")
    return re.sub(r"\s+", " ", s)


def parse_bbox(obj):
    if "bbox" in obj:
        b = obj["bbox"]
        if isinstance(b, dict) and "min" in b and "max" in b:
            mn = np.array(b["min"], dtype=np.float32)
            mx = np.array(b["max"], dtype=np.float32)
            if mn.shape == (3,) and mx.shape == (3,):
                box = np.concatenate([mn, mx])
            else:
                return None
        elif isinstance(b, (list, tuple)) and len(b) == 6:
            box = np.array(b, dtype=np.float32)
        else:
            return None
    elif "bbox_min" in obj and "bbox_max" in obj:
        mn = np.array(obj["bbox_min"], dtype=np.float32)
        mx = np.array(obj["bbox_max"], dtype=np.float32)
        if mn.shape != (3,) or mx.shape != (3,):
            return None
        box = np.concatenate([mn, mx])
    else:
        return None

    if np.any(np.isnan(box)) or np.any(np.isinf(box)):
        return None
    if not np.all(box[3:] > box[:3]):
        return None
    return box


def iou_3d(box, boxes):
    inter_min = np.maximum(box[:3], boxes[:, :3])
    inter_max = np.minimum(box[3:], boxes[:, 3:])
    inter_dim = np.clip(inter_max - inter_min, 0.0, None)
    inter_vol = inter_dim[:, 0] * inter_dim[:, 1] * inter_dim[:, 2]
    vol1 = float(np.prod(np.clip(box[3:] - box[:3], 0.0, None)))
    vol2 = np.prod(np.clip(boxes[:, 3:] - boxes[:, :3], 0.0, None), axis=1)
    union = vol1 + vol2 - inter_vol
    iou = np.zeros_like(union, dtype=np.float32)
    valid = union > 1e-9
    iou[valid] = inter_vol[valid] / union[valid]
    return iou


def calculate_ap(precisions, recalls):
    m_prec = np.concatenate(([0.0], precisions, [0.0]))
    m_rec = np.concatenate(([0.0], recalls, [1.0]))
    for i in range(len(m_prec) - 1, 0, -1):
        m_prec[i - 1] = np.maximum(m_prec[i - 1], m_prec[i])
    idx = np.where(m_rec[1:] != m_rec[:-1])[0]
    return float(np.sum((m_rec[idx + 1] - m_rec[idx]) * m_prec[idx + 1]))


def compute_ap_for_class(dets, gt_by_scene, thr):
    npos = sum(v.shape[0] for v in gt_by_scene.values())
    if npos == 0 or len(dets) == 0:
        return 0.0

    dets = sorted(dets, key=lambda x: x[0], reverse=True)
    matched = {sid: np.zeros(gt.shape[0], dtype=bool) for sid, gt in gt_by_scene.items()}
    tp = np.zeros(len(dets), dtype=np.float32)
    fp = np.zeros(len(dets), dtype=np.float32)

    for i, (_, sid, p_box) in enumerate(dets):
        gts = gt_by_scene.get(sid)
        if gts is None or gts.shape[0] == 0:
            fp[i] = 1.0
            continue
        ious = iou_3d(p_box, gts)
        j = int(np.argmax(ious))
        if float(ious[j]) >= thr and not matched[sid][j]:
            tp[i] = 1.0
            matched[sid][j] = True
        else:
            fp[i] = 1.0

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recalls = tp_cum / float(npos)
    precisions = tp_cum / (tp_cum + fp_cum + 1e-8)
    return calculate_ap(precisions, recalls)

def pick_score(obj):
    if "score" in obj:
        try:
            return float(obj["score"])
        except Exception:
            pass
    try:
        return float(obj["label_scores"][obj["name"]])
    except Exception:
        return 1.0


def pred_to_cls_20(obj):
    if "class_id" in obj:
        try:
            cid = int(obj["class_id"])
            if 0 <= cid < 20:
                return cid
        except Exception:
            pass
    if "name" not in obj:
        return None
    name = normalize_name(str(obj["name"]))
    return NAME_TO_ID_20.get(name)


def rotz(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def obb_to_aabb(center, half, heading):
    hx, hy, hz = float(half[0]), float(half[1]), float(half[2])
    corners = np.array([
        [ hx,  hy,  hz], [ hx, -hy,  hz], [-hx, -hy,  hz], [-hx,  hy,  hz],
        [ hx,  hy, -hz], [ hx, -hy, -hz], [-hx, -hy, -hz], [-hx,  hy, -hz],
    ], dtype=np.float32)
    corners = corners @ rotz(float(heading)).T + center.reshape(1, 3)
    return np.concatenate([corners.min(axis=0), corners.max(axis=0)]).astype(np.float32)


def resolve_gt_path(scene_id, gt_dir):
    for name in [f"{scene_id}_bbox_scannet20.npy", f"{scene_id}_aligned_bbox.npy", f"{scene_id}.npy"]:
        p = os.path.join(gt_dir, name)
        if os.path.exists(p):
            return p
    return None


def load_gt_scene_20(scene_id, gt_dir, gt_label_mode):
    gt_path = resolve_gt_path(scene_id, gt_dir)
    if gt_path is None:
        raise FileNotFoundError(f"GT not found for {scene_id} in {gt_dir}")

    raw = np.load(gt_path)
    if raw.ndim != 2 or raw.shape[1] < 7:
        raise ValueError(f"Unexpected GT shape {raw.shape} for {gt_path}")

    out = {i: [] for i in range(20)}

    for row in raw:
        if raw.shape[1] >= 8:
            center = row[:3].astype(np.float32)
            half   = row[3:6].astype(np.float32)
            heading = float(row[6])
            raw_cls = int(row[7])
            if gt_label_mode == "ov3det":
                if not (0 <= raw_cls < 20):
                    continue
                cls = raw_cls
            elif gt_label_mode == "nyu40":
                cls = NYU40_TO_OV3DET.get(raw_cls)
                if cls is None:
                    continue
            else:
                cls = raw_cls if 0 <= raw_cls < 20 else NYU40_TO_OV3DET.get(raw_cls)
                if cls is None:
                    continue
            box = obb_to_aabb(center, half, heading)
        else:
            center  = row[:3].astype(np.float32)
            size    = row[3:6].astype(np.float32)
            raw_cls = int(row[6])
            if gt_label_mode == "ov3det":
                if not (0 <= raw_cls < 20):
                    continue
                cls = raw_cls
            elif gt_label_mode == "nyu40":
                cls = NYU40_TO_OV3DET.get(raw_cls)
                if cls is None:
                    continue
            else:
                cls = raw_cls if 0 <= raw_cls < 20 else NYU40_TO_OV3DET.get(raw_cls)
                if cls is None:
                    continue
            box = np.concatenate([center - size / 2.0, center + size / 2.0]).astype(np.float32)

        if np.all(box[3:] > box[:3]):
            out[cls].append(box)

    return {cls: (np.stack(out[cls]) if out[cls] else np.zeros((0, 6), dtype=np.float32))
            for cls in range(20)}


def load_split_list(split_txt):
    with open(split_txt) as f:
        return [s.strip() for s in f.read().split() if s.strip()]


def discover_pred_map(pred_dir):
    out = {}
    for f in sorted(os.listdir(pred_dir)):
        if not f.endswith(".json"):
            continue
        m = re.search(r"(scene\d+_\d+)", f)
        if m:
            out[m.group(1)] = os.path.join(pred_dir, f)
    return out


def evaluate_scannet20(pred_dir, gt_dir, split_txt, gt_label_mode, per_class=False, verbose=False):
    scenes = load_split_list(split_txt)
    pred_map = discover_pred_map(pred_dir)

    gt_by_cls = [dict() for _ in range(20)]
    pred_by_cls = [[] for _ in range(20)]

    n_missing_gt = 0
    n_missing_pred = 0
    n_used = 0

    for scene_id in scenes:
        if resolve_gt_path(scene_id, gt_dir) is None:
            n_missing_gt += 1
            if verbose:
                print(f"[WARN] Missing GT: {scene_id}")
            continue

        gt_scene = load_gt_scene_20(scene_id, gt_dir, gt_label_mode)
        for cls in range(20):
            gt_by_cls[cls][scene_id] = gt_scene[cls]

        pred_path = pred_map.get(scene_id)
        if pred_path is None:
            n_missing_pred += 1
        else:
            with open(pred_path) as f:
                data = json.load(f)
            for obj in data:
                cls = pred_to_cls_20(obj)
                box = parse_bbox(obj)
                if cls is None or box is None:
                    continue
                pred_by_cls[cls].append((pick_score(obj), scene_id, box))

        n_used += 1

    if n_used == 0:
        raise RuntimeError("No scenes evaluated. Check --split_txt, --gt_dir, and filenames.")

    if verbose:
        print(f"Scenes in split: {len(scenes)}")
        print(f"Scenes used (GT found): {n_used}")
        print(f"Missing GT: {n_missing_gt}  |  Missing pred (treated as empty): {n_missing_pred}")

    ap_table = {thr: [] for thr in IOU_THRESHOLDS}
    for thr in IOU_THRESHOLDS:
        for cls in range(20):
            ap_table[thr].append(compute_ap_for_class(pred_by_cls[cls], gt_by_cls[cls], thr))

    mAP25 = np.mean(ap_table[0.25]) * 100.0
    mAP50 = np.mean(ap_table[0.50]) * 100.0

    print(f"\nVal scenes: {n_used}/{len(scenes)}")
    print(f"mAP@0.25 = {mAP25:.2f}")
    print(f"mAP@0.50 = {mAP50:.2f}")

    if per_class:
        print("\nPer-class AP:")
        for i, name in enumerate(OV3DET_CLASSES):
            print(f"  {i:2d} {name:14s}  AP25={ap_table[0.25][i] * 100:.2f}  AP50={ap_table[0.50][i] * 100:.2f}")

def _find_ply_path(scene_id, scan_root):
    candidates = [
        os.path.join(scan_root, f"{scene_id}.ply"),
        os.path.join(scan_root, scene_id, f"{scene_id}.ply"),
        os.path.join(scan_root, scene_id, f"{scene_id}_vh_clean_2.ply"),
        os.path.join(scan_root, scene_id, f"{scene_id}_vh_clean_2.0.010000.ply"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]


def pred_to_cls_200(obj):
    if "class_id" in obj:
        try:
            cid = int(obj["class_id"])
            if cid in _VALID_CLASS_IDS_200_SET:
                return cid
        except Exception:
            pass
    if "name" not in obj:
        return None
    name = normalize_name(str(obj["name"]))
    return LABEL_TO_ID_200.get(name)


def load_gt_scene_200(scene_id, scan_root, gt_dir):
    ply_path = _find_ply_path(scene_id, scan_root)
    agg_path = os.path.join(gt_dir, scene_id, f"{scene_id}.aggregation.json")
    seg_path = os.path.join(gt_dir, scene_id, f"{scene_id}_vh_clean_2.0.010000.segs.json")

    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"PLY not found: {ply_path}")
    if not os.path.exists(agg_path):
        raise FileNotFoundError(f"Aggregation JSON not found: {agg_path}")
    if not os.path.exists(seg_path):
        raise FileNotFoundError(f"Seg JSON not found: {seg_path}")

    plydata = PlyData.read(ply_path)
    vertices = np.vstack([
        plydata['vertex']['x'],
        plydata['vertex']['y'],
        plydata['vertex']['z'],
    ]).T.astype(np.float32)

    with open(seg_path) as f:
        seg_indices = np.array(json.load(f)['segIndices'])
    with open(agg_path) as f:
        agg_data = json.load(f)

    raw = {}
    for obj in agg_data['segGroups']:
        label = obj['label']
        if label not in LABEL_TO_ID_200:
            continue
        class_id = LABEL_TO_ID_200[label]
        pts = vertices[np.isin(seg_indices, obj['segments'])]
        if len(pts) == 0:
            continue
        box = np.concatenate([pts.min(axis=0), pts.max(axis=0)])
        if np.all(box[3:] > box[:3]):
            raw.setdefault(class_id, []).append(box)

    result = {}
    for cid in VALID_CLASS_IDS_200:
        result[cid] = np.stack(raw[cid]).astype(np.float32) if cid in raw else np.zeros((0, 6), dtype=np.float32)
    return result


def evaluate_scannet200(pred_dir, scan_root, gt_dir, split_txt=None, per_class=False, verbose=False):
    pred_map = discover_pred_map(pred_dir)

    if split_txt is not None:
        scenes = load_split_list(split_txt)
        pred_pairs = [(s, pred_map.get(s)) for s in scenes]
    else:
        pred_pairs = [(s, p) for s, p in pred_map.items()]
        pred_pairs.sort()

    gt_by_cls = {cid: {} for cid in VALID_CLASS_IDS_200}
    pred_by_cls = {cid: [] for cid in VALID_CLASS_IDS_200}

    n_missing_gt = 0
    n_missing_pred = 0
    n_used = 0
    for scene_id, pred_path in tqdm(pred_pairs, desc="Scenes"):
        try:
            gt_scene = load_gt_scene_200(scene_id, scan_root, gt_dir)
        except Exception as e:
            n_missing_gt += 1
            if verbose:
                print(f"[WARN] GT load failed for {scene_id}: {e}")
            continue

        for cid in VALID_CLASS_IDS_200:
            gt_by_cls[cid][scene_id] = gt_scene[cid]

        if pred_path is None:
            n_missing_pred += 1
            n_used += 1
            continue

        try:
            with open(pred_path) as fh:
                data = json.load(fh)
        except Exception as e:
            if verbose:
                print(f"[WARN] Pred load failed for {scene_id}: {e}")
            n_used += 1
            continue

        for obj in data:
            cid = pred_to_cls_200(obj)
            box = parse_bbox(obj)
            if cid is None or box is None:
                continue
            pred_by_cls[cid].append((pick_score(obj), scene_id, box.astype(np.float32)))

        n_used += 1

    if n_used == 0:
        raise RuntimeError("No scenes evaluated. Check --gt_dir and filenames.")

    if verbose:
        print(f"Scenes in split: {len(pred_pairs)}")
        print(f"Scenes used (GT found): {n_used}")
        print(f"Missing GT: {n_missing_gt}  |  Missing pred (treated as empty): {n_missing_pred}")

    ap_table = {thr: [] for thr in IOU_THRESHOLDS}
    for thr in IOU_THRESHOLDS:
        for cid in VALID_CLASS_IDS_200:
            ap_table[thr].append(compute_ap_for_class(pred_by_cls[cid], gt_by_cls[cid], thr))

    mAP25 = np.mean(ap_table[0.25]) * 100.0
    mAP50 = np.mean(ap_table[0.50]) * 100.0

    print(f"\nVal scenes: {n_used}/{len(pred_pairs)}")
    print(f"mAP@0.25 = {mAP25:.2f}")
    print(f"mAP@0.50 = {mAP50:.2f}")

    if per_class:
        print("\nPer-class AP:")
        for i, cid in enumerate(VALID_CLASS_IDS_200):
            label = ID_TO_LABEL_200[cid]
            print(f"  {cid:4d} {label:28s}  AP25={ap_table[0.25][i] * 100:.2f}  AP50={ap_table[0.50][i] * 100:.2f}")


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate Group3D outputs on ScanNet20 or ScanNet200")
    parser.add_argument("--benchmark", type=str, default="scannet20", choices=["scannet20", "scannet200"])
    parser.add_argument("--pred_dir", type=str, required=True,
                        help="Directory of prediction JSON files")
    parser.add_argument("--gt_dir", type=str, required=True,
                        help="[scannet20] Directory containing .npy GT files. "
                             "[scannet200] ScanNet scans directory containing {scene_id}/ subdirs "
                             "with .aggregation.json and .segs.json")
    parser.add_argument("--per_class", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--split_txt", type=str, default=None,
                        help="Val split scene list. Required for scannet20. "
                             "Optional for scannet200 (if omitted, scenes are inferred from pred_dir)")
    parser.add_argument("--gt_label_mode", type=str, default="ov3det",
                        choices=["ov3det", "nyu40", "auto"],
                        help="[scannet20] GT class ID format")
    parser.add_argument("--scan_root", type=str, default=None,
                        help="[scannet200] Directory containing scene PLY files. "
                             "If omitted, PLY files are looked up under --gt_dir")
    args = parser.parse_args()

    if args.benchmark == "scannet20":
        if not args.split_txt:
            parser.error("--split_txt is required for scannet20")
        evaluate_scannet20(args.pred_dir, args.gt_dir, args.split_txt,
                           args.gt_label_mode, args.per_class, args.verbose)

    elif args.benchmark == "scannet200":
        scan_root = args.scan_root if args.scan_root else args.gt_dir
        evaluate_scannet200(args.pred_dir, scan_root, args.gt_dir, args.split_txt,
                            args.per_class, args.verbose)


if __name__ == "__main__":
    main()
