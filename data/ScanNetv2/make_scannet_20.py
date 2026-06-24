import os
import json
import csv
import numpy as np
from plyfile import PlyData
from tqdm import tqdm

NYU40IDS_SCANNET20 = np.array([33,4,5,6,17,7,3,10,18,34,36,24,14,32,12,8,16,29,35,37], dtype=int)
NYU40ID2CLASS20 = {int(nyu): i for i, nyu in enumerate(NYU40IDS_SCANNET20.tolist())}

def read_mesh_vertices(mesh_file: str) -> np.ndarray:
    with open(mesh_file, "rb") as f:
        ply = PlyData.read(f)
    v = ply["vertex"].data
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    return xyz

def read_axis_align_matrix(meta_file: str) -> np.ndarray:
    # default: identity
    axis_align = np.eye(4, dtype=np.float32)
    with open(meta_file, "r") as f:
        for line in f:
            if "axisAlignment" in line:
                # e.g. "axisAlignment = 0.999 ... (16 floats)"
                parts = line.strip().split("=")
                if len(parts) < 2:
                    continue
                vals = [float(x) for x in parts[1].strip().split()]
                if len(vals) == 16:
                    axis_align = np.array(vals, dtype=np.float32).reshape(4, 4)
                break
    return axis_align

def apply_axis_align(xyz: np.ndarray, axis_align: np.ndarray) -> np.ndarray:
    pts = np.ones((xyz.shape[0], 4), dtype=np.float32)
    pts[:, :3] = xyz
    pts = pts @ axis_align.T
    return pts[:, :3]

def read_label_map_tsv(tsv_file: str) -> dict[str, int]:
    # raw_category -> nyu40id
    mapping = {}
    with open(tsv_file, "r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            raw = row["raw_category"].strip().lower()
            nyu = int(row["nyu40id"])
            mapping[raw] = nyu
    return mapping

def read_seg_to_verts(seg_file: str):
    with open(seg_file, "r") as f:
        data = json.load(f)
    seg_indices = data["segIndices"]  # len = num_verts
    seg_to_verts = {}
    for vidx, seg_id in enumerate(seg_indices):
        seg_to_verts.setdefault(seg_id, []).append(vidx)
    return seg_to_verts, len(seg_indices)

def read_instances(agg_file: str):
    with open(agg_file, "r") as f:
        data = json.load(f)
    # data["segGroups"]: each has objectId, label, segments
    return data["segGroups"]

def compute_aabb(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mn = xyz.min(axis=0)
    mx = xyz.max(axis=0)
    ctr = (mn + mx) * 0.5
    size = (mx - mn)
    return ctr, size

def build_scene_gt(scene_dir: str, label_map: dict[str, int], out_dir: str):
    scene_id = os.path.basename(scene_dir.rstrip("/"))
    mesh_file = os.path.join(scene_dir, f"{scene_id}_vh_clean_2.ply")
    agg_file  = os.path.join(scene_dir, f"{scene_id}.aggregation.json")
    seg_file  = os.path.join(scene_dir, f"{scene_id}_vh_clean_2.0.010000.segs.json")
    meta_file = os.path.join(scene_dir, f"{scene_id}.txt")

    for p in [mesh_file, agg_file, seg_file, meta_file]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"missing: {p}")

    xyz = read_mesh_vertices(mesh_file)
    seg_to_verts, num_verts = read_seg_to_verts(seg_file)
    if xyz.shape[0] != num_verts:
        raise RuntimeError(
            f"[{scene_id}] vertex count mismatch: mesh={xyz.shape[0]} vs segs={num_verts}. "
            f"wrong mesh/segs pairing?"
        )

    axis_align = read_axis_align_matrix(meta_file)
    xyz_aligned = apply_axis_align(xyz, axis_align)

    seg_groups = read_instances(agg_file)

    boxes_nyu40 = []   # (cx,cy,cz, dx,dy,dz, nyu40id)
    boxes_cls20 = []   # (cx,cy,cz, dx/2,dy/2,dz/2, heading=0, cls20)

    for g in seg_groups:
        raw_label = str(g["label"]).strip().lower()
        nyu40id = label_map.get(raw_label, 0)
        if nyu40id not in NYU40ID2CLASS20:
            continue

        segs = g["segments"]
        verts = []
        for seg_id in segs:
            vlist = seg_to_verts.get(seg_id, [])
            if vlist:
                verts.extend(vlist)
        if len(verts) == 0:
            continue

        obj_xyz = xyz_aligned[np.array(verts, dtype=np.int64)]
        ctr, size = compute_aabb(obj_xyz)

        boxes_nyu40.append([ctr[0], ctr[1], ctr[2], size[0], size[1], size[2], float(nyu40id)])

        cls20 = NYU40ID2CLASS20[nyu40id]
        half = size * 0.5
        boxes_cls20.append([ctr[0], ctr[1], ctr[2], half[0], half[1], half[2], 0.0, float(cls20)])

    os.makedirs(out_dir, exist_ok=True)
    boxes_nyu40 = np.array(boxes_nyu40, dtype=np.float32).reshape(-1, 7)
    boxes_cls20 = np.array(boxes_cls20, dtype=np.float32).reshape(-1, 8)

    np.save(os.path.join(out_dir, f"{scene_id}_bbox_nyu40.npy"), boxes_nyu40)
    np.save(os.path.join(out_dir, f"{scene_id}_bbox_scannet20.npy"), boxes_cls20)

    return boxes_cls20.shape[0]

def load_split_list(split_txt: str):
    with open(split_txt, "r") as f:
        content = f.read()
    scenes = [s.strip() for s in content.split() if s.strip()]
    return scenes

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scans_root", default="data/ScanNetv2/scans", help=".../scannet/scans")
    parser.add_argument("--split_txt", default="data/ScanNetv2/scannet_val.txt", help="scannetv2_val.txt")
    parser.add_argument("--label_map_tsv", default="data/ScanNetv2/scannetv2-labels.combined.tsv", help="scannetv2-labels.combined.tsv")
    parser.add_argument("--out_dir", default="data/ScanNetv2/scannet_20")
    args = parser.parse_args()

    scenes = load_split_list(args.split_txt)
    label_map = read_label_map_tsv(args.label_map_tsv)

    total = 0
    for i, scene_id in tqdm(enumerate(scenes)):
        scene_dir = os.path.join(args.scans_root, scene_id)
        if not os.path.isdir(scene_dir):
            print(f"[WARN] missing scene dir: {scene_dir}")
            continue
        try:
            n = build_scene_gt(scene_dir, label_map, args.out_dir)
            total += n
            if (i + 1) % 20 == 0:
                print(f"{i+1}/{len(scenes)} scenes processed, total boxes so far={total}")
        except Exception as e:
            print(f"[ERR] {scene_id}: {e}")

    print(f"[DONE] scenes={len(scenes)}, total_boxes={total}, out={args.out_dir}")

if __name__ == "__main__":
    main()
