import open3d as o3d
import numpy as np
import json
import colorsys
import os
import argparse
import yaml

try:
    import pygltflib
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pygltflib", "-q"])
    import pygltflib


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    return parser.parse_args()


def generate_distinct_colors(n):
    result = []
    for i in range(n):
        h = i / n
        s = 0.75
        v = 0.95
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        result.append((int(r * 255), int(g * 255), int(b * 255)))
    return result


all_binary = bytearray()
accessors = []
buffer_views = []
meshes = []
nodes = []
materials = []


def add_buffer_view(data: bytes, target=None):
    offset = len(all_binary)
    all_binary.extend(data)
    while len(all_binary) % 4 != 0:
        all_binary.extend(b'\x00')
    bv = pygltflib.BufferView(
        buffer=0,
        byteOffset=offset,
        byteLength=len(data),
        target=target
    )
    buffer_views.append(bv)
    return len(buffer_views) - 1


def add_accessor_f32(data: np.ndarray, type_str: str, target=None):
    flat = data.flatten().astype(np.float32)
    raw = flat.tobytes()
    bv_idx = add_buffer_view(raw, target)
    acc = pygltflib.Accessor(
        bufferView=bv_idx,
        byteOffset=0,
        componentType=pygltflib.FLOAT,
        count=len(data),
        type=type_str,
        min=[float(data[:, i].min()) for i in range(data.shape[1])] if data.ndim > 1 else [float(data.min())],
        max=[float(data[:, i].max()) for i in range(data.shape[1])] if data.ndim > 1 else [float(data.max())],
    )
    accessors.append(acc)
    return len(accessors) - 1


def add_accessor_u8_rgb(data: np.ndarray):
    raw = data.astype(np.uint8).tobytes()
    bv_idx = add_buffer_view(raw)
    acc = pygltflib.Accessor(
        bufferView=bv_idx,
        byteOffset=0,
        componentType=pygltflib.UNSIGNED_BYTE,
        count=len(data),
        type="VEC3",
        normalized=True,
    )
    accessors.append(acc)
    return len(accessors) - 1


def add_accessor_u32(data: np.ndarray, type_str="SCALAR"):
    flat = data.flatten().astype(np.uint32)
    raw = flat.tobytes()
    bv_idx = add_buffer_view(raw, target=pygltflib.ELEMENT_ARRAY_BUFFER)
    acc = pygltflib.Accessor(
        bufferView=bv_idx,
        byteOffset=0,
        componentType=pygltflib.UNSIGNED_INT,
        count=len(flat),
        type=type_str,
    )
    accessors.append(acc)
    return len(accessors) - 1


def main():
    args = parse_args()
    scene = args.scene

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_root = cfg['dataset']['root']
    output_dir = cfg['output']['dir']
    json_suffix = cfg['output']['json_suffix']
    glb_suffix = json_suffix.replace('.json', '.glb')

    ply_path = os.path.join(dataset_root, scene, f'{scene}_vh_clean_2_aligned.ply')
    json_path = os.path.join(script_dir, output_dir, f'scene_instances_{scene}{json_suffix}')
    out_path = os.path.join(script_dir, output_dir, f'{scene}{glb_suffix}')

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points, dtype=np.float32)

    if not pcd.has_colors():
        raise ValueError("No color information found in PLY file.")

    colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)

    sample = 1
    points_ds = points[::sample]
    colors_ds = colors[::sample]

    with open(json_path, "r") as f:
        bbox_data = json.load(f)

    unique_classes = sorted(set(obj["name"] for obj in bbox_data))
    palette = generate_distinct_colors(len(unique_classes))
    class_color_map = {cls: palette[i] for i, cls in enumerate(unique_classes)}

    lines = [
        [0,1],[1,7],[7,2],[2,0],
        [3,6],[6,4],[4,5],[5,3],
        [0,3],[1,6],[7,4],[2,5]
    ]

    gltf = pygltflib.GLTF2()
    gltf.scene = 0
    gltf.scenes = [pygltflib.Scene(nodes=[])]

    print(f"Processing point cloud: {len(points_ds):,} points...")

    pos_acc = add_accessor_f32(points_ds, "VEC3", pygltflib.ARRAY_BUFFER)
    col_acc = add_accessor_u8_rgb(colors_ds)

    mat_pc = pygltflib.Material(
        name="PointCloudMaterial",
        pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
            metallicFactor=0.0,
            roughnessFactor=1.0,
        ),
        doubleSided=True,
    )
    materials.append(mat_pc)
    mat_pc_idx = len(materials) - 1

    prim = pygltflib.Primitive(
        attributes=pygltflib.Attributes(POSITION=pos_acc, COLOR_0=col_acc),
        material=mat_pc_idx,
        mode=0,
    )
    mesh = pygltflib.Mesh(name="PointCloud", primitives=[prim])
    meshes.append(mesh)
    mesh_idx = len(meshes) - 1

    node = pygltflib.Node(name="PointCloud", mesh=mesh_idx)
    nodes.append(node)
    gltf.scenes[0].nodes.append(len(nodes) - 1)

    print("Processing bounding boxes...")

    bbox_count = 0
    for obj in bbox_data:
        name = obj["name"]

        if name in ("wall", "floor"):
            continue

        bbox_min = obj["bbox"]["min"]
        bbox_max = obj["bbox"]["max"]

        aabb = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=bbox_min,
            max_bound=bbox_max
        )
        box_pts = np.asarray(aabb.get_box_points(), dtype=np.float32)

        r, g, b = class_color_map[name]

        seg_pts = []
        for edge in lines:
            seg_pts.append(box_pts[edge[0]])
            seg_pts.append(box_pts[edge[1]])
        seg_pts = np.array(seg_pts, dtype=np.float32)

        seg_cols = np.tile(np.array([[r, g, b]], dtype=np.uint8), (len(seg_pts), 1))
        indices = np.arange(len(seg_pts), dtype=np.uint32)

        pos_acc_b = add_accessor_f32(seg_pts, "VEC3", pygltflib.ARRAY_BUFFER)
        col_acc_b = add_accessor_u8_rgb(seg_cols)
        idx_acc_b = add_accessor_u32(indices)

        r_f, g_f, b_f = r / 255.0, g / 255.0, b / 255.0
        mat = pygltflib.Material(
            name=f"BBoxMat_{name}_{bbox_count}",
            pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
                baseColorFactor=[r_f, g_f, b_f, 1.0],
                metallicFactor=0.0,
                roughnessFactor=1.0,
            ),
            emissiveFactor=[r_f * 0.8, g_f * 0.8, b_f * 0.8],
            doubleSided=True,
        )
        materials.append(mat)
        mat_idx = len(materials) - 1

        prim_b = pygltflib.Primitive(
            attributes=pygltflib.Attributes(POSITION=pos_acc_b, COLOR_0=col_acc_b),
            indices=idx_acc_b,
            material=mat_idx,
            mode=1,
        )
        mesh_b = pygltflib.Mesh(name=f"BBox_{name}_{bbox_count}", primitives=[prim_b])
        meshes.append(mesh_b)
        mesh_b_idx = len(meshes) - 1

        node_b = pygltflib.Node(name=f"BBox_{name}_{bbox_count}", mesh=mesh_b_idx)
        nodes.append(node_b)
        gltf.scenes[0].nodes.append(len(nodes) - 1)
        bbox_count += 1

    print(f"{bbox_count} bounding boxes added.")

    binary_data = bytes(all_binary)

    buf = pygltflib.Buffer(byteLength=len(binary_data))
    gltf.buffers = [buf]
    gltf.bufferViews = buffer_views
    gltf.accessors = accessors
    gltf.meshes = meshes
    gltf.nodes = nodes
    gltf.materials = materials

    gltf.set_binary_blob(binary_data)
    gltf.save(out_path)
    print(f"Saved to: {out_path}")
    print(f"File size: {os.path.getsize(out_path) / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
