<div align="center">

<h1>Group3D</h1>

<p><b>MLLM-Driven Semantic Grouping for Open-Vocabulary 3D Object Detection</b></p>

<p>
<a href="https://github.com/Ubin108">Youbin Kim</a><sup>1</sup> &nbsp;·&nbsp;
<a href="https://github.com/zinosii">Jinho Park</a><sup>1</sup> &nbsp;·&nbsp;
<a href="https://hogunpark.com/">Hogun Park</a><sup>1</sup> &nbsp;·&nbsp;
<a href="https://silverbottlep.github.io">Eunbyung Park</a><sup>2</sup>
</p>

<p>
<sup>1</sup> Sungkyunkwan University &nbsp;&nbsp; <sup>2</sup> Yonsei University
</p>

<p><font size="4"><b>ECCV 2026</b></font></p>

<p>
<a href="https://arxiv.org/abs/2603.21944"><img src="https://img.shields.io/badge/arXiv-2603.21944-b31b1b.svg?style=flat-square" alt="arXiv"/></a>
<a href="https://ubin108.github.io/Group3D/"><img src="https://img.shields.io/badge/Project-Page-DAA520?style=flat-square" alt="Project Page"/></a>
</p>

https://github.com/user-attachments/assets/05a9bed7-af35-4ae8-a9f1-29c4a1c974a6

</div>

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Ubin108/Group3D.git --recursive
cd Group3D
```

### 2. Install dependencies

```bash
conda create -n group3d python=3.12
conda activate group3d
pip install torch==2.7.0+cu118 torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
cd third_party/Depth-Anything-3
pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70
pip install -e ".[all]" --no-build-isolation
cd ../..
pip install -e third_party/sam3
```

### 3. HuggingFace login (for SAM3 weights)

SAM3 model weights are gated on HuggingFace. Visit the [SAM3 model page](https://huggingface.co/facebook/sam3), agree to share your contact information with Meta, then log in:

```bash
huggingface-cli login
```

### 4. Set up API keys

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
```

If using a self-hosted Qwen model via vLLM, set `qwen_base_url` in the config instead.

## Data Preparation

### 1. Download ScanNetv2

Download [ScanNetv2](http://www.scan-net.org/) and place it as follows:

```
data/ScanNetv2/scans/
└── scene0011_00/
    ├── color/
    ├── depth/
    ├── intrinsic/
    └── pose/
```

### 2. Frame sampling

Sample 128 frames uniformly from each scene. This creates `video_color_128/` under each scene directory.

```bash
python preprocess/sample_frames.py \
    --root data/ScanNetv2/scans \
    --target_n 128
```

### 3. Depth estimation & point cloud building

Run Depth-Anything-3 on the sampled frames. This creates `point_cloud/` with depth maps, confidence maps, and estimated camera extrinsics.

```bash
python preprocess/build_pointcloud_da3.py \
    --root data/ScanNetv2/scans \
    --target_n 128
```

### 4. Point cloud alignment

Align the point cloud to the scene coordinate frame. This creates `point_cloud_aligned_pose_free/` (or `point_cloud_aligned_pose_known/`).

```bash
# Pose-free (uses DA3 estimated poses)
python preprocess/align_pointcloud.py \
    --dataset_root data/ScanNetv2/scans \
    --pose_mode pose_free

# Pose-known (uses ground-truth poses)
python preprocess/align_pointcloud.py \
    --dataset_root data/ScanNetv2/scans \
    --pose_mode pose_known
```

### Final dataset structure

```
data/ScanNetv2/scans/
└── scene0011_00/
    ├── color/
    ├── depth/
    ├── intrinsic/
    ├── pose/
    ├── video_color_128/
    ├── point_cloud/
    ├── point_cloud_aligned_pose_free/
    └── point_cloud_aligned_pose_known/
```

## Running Group3D

### Run the full pipeline

```bash
# All scenes
python run.py --config configs/pose_free.yaml

# Single scene
python run.py --config configs/pose_free.yaml --scene scene0011_00
```

## Evaluation

### ScanNet20

Before evaluating, generate GT bounding boxes — see [eval/README.md](eval/README.md) for required files and instructions.

```bash
python eval/evaluation.py \
    --pred_dir results \
    --gt_dir data/ScanNetv2/scannet_20 \
    --split_txt data/ScanNetv2/scannet_val.txt \
    [--gt_label_mode ov3det] [--per_class] [--verbose]
```

- `--gt_dir`: directory containing GT `.npy` files (`{scene_id}_bbox_scannet20.npy` or `{scene_id}_aligned_bbox.npy`)
- `--split_txt`: val split scene list (312 scenes)
- `--gt_label_mode`: `ov3det` (default, class IDs 0–19) or `nyu40`

### ScanNet200

```bash
python eval/evaluation.py --benchmark scannet200 \
    --pred_dir results \
    --scan_root /path/to/scans_200/val \
    --gt_dir /path/to/scannet/scans \
    --split_txt data/ScanNetv2/scannet_val.txt \
    [--per_class] [--verbose]
```

- `--scan_root`: directory containing scene PLY files (e.g. `scans_200/val/`)
- `--gt_dir`: ScanNet scans directory containing `{scene_id}/` subdirs with `.aggregation.json` and `.segs.json`
- `--split_txt`: val split scene list (optional; if omitted, scenes are inferred from `--pred_dir`)
- If `--scan_root` is omitted, PLY files are looked up under `--gt_dir`

## Visualization

Render detected instances as 3D bounding boxes:

```bash
python visualization.py --scene scene0011_00 --config configs/pose_free.yaml
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{kim2026group3d,
  title     = {Group3D: MLLM-Driven Semantic Grouping for Open-Vocabulary 3D Object Detection},
  author    = {Kim, Youbin and Park, Jinho and Park, Hogun and Park, Eunbyung},
  journal   = {arXiv preprint arXiv:2603.21944},
  year      = {2026}
}
```

## Acknowledgements

We thank the authors of [SAM3](https://github.com/facebookresearch/sam3), [Depth-Anything-3](https://github.com/DepthAnything/Depth-Anything-V3) for their excellent work and open-source contributions.
