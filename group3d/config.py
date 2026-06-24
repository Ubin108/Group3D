from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal
import yaml


_PKG_ROOT = Path(__file__).resolve().parent.parent

SAM3_ROOT = str(_PKG_ROOT / "third_party" / "sam3")

_POINTS_SUBDIR = {
    "pose_free":  "point_cloud_aligned_pose_free",
    "pose_known": "point_cloud_aligned_pose_known",
}


@dataclass
class VLMConfig:
    gpt_model: str = "gpt-5.1"
    qwen_model: str = "Qwen/Qwen3-VL-8B-Instruct"
    qwen_base_url: str = "http://localhost:8000/v1"


@dataclass
class MergeConfig:
    voxel_iou_thresh: float = 0.01
    voxel_contain_thresh: float = 0.1
    post_voxel_iou_thresh: float = 0.1
    post_voxel_contain_thresh: float = 0.1
    force_merge_labels: List[str] = field(default_factory=lambda: ["floor"])


@dataclass
class DatasetConfig:
    root: str = "/hdd/ubin108/ECCV/Group3D_release/data/ScanNetv2/scans"
    name: str = "scannet"
    image_subdir: str = "video_color_128"
    points_subdir: str = _POINTS_SUBDIR["pose_free"]
    label_filename: str = "objects_128.jsonl"
    mask_root_pattern: str = (
        "/hdd/ubin108/ECCV/Group3D_release/data/ScanNetv2/scans/{scene}/sam3_masks"
    )

    def scene_dir(self, scene: str) -> str:
        return str(Path(self.root) / scene)

    def image_dir(self, scene: str) -> str:
        return str(Path(self.scene_dir(scene)) / self.image_subdir)

    def points_dir(self, scene: str) -> str:
        return str(Path(self.scene_dir(scene)) / self.points_subdir)

    def label_path(self, scene: str) -> str:
        return str(Path(self.scene_dir(scene)) / self.label_filename)

    def mask_root(self, scene: str) -> str:
        return self.mask_root_pattern.format(dataset=self.name, scene=scene)


@dataclass
class OutputConfig:
    dir: str = "results"
    json_suffix: str = "_128.json"


@dataclass
class Group3DConfig:
    pose_mode: Literal["pose_free", "pose_known"] = "pose_free"

    voxel_size: float = 0.05
    conf_percentile: int = 15
    num_frames: int = 128
    sam_score_thresh: float = 0.5
    depth_thresh: float = 0.05
    num_category_hypotheses: int = 5

    vlm: VLMConfig = field(default_factory=VLMConfig)
    merge: MergeConfig = field(default_factory=MergeConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def load_config(path: str) -> Group3DConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    pose_mode = raw.get("pose_mode", "pose_free")
    if pose_mode not in _POINTS_SUBDIR:
        raise ValueError(
            f"Invalid pose_mode '{pose_mode}'. Choose from: {list(_POINTS_SUBDIR)}"
        )

    cfg = Group3DConfig(
        pose_mode=pose_mode,
        voxel_size=raw.get("voxel_size", 0.05),
        conf_percentile=raw.get("conf_percentile", 15),
        num_frames=raw.get("num_frames", 128),
        sam_score_thresh=raw.get("sam_score_thresh", 0.5),
        depth_thresh=raw.get("depth_thresh", 0.05),
        num_category_hypotheses=raw.get("num_category_hypotheses", 5),
    )

    if "vlm" in raw:
        cfg.vlm = VLMConfig(**raw["vlm"])
    if "merge" in raw:
        cfg.merge = MergeConfig(**raw["merge"])
    if "dataset" in raw:
        cfg.dataset = DatasetConfig(**raw["dataset"])
    if "output" in raw:
        cfg.output = OutputConfig(**raw["output"])

    if "dataset" not in raw or "points_subdir" not in raw.get("dataset", {}):
        cfg.dataset.points_subdir = _POINTS_SUBDIR[pose_mode]

    return cfg
