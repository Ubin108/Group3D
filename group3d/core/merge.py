from collections import Counter
from typing import List, Tuple

import numpy as np

from .instance import merge_instances
from .voxel import voxel_overlap
from ..utils.label_utils import grouping_synonym, build_label_to_group, same_group


def merge_loop_voxel(
    instances_3d: dict,
    force_merge_labels: set,
    gpt_model: str,
    voxel_iou_thresh: float,
    voxel_contain_thresh: float,
) -> Tuple[List[dict], dict]:
    
    label_freq = Counter(inst["name"] for insts in instances_3d.values() for inst in insts)
    print("Label freq:", label_freq, flush=True)

    labels = list(label_freq.keys())
    groups = grouping_synonym(labels, gpt_model=gpt_model)
    label_group = build_label_to_group(groups)
    print("Synonym Groups:", groups, flush=True)

    all_instances = []
    for insts in instances_3d.values():
        all_instances.extend(insts)

    all_instances = sorted(all_instances, key=lambda x: np.prod(x["size"]), reverse=True)
    merged_instances = []

    for inst in all_instances:
        merged = False
        for i, base in enumerate(merged_instances):
            if base["name"] in force_merge_labels and inst["name"] in force_merge_labels:
                merged_instances[i] = merge_instances(base, inst)
                merged = True
                break

            if not same_group(base["name"], inst["name"], label_group):
                continue

            if voxel_overlap(base["voxels"], inst["voxels"],
                             iou_thresh=voxel_iou_thresh,
                             contain_thresh=voxel_contain_thresh):
                merged_instances[i] = merge_instances(base, inst)
                merged = True
                break

        if not merged:
            merged_instances.append(inst)

    return merged_instances, label_group


def post_merge_instances(
    instances: List[dict],
    label_to_group: dict,
    voxel_iou_thresh: float = 0.1,
    voxel_contain_thresh: float = 0.1,
) -> List[dict]:

    merged = True
    instances = instances.copy()

    while merged:
        merged = False
        new_instances = []
        used = [False] * len(instances)

        for i in range(len(instances)):
            if used[i]:
                continue
            base = instances[i]

            for j in range(i + 1, len(instances)):
                if used[j]:
                    continue
                other = instances[j]

                if not same_group(base["name"], other["name"], label_to_group):
                    continue
                if not voxel_overlap(base["voxels"], other["voxels"],
                                     iou_thresh=voxel_iou_thresh,
                                     contain_thresh=voxel_contain_thresh):
                    continue

                base = merge_instances(base, other)
                used[j] = True
                merged = True

            used[i] = True
            new_instances.append(base)

        instances = new_instances

    return instances
