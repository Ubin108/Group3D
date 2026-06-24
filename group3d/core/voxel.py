import numpy as np


def voxelize_points(points: np.ndarray, voxel_size: float) -> set:
    vox = np.floor(points / voxel_size).astype(np.int32)
    return set(map(tuple, vox))


def voxel_overlap(
    a: set,
    b: set,
    iou_thresh: float = 0.01,
    contain_thresh: float = 0.1,
) -> bool:

    if not a or not b:
        return False
    inter = len(a & b)
    if inter == 0:
        return False
    union = len(a | b)
    iou = inter / union
    return (
        iou >= iou_thresh
        or inter / len(a) >= contain_thresh
        or inter / len(b) >= contain_thresh
    )
