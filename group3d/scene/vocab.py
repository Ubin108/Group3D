import os
import json
import time
import asyncio
from pathlib import Path
from tqdm import tqdm
from ..config import Group3DConfig
from ..mllm.client import ask_gpt, ask_gpt_async
from ..mllm.prompts import OBJECT_PROPOSAL_Q
from ..utils.label_utils import parse_object_list


async def _process_frame_async(i: int, img_path: str, cfg: Group3DConfig,
                                semaphore: asyncio.Semaphore):
    async with semaphore:
        t0 = time.time()
        raw = await ask_gpt_async(
            image_paths=[img_path],
            question=OBJECT_PROPOSAL_Q,
            model=cfg.vlm.gpt_model,
        )
        t_elapsed = time.time() - t0
        objects = parse_object_list(raw, max_objects=cfg.num_category_hypotheses)
        return i, img_path, objects, t_elapsed


async def _build_scene_vocab_async(scene: str, cfg: Group3DConfig,
                                   max_concurrent: int):
    ds          = cfg.dataset
    image_dir   = ds.image_dir(scene)
    output_path = ds.label_path(scene)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    frame_paths = []
    for i in range(cfg.num_frames):
        img_path = os.path.join(image_dir, f"{i:05d}.jpg")
        if os.path.exists(img_path):
            frame_paths.append((i, img_path))

    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = [_process_frame_async(i, p, cfg, semaphore) for i, p in frame_paths]

    t_wall_start = time.time()
    results = await asyncio.gather(*tasks)
    t_wall = time.time() - t_wall_start

    results = sorted(results, key=lambda x: x[0])

    with open(output_path, "w", encoding="utf-8") as f_out:
        for i, img_path, objects, _ in results:
            record = {
                "frame_id": i,
                "image_path": img_path,
                "objects": objects,
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

    t_per_frame   = [r[3] for r in results]
    t_total_serial = sum(t_per_frame)
    t_mean         = t_total_serial / len(t_per_frame) if t_per_frame else 0.0

    print(
        f"[{scene}] Vocab saved ({len(results)} frames) → {output_path}\n"
        f"  per-frame avg: {t_mean:.2f}s"
    )


def build_scene_vocab(scene: str, cfg: Group3DConfig,
                      max_concurrent: int = 20):
    if os.path.exists(cfg.dataset.label_path(scene)):
        print(f"[{scene}] Vocab already exists, skipping.")
        return
    asyncio.run(_build_scene_vocab_async(scene, cfg, max_concurrent))


def build_all_vocabs(cfg: Group3DConfig, max_concurrent: int = 20):
    root = Path(cfg.dataset.root)
    scenes = sorted(d.name for d in root.iterdir() if d.is_dir())

    for scene in tqdm(scenes, desc="Building vocabs"):
        print("=" * 40)
        print(f"concurrent: {max_concurrent}")
        print(f"SCENE: {scene}")
        try:
            build_scene_vocab(scene, cfg, max_concurrent=max_concurrent)
        except Exception as e:
            print(f"[ERROR] {scene}: {e}")