# GT Preparation

## ScanNet20

Download ScanNetv2 from [https://github.com/scannet/scannet](https://github.com/scannet/scannet). Each scene directory should contain:

```
data/ScanNetv2/scans/
└── scene0011_00/
    ├── color/
    ├── depth/
    ├── intrinsic/
    ├── pose/
    ├── scene0011_00.txt
    ├── scene0011_00_vh_clean_2.ply
    ├── scene0011_00.aggregation.json
    └── scene0011_00_vh_clean_2.0.010000.segs.json
```

Then generate GT bounding boxes:

```bash
python data/ScanNetv2/make_scannet_20.py \
    --scans_root data/ScanNetv2/scans \
    --split_txt data/ScanNetv2/scannet_val.txt \
    --label_map_tsv data/ScanNetv2/scannetv2-labels.combined.tsv \
    --out_dir data/ScanNetv2/scannet_20
```

This produces `{scene_id}_bbox_scannet20.npy` and `{scene_id}_bbox_nyu40.npy` under `data/ScanNetv2/scannet_20/`.

## ScanNet200

Download the ScanNet200 dataset following [LanguageGroundedSemseg](https://github.com/RozDavid/LanguageGroundedSemseg) and place the validation set PLY files as follows:

```
data/ScanNetv2/scannet_200/val/
├── scene0011_00.ply
├── scene0015_00.ply
└── ...
```

Pass `data/ScanNetv2/scannet_200/val` as `--scan_root` when running evaluation.
