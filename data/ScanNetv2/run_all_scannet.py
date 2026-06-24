import os
import subprocess
from tqdm import tqdm

SCANS_ROOT = "/PATH/TO/Group3D/data/ScanNetv2/scans"
READER = "reader.py"

for scene in tqdm(sorted(os.listdir(SCANS_ROOT))):
    
    scene_dir = os.path.join(SCANS_ROOT, scene)

    if not os.path.isdir(scene_dir):
        continue

    sens_file = os.path.join(scene_dir, f"{scene}.sens")

    if not os.path.exists(sens_file):
        print(f"[SKIP] {scene}: .sens not found")
        continue

    print(f"[PROCESS] {scene}")

    cmd = [
        "python", READER,
        "--filename", sens_file,
        "--output_path", scene_dir,
        "--export_color_images",
        "--export_depth_images",
        "--export_poses",
        "--export_intrinsics"
    ]

    subprocess.run(cmd, check=True)
    
print("All scenes processed")
