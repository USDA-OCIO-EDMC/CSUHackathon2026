"""Download summer HLS scenes for our 5-state corn belt.

Strategy: 3 narrow time windows × multiple years to unlock real temporal data
for aug_01 and sep_01 checkpoints. Targets ~100 GB total.

Existing data: May 2022-2025 (DOY 121-130)
Adding:        July, August, September scenes for 2022-2024

Output: appends to data/raw/hls/scenes/ (same folder as existing)
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# 5-state corn belt bounding box
BBOX = "-104.5,36.0,-86.8,47.0"

# Time windows + years to download. Each tuple: (year, start, end, label)
# 10-day windows in peak imaging conditions
DOWNLOADS = [
    # 2024 - critical (holdout year, most chips) - priority
    (2024, "07-15", "07-25", "Jul-2024"),
    (2024, "08-05", "08-15", "Aug-2024"),
    (2024, "09-01", "09-10", "Sep-2024"),
    # 2023 - val year, for hindcast bias correction quality
    (2023, "07-15", "07-25", "Jul-2023"),
    (2023, "08-05", "08-15", "Aug-2023"),
    (2023, "09-01", "09-10", "Sep-2023"),
    # 2022 - earliest training year
    (2022, "07-15", "07-25", "Jul-2022"),
    (2022, "08-05", "08-15", "Aug-2022"),
    (2022, "09-01", "09-10", "Sep-2022"),
]

# Limit per window — calibrated to fit 100 GB budget
# 9 windows × 80 scenes × ~140 MB avg = ~100 GB
MAX_SCENES_PER_WINDOW = 80
MAX_CLOUD = 20


def main() -> None:
    fetch_script = ROOT / "scripts" / "data" / "fetch_hls_earthaccess.py"
    if not fetch_script.exists():
        sys.exit(f"missing {fetch_script}")

    print(f"Downloading {len(DOWNLOADS)} summer HLS windows")
    print(f"BBOX: {BBOX}  cloud<{MAX_CLOUD}%  max {MAX_SCENES_PER_WINDOW} scenes/window")
    print(f"Estimated total: ~{len(DOWNLOADS) * MAX_SCENES_PER_WINDOW * 0.14:.0f} GB\n")

    for i, (year, start_md, end_md, label) in enumerate(DOWNLOADS):
        env = os.environ.copy()
        env["HLS_BBOX"]       = BBOX
        env["HLS_DATE_START"] = f"{year}-{start_md}"
        env["HLS_DATE_END"]   = f"{year}-{end_md}"
        env["HLS_MAX_SCENES"] = str(MAX_SCENES_PER_WINDOW)
        env["HLS_MAX_CLOUD"]  = str(MAX_CLOUD)
        env["HLS_PRODUCT"]    = "both"

        print(f"\n=== [{i+1}/{len(DOWNLOADS)}] {label} "
              f"({env['HLS_DATE_START']} → {env['HLS_DATE_END']}) ===", flush=True)
        result = subprocess.run(
            [sys.executable, str(fetch_script)],
            env=env, cwd=str(ROOT)
        )
        if result.returncode != 0:
            print(f"  ⚠ {label} failed (return code {result.returncode}), continuing...")

    print("\n✓ All download windows attempted")
    # Show how much we actually downloaded
    scene_dir = ROOT / "data" / "raw" / "hls" / "scenes"
    n_files   = sum(1 for _ in scene_dir.iterdir())
    total_gb  = sum(f.stat().st_size for f in scene_dir.iterdir()) / 1e9
    print(f"Total scenes folder: {n_files:,} files, {total_gb:.1f} GB")


if __name__ == "__main__":
    main()
