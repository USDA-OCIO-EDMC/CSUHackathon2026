"""Download June HLS scenes for 2022-2024 to fill the missing month between
existing May data and the summer download (Jul/Aug/Sep).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

BBOX = "-104.5,36.0,-86.8,47.0"
DOWNLOADS = [
    (2024, "06-10", "06-20", "Jun-2024"),
    (2023, "06-10", "06-20", "Jun-2023"),
    (2022, "06-10", "06-20", "Jun-2022"),
]
MAX_SCENES_PER_WINDOW = 80
MAX_CLOUD = 20


def main() -> None:
    fetch_script = ROOT / "scripts" / "data" / "fetch_hls_earthaccess.py"
    print(f"June download: {len(DOWNLOADS)} windows × {MAX_SCENES_PER_WINDOW} scenes")
    for i, (year, start_md, end_md, label) in enumerate(DOWNLOADS):
        env = os.environ.copy()
        env["HLS_BBOX"]       = BBOX
        env["HLS_DATE_START"] = f"{year}-{start_md}"
        env["HLS_DATE_END"]   = f"{year}-{end_md}"
        env["HLS_MAX_SCENES"] = str(MAX_SCENES_PER_WINDOW)
        env["HLS_MAX_CLOUD"]  = str(MAX_CLOUD)
        env["HLS_PRODUCT"]    = "both"

        print(f"\n=== [{i+1}/{len(DOWNLOADS)}] {label} ===", flush=True)
        result = subprocess.run([sys.executable, str(fetch_script)], env=env, cwd=str(ROOT))
        if result.returncode != 0:
            print(f"  ⚠ {label} failed, continuing...")

    print("\n✓ June download done")


if __name__ == "__main__":
    main()
