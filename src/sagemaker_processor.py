"""
SageMaker Processing Job entry point — runs inside the container.

Installs missing deps, then runs the Prithvi extraction pipeline for
one state/year and writes the parquet to /opt/ml/processing/output/
so SageMaker uploads it to S3 automatically.
"""

import subprocess
import sys

# Install deps not in the base PyTorch container
subprocess.check_call([
    sys.executable, "-m", "pip", "install",
    "earthaccess", "huggingface_hub", "python-dotenv", "pyarrow", "tqdm",
    "rasterio", "xgboost", "scikit-learn", "einops==0.8.0", "timm==1.0.26", "requests",
])

import os
import argparse
from pathlib import Path

# SageMaker copies source_dir contents to /opt/ml/code — make them importable
sys.path.insert(0, "/opt/ml/code")

OUTPUT_DIR = "/opt/ml/processing/output"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--state", type=str, required=True)
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--forecast-date", type=str, default="final",
                   choices=["aug1", "sep1", "oct1", "final"])
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Credentials are passed as job environment variables by sagemaker_run.py
    for var in ("EARTHDATA_USERNAME", "EARTHDATA_PASSWORD"):
        if not os.environ.get(var):
            raise EnvironmentError(f"Missing required env var: {var}")

    from prithvi_pipeline import build_prithvi_embeddings

    df = build_prithvi_embeddings(
        state_abbr=args.state,
        year=args.year,
        forecast_date=args.forecast_date,
        device="cpu",
        output_dir=OUTPUT_DIR,
    )

    if df.empty:
        print("WARNING: no embeddings extracted — output parquet not written.")
    else:
        print(f"Done. {len(df)} records written to {OUTPUT_DIR}/")
