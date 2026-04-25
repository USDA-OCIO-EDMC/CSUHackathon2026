"""
Submit a SageMaker Processing Job to extract Prithvi embeddings on GPU.

Usage:
    # Single state/year
    python3 sagemaker_run.py --state IA --year 2023

    # All states for a year, submitted as parallel jobs
    python3 sagemaker_run.py --year 2023 --all-states

Output lands at:
    s3://cornsight-data/processed/prithvi_embeddings/{STATE}/{YEAR}/{FORECAST_DATE}/embeddings.parquet
"""

import os
import argparse
from dotenv import load_dotenv, find_dotenv
from sagemaker.core.processing import FrameworkProcessor, ProcessingOutput
from sagemaker.core.shapes.shapes import ProcessingS3Output
from sagemaker.core.network import NetworkConfig

load_dotenv(find_dotenv(usecwd=False), override=True)

ROLE   = "arn:aws:iam::851725650419:role/AmazonSageMakerExecutionRole-sagemaker-domain-with-vpc-hackathon"
BUCKET = "cornsight-data"
STATES = ["IA", "CO", "WI", "MO", "NE"]

# PyTorch 2.1 GPU container (us-west-2)
IMAGE_URI = "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.1.0-cpu-py310-ubuntu20.04-sagemaker"

NETWORK_CONFIG = NetworkConfig(
    subnets=["subnet-0ddd6c67b3cbab520", "subnet-0b79bbefaffe2dd7d"],
    security_group_ids=["sg-02c7fef5be7a3d1fc"],
)


def submit_job(state: str, year: int, forecast_date: str, wait: bool = False):
    processor = FrameworkProcessor(
        image_uri=IMAGE_URI,
        command=["python3"],
        role=ROLE,
        instance_type="ml.m4.xlarge",
        instance_count=1,
        base_job_name=f"prithvi-{state.lower()}-{year}",
        env={
            "EARTHDATA_USERNAME": os.environ["EARTHDATA_USERNAME"],
            "EARTHDATA_PASSWORD": os.environ["EARTHDATA_PASSWORD"],
        },
        network_config=NETWORK_CONFIG,
    )

    s3_output = f"s3://{BUCKET}/processed/prithvi_embeddings/{state}/{year}/{forecast_date}/"

    processor.run(
        code="sagemaker_processor.py",
        source_dir="src/",
        arguments=[
            "--state", state,
            "--year", str(year),
            "--forecast-date", forecast_date,
        ],
        outputs=[
            ProcessingOutput(
                output_name="embeddings",
                s3_output=ProcessingS3Output(
                    s3_uri=s3_output,
                    s3_upload_mode="EndOfJob",
                    local_path="/opt/ml/processing/output",
                ),
            )
        ],
        wait=wait,
        logs=wait,
    )

    job_name = processor.latest_job.processing_job_name
    print(f"Submitted: {job_name}")
    print(f"  Output → {s3_output}")
    return job_name


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--state", type=str, help="Single state: IA, CO, WI, MO, NE")
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--forecast-date", type=str, default="final",
                   choices=["aug1", "sep1", "oct1", "final"])
    p.add_argument("--all-states", action="store_true",
                   help="Submit one job per state in parallel")
    p.add_argument("--wait", action="store_true",
                   help="Block until job completes and stream logs")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.all_states:
        job_names = []
        for state in STATES:
            jn = submit_job(state, args.year, args.forecast_date, wait=False)
            job_names.append(jn)
        print(f"\n{len(job_names)} jobs running in parallel.")
        print("Check status:")
        print("  aws sagemaker list-processing-jobs --status-equals InProgress")
    else:
        if not args.state:
            raise ValueError("Provide --state or --all-states")
        submit_job(args.state, args.year, args.forecast_date, wait=args.wait)
