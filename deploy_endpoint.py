#!/usr/bin/env python3
"""Deploy the completed FruitGuard model artifact as a SageMaker endpoint.

Run this in SageMaker Studio or another AWS-authenticated environment.
Creating an endpoint starts a billable AWS resource; delete it when the demo is done.
"""

import argparse
import datetime as dt
from urllib.parse import urlparse
import tarfile
import time
from pathlib import Path

import boto3
from botocore.exceptions import ClientError


DEFAULT_MODEL_ARTIFACT = ""
DEFAULT_IMAGE_URI = (
    "246618743249.dkr.ecr.us-west-2.amazonaws.com/"
    "sagemaker-scikit-learn:1.2-1-cpu-py3"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Deploy FruitGuard SageMaker endpoint.")
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument("--bucket", default="bucket-for-xgboost")
    parser.add_argument(
        "--model-artifact",
        default=DEFAULT_MODEL_ARTIFACT,
        help="S3 path printed by launch_training.py, ending in output/model.tar.gz.",
    )
    parser.add_argument(
        "--training-job-name",
        default="",
        help="Use the model artifact from this specific SageMaker training job.",
    )
    parser.add_argument(
        "--job-prefix",
        default="fruitfly-xgb",
        help="When --model-artifact is omitted, use the latest completed job with this prefix.",
    )
    parser.add_argument("--endpoint-name", default="fruitfly-risk-endpoint")
    parser.add_argument("--instance-type", default="ml.m5.large")
    parser.add_argument("--role-arn", default="")
    parser.add_argument(
        "--update-existing",
        action="store_true",
        help="If the endpoint already exists, update it to the new model instead of failing.",
    )
    parser.add_argument("--wait", action="store_true", help="Wait until the endpoint is in service.")
    return parser.parse_args()


def find_sagemaker_role(iam_client):
    roles = iam_client.list_roles(MaxItems=100)
    for role in roles["Roles"]:
        name = role["RoleName"]
        if "SageMaker" in name or "sagemaker" in name:
            return role["Arn"]
    raise RuntimeError("No SageMaker execution role found. Pass --role-arn explicitly.")


def package_inference_source(bucket, region):
    source_tar = Path("inference_source.tar.gz")
    with tarfile.open(source_tar, "w:gz") as tar:
        tar.add("inference.py")
        tar.add("requirements.txt")

    key = "source/inference_source.tar.gz"
    boto3.client("s3", region_name=region).upload_file(str(source_tar), bucket, key)
    return f"s3://{bucket}/{key}"


def endpoint_exists(sm_client, endpoint_name):
    try:
        sm_client.describe_endpoint(EndpointName=endpoint_name)
        return True
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code in {"ValidationException", "ResourceNotFound", "ResourceNotFoundException"}:
            return False
        raise


def model_artifact_from_job(sm_client, training_job_name):
    status = sm_client.describe_training_job(TrainingJobName=training_job_name)
    job_status = status["TrainingJobStatus"]
    if job_status != "Completed":
        raise RuntimeError(f"Training job {training_job_name} is {job_status}, not Completed.")
    return status["ModelArtifacts"]["S3ModelArtifacts"]


def latest_completed_model_artifact(sm_client, job_prefix):
    response = sm_client.list_training_jobs(
        NameContains=job_prefix,
        StatusEquals="Completed",
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=10,
    )
    summaries = response.get("TrainingJobSummaries", [])
    if not summaries:
        raise RuntimeError(
            f"No completed SageMaker training jobs found with prefix '{job_prefix}'. "
            "Run launch_training.py first, or pass --model-artifact explicitly."
        )
    job_name = summaries[0]["TrainingJobName"]
    artifact = model_artifact_from_job(sm_client, job_name)
    print(f"Using latest completed training job: {job_name}")
    return artifact


def parse_s3_uri(uri):
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError(f"Expected an S3 URI like s3://bucket/path/model.tar.gz, got: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def validate_model_artifact(s3_client, model_artifact):
    if "YOUR_NEW_JOB" in model_artifact:
        raise ValueError(
            "Replace YOUR_NEW_JOB with the actual training job folder, or omit "
            "--model-artifact to use the latest completed fruitfly-xgb job."
        )
    bucket, key = parse_s3_uri(model_artifact)
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
    except Exception as exc:
        raise FileNotFoundError(
            f"Could not find model artifact at {model_artifact}. Copy the exact "
            "'Model artifacts:' path printed by launch_training.py."
        ) from exc


def main():
    args = parse_args()
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_name = f"{args.endpoint_name}-model-{timestamp}"
    config_name = f"{args.endpoint_name}-config-{timestamp}"

    iam = boto3.client("iam", region_name=args.region)
    sm = boto3.client("sagemaker", region_name=args.region)
    s3 = boto3.client("s3", region_name=args.region)
    role_arn = args.role_arn or find_sagemaker_role(iam)
    source_uri = package_inference_source(args.bucket, args.region)
    model_artifact = args.model_artifact.strip()

    if args.training_job_name:
        model_artifact = model_artifact_from_job(sm, args.training_job_name)
    elif not model_artifact:
        model_artifact = latest_completed_model_artifact(sm, args.job_prefix)

    validate_model_artifact(s3, model_artifact)
    existing_endpoint = endpoint_exists(sm, args.endpoint_name)
    if existing_endpoint and not args.update_existing:
        raise RuntimeError(
            f"Endpoint already exists: {args.endpoint_name}. Re-run with --update-existing "
            "to point it at the new model, or choose a new --endpoint-name."
        )

    print(f"Using role: {role_arn}")
    print(f"Model artifact: {model_artifact}")
    print(f"Inference source: {source_uri}")

    sm.create_model(
        ModelName=model_name,
        ExecutionRoleArn=role_arn,
        PrimaryContainer={
            "Image": DEFAULT_IMAGE_URI,
            "ModelDataUrl": model_artifact,
            "Environment": {
                "SAGEMAKER_PROGRAM": "inference.py",
                "SAGEMAKER_SUBMIT_DIRECTORY": source_uri,
                "SAGEMAKER_REQUIREMENTS": "requirements.txt",
                "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
                "SAGEMAKER_REGION": args.region,
            },
        },
    )

    sm.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "InitialInstanceCount": 1,
                "InstanceType": args.instance_type,
            }
        ],
    )

    if existing_endpoint:
        sm.update_endpoint(EndpointName=args.endpoint_name, EndpointConfigName=config_name)
        print(f"Endpoint update started: {args.endpoint_name}")
    else:
        sm.create_endpoint(EndpointName=args.endpoint_name, EndpointConfigName=config_name)
        print(f"Endpoint creation started: {args.endpoint_name}")

    if args.wait:
        while True:
            status = sm.describe_endpoint(EndpointName=args.endpoint_name)["EndpointStatus"]
            print(f"  Status: {status}")
            if status in {"InService", "Failed"}:
                break
            time.sleep(30)
        if status == "InService":
            print("===== ENDPOINT READY =====")
        else:
            raise RuntimeError("Endpoint deployment failed. Check SageMaker endpoint logs.")


if __name__ == "__main__":
    main()
