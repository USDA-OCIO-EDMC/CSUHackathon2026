
import boto3
import tarfile
import datetime
import os

region = "us-west-2"
bucket = "bucket-for-xgboost"

# ============================================================
# 1. Get your SageMaker execution role
# ============================================================
iam = boto3.client("iam", region_name=region)
roles = iam.list_roles(MaxItems=100)
sm_role = None
for r in roles["Roles"]:
    if "SageMaker" in r["RoleName"] or "sagemaker" in r["RoleName"]:
        sm_role = r["Arn"]
        break

if sm_role is None:
    print("ERROR: No SageMaker execution role found.")
    print("Go to IAM -> Roles and find your SageMaker role ARN.")
    exit(1)

print(f"Using role: {sm_role}")

# ============================================================
# 2. Package training script + requirements.txt into tarball
# ============================================================
s3 = boto3.client("s3", region_name=region)

with tarfile.open("sourcedir.tar.gz", "w:gz") as tar:
    tar.add("train_fruitfly.py")
    tar.add("requirements.txt")  # <-- NOW INCLUDED

s3.upload_file("sourcedir.tar.gz", bucket, "source/sourcedir.tar.gz")
print("Uploaded training script + requirements.txt to S3")

# ============================================================
# 3. Launch training job
# ============================================================
sm = boto3.client("sagemaker", region_name=region)
image_uri = "246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3"
job_name = f"fruitfly-xgb-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"

response = sm.create_training_job(
    TrainingJobName=job_name,
    AlgorithmSpecification={
        "TrainingImage": image_uri,
        "TrainingInputMode": "File",
    },
    RoleArn=sm_role,
    InputDataConfig=[
        {
            "ChannelName": "training",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": f"s3://{bucket}/data/",
                    "S3DataDistributionType": "FullyReplicated",
                }
            },
            "ContentType": "text/csv",
        }
    ],
    OutputDataConfig={
        "S3OutputPath": f"s3://{bucket}/model-output"
    },
    ResourceConfig={
        "InstanceCount": 1,
        "InstanceType": "ml.m5.xlarge",
        "VolumeSizeInGB": 10,
    },
    StoppingCondition={"MaxRuntimeInSeconds": 3600},
    HyperParameters={
        "sagemaker_program": "train_fruitfly.py",
        "sagemaker_submit_directory": f"s3://{bucket}/source/sourcedir.tar.gz",
        "sagemaker_requirements": "requirements.txt",
    },
)

print(f"Training job launched: {job_name}")
print(f"Monitor in AWS Console: SageMaker -> Training -> Training jobs -> {job_name}")

# ============================================================
# 4. Wait for completion and stream status
# ============================================================
import time

print("Waiting for training job to complete...")
while True:
    status = sm.describe_training_job(TrainingJobName=job_name)
    current = status["TrainingJobStatus"]
    print(f"  Status: {current}")

    if current in ["Completed", "Failed", "Stopped"]:
        break
    time.sleep(30)

if current == "Completed":
    print(f"===== TRAINING COMPLETE =====")
    print(f"Model artifacts: {status['ModelArtifacts']['S3ModelArtifacts']}")
elif current == "Failed":
    print(f"===== TRAINING FAILED =====")
    print(f"Reason: {status.get('FailureReason', 'Unknown')}")
else:
    print(f"Job stopped with status: {current}")

