# AWS Integration Setup for VS Code (Hackathon)

## 1. Configure AWS Credentials

The safest approach for temporary credentials is the AWS CLI config file — **not** your `.env`.

```powershell
# Install AWS CLI if not already installed
winget install Amazon.AWSCLI

# Configure your credentials
aws configure
```

This stores credentials in `~/.aws/credentials`, outside the repo and never accidentally committed.

If your hackathon issued **temporary session tokens** (STS), set them individually:

```powershell
aws configure set aws_access_key_id YOUR_KEY
aws configure set aws_secret_access_key YOUR_SECRET
aws configure set aws_session_token YOUR_TOKEN
aws configure set region us-east-1  # or your hackathon region
```

---

## 2. Install the AWS Toolkit Extension

Search for **"AWS Toolkit"** (by Amazon Web Services) in the VS Code Extensions panel.

Features include:
- S3 bucket browser
- Lambda / EC2 explorer
- CloudWatch Logs viewer
- Direct credential profile switching

---

## 3. Use AWS in Python

Install `boto3`:

```powershell
pip install boto3
```

Credentials are picked up automatically from `~/.aws/credentials` — no keys in code:

```python
import boto3

s3 = boto3.client('s3')  # credentials loaded from ~/.aws/credentials automatically
```

---

## 4. Protect Your Credentials

- **Never** add AWS keys to `.env` or any source file
- Make sure `.env` and `.aws/` are listed in your `.gitignore`
- If your hackathon uses a named AWS profile, activate it before launching VS Code:

```powershell
$env:AWS_PROFILE = "your-profile-name"
code .
```

---

## 5. Recommended AWS Tools for This Project

### S3 — Data Storage (Start Here)

Your pipeline ingests large geospatial files (NAIP tiles, HLS imagery, CDL rasters) that don't belong in a repo. S3 is the foundation everything else reads from.

```python
import boto3, rasterio
from io import BytesIO

s3 = boto3.client('s3')

def load_naip_from_s3(bucket, key):
    obj = s3.get_object(Bucket=bucket, Key=key)
    with rasterio.open(BytesIO(obj['Body'].read())) as src:
        return src.read()
```

Suggested bucket layout:
```
s3://cornsight-data/
  raw/naip/{state}/{tile}.tif
  raw/hls/{state}/{date}/
  processed/features/{state}/{date}.npy
  outputs/forecasts/
  models/
```

---

### EC2 (GPU) — Prithvi Feature Extraction

`privthi_extractor.py` loads `ibm-nasa-geospatial/Prithvi-100M` — a 100M parameter model that will be very slow on CPU. A `g4dn.xlarge` (NVIDIA T4, ~$0.50/hr) is the right instance for this.

- Use a **Deep Learning AMI** (comes with CUDA/PyTorch pre-installed)
- Pull your repo, run extraction against tiles stored in S3

---

### AWS Batch — Parallel Tile Processing

You have 5 states × 4 forecast dates × many county tiles. Batch lets you fan out `extract_features()` jobs across many containers in parallel instead of running them sequentially.

Each job: pull one NAIP tile from S3 → run Prithvi → write `.npy` feature back to S3.

---

### SageMaker — Forecasting Model Training

`forecaster.py` is where XGBoost training goes. SageMaker has a built-in XGBoost container so you don't need to manage infrastructure:

```python
from sagemaker.xgboost import XGBoost

estimator = XGBoost(
    entry_point='forecaster.py',
    role='your-sagemaker-role',
    instance_type='ml.m5.xlarge',
    framework_version='1.7-1'
)
estimator.fit({'train': 's3://cornsight-data/processed/features/'})
```

---

### Step Functions — Pipeline Orchestration

The pipeline has a natural 4-stage sequence (Aug 1 → Sep 1 → Oct 1 → Final). Step Functions can chain:

`Ingest data → Extract Prithvi features → Train/update model → Generate forecast + cone of uncertainty`

This is especially useful for re-running the pipeline at each forecast date.

---

### Priority Order for a Hackathon

| Priority | Service | Why |
|---|---|---|
| 1 | **S3** | Everything reads/writes here |
| 2 | **EC2 GPU** | Prithvi won't run usably on CPU |
| 3 | **SageMaker** | Clean training + model artifact storage |
| 4 | **Batch** | Only needed if tile count is large |
| 5 | **Step Functions** | Nice-to-have for demo polish |

Start with S3 + one EC2 GPU instance — that unblocks `privthi_extractor.py` and gives you a place to store all your raster data.
