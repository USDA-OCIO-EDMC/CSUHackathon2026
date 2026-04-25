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
