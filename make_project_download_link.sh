#!/usr/bin/env bash
set -euo pipefail

# Run this in SageMaker Studio. It packages s3://bucket-for-xgboost/project/
# and prints a temporary HTTPS download URL for your local Mac.

S3_PROJECT_URI="${S3_PROJECT_URI:-s3://bucket-for-xgboost/project/}"
EXPORT_BUCKET="${EXPORT_BUCKET:-bucket-for-xgboost}"
EXPORT_KEY="${EXPORT_KEY:-project_export/project_export.zip}"
EXPIRES_SECONDS="${EXPIRES_SECONDS:-3600}"
WORK_DIR="${WORK_DIR:-/tmp/fruitguard_project_export}"

rm -rf "${WORK_DIR}"
mkdir -p "${WORK_DIR}/sagemakerenv"

echo "Syncing ${S3_PROJECT_URI} into a temporary package folder..."
aws s3 sync "${S3_PROJECT_URI}" "${WORK_DIR}/sagemakerenv/"

echo "Creating zip package..."
python3 - <<PY
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

work_dir = Path("${WORK_DIR}")
zip_path = work_dir / "project_export.zip"
source_dir = work_dir / "sagemakerenv"

with ZipFile(zip_path, "w", ZIP_DEFLATED) as zf:
    for path in source_dir.rglob("*"):
        if path.is_file():
            zf.write(path, path.relative_to(work_dir))

print(f"Created {zip_path}")
PY

echo "Uploading package to s3://${EXPORT_BUCKET}/${EXPORT_KEY}..."
aws s3 cp "${WORK_DIR}/project_export.zip" "s3://${EXPORT_BUCKET}/${EXPORT_KEY}"

echo
echo "Temporary download URL:"
aws s3 presign "s3://${EXPORT_BUCKET}/${EXPORT_KEY}" --expires-in "${EXPIRES_SECONDS}"
echo
echo "This URL expires in ${EXPIRES_SECONDS} seconds."
