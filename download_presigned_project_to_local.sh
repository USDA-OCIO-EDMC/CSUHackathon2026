#!/usr/bin/env bash
set -euo pipefail

# Run this on your Mac from /Users/elliott/Desktop/hack.
# Usage:
#   ./download_presigned_project_to_local.sh "https://presigned-s3-url..."

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 'PRESIGNED_DOWNLOAD_URL'"
  exit 1
fi

URL="$1"
DEST_DIR="${DEST_DIR:-sagemakerenv}"
ZIP_PATH="${DEST_DIR}/project_export.zip"

mkdir -p "${DEST_DIR}"

echo "Downloading project package to ${ZIP_PATH}..."
curl -L "${URL}" -o "${ZIP_PATH}"

echo "Unpacking into $(pwd)/${DEST_DIR}..."
python3 - <<PY
from pathlib import Path
from zipfile import ZipFile

zip_path = Path("${ZIP_PATH}")
with ZipFile(zip_path) as zf:
    zf.extractall(".")
print(f"Extracted {zip_path}")
PY

echo
echo "Done. Local files are in $(pwd)/${DEST_DIR}/"
