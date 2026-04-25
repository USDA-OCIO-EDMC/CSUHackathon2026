#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

S3_URI="${S3_URI:-s3://bucket-for-xgboost/project/}"
DEST_DIR="${DEST_DIR:-sagemakerenv}"

mkdir -p "${DEST_DIR}"

echo "Downloading project files"
echo "  From: ${S3_URI}"
echo "  To  : $(pwd)/${DEST_DIR}"
echo

aws s3 sync "${S3_URI}" "${DEST_DIR}/"

echo
echo "Done. Files downloaded into ${DEST_DIR}/"
