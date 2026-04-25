#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python3}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8765}"
SAGEMAKER_ENDPOINT_NAME="${SAGEMAKER_ENDPOINT_NAME:-fruitfly-risk-endpoint}"
AWS_REGION="${AWS_REGION:-us-west-2}"

export HOST PORT SAGEMAKER_ENDPOINT_NAME AWS_REGION

echo "============================================================"
echo "FRUITGUARD — Endpoint-Backed App Server"
echo "============================================================"
echo "Endpoint: ${SAGEMAKER_ENDPOINT_NAME}"
echo "Main dashboard: http://${HOST}:${PORT}/fruitguard_live.html"
echo "ML pathways:    http://${HOST}:${PORT}/ml_dashboard.html"
echo

"$PYTHON_BIN" fruitguard_endpoint_server.py
