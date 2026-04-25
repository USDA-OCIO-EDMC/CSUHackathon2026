#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python3}"
AWS_REGION="${AWS_REGION:-us-west-2}"
ENDPOINT_NAME="${ENDPOINT_NAME:-fruitfly-risk-endpoint}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8765}"

# Set any of these to 1 to skip that step during rapid demo iteration.
SKIP_TRAINING="${SKIP_TRAINING:-0}"
SKIP_DEPLOY="${SKIP_DEPLOY:-0}"
SKIP_DASHBOARDS="${SKIP_DASHBOARDS:-0}"
SKIP_SERVER="${SKIP_SERVER:-0}"

echo "============================================================"
echo "FRUITGUARD — Full SageMaker Automation"
echo "============================================================"
echo "Region        : ${AWS_REGION}"
echo "Endpoint      : ${ENDPOINT_NAME}"
echo "App URL       : http://${HOST}:${PORT}/fruitguard_live.html"
echo "ML URL        : http://${HOST}:${PORT}/ml_dashboard.html"
echo "Training data : s3://bucket-for-xgboost/data/"
echo
echo "Note: a SageMaker endpoint is billable while it is running."
echo

echo "0/4 Validating local scripts..."
"$PYTHON_BIN" -m py_compile \
  launch_training.py \
  train_fruitfly.py \
  deploy_endpoint.py \
  inference.py \
  fruitguard_endpoint_server.py \
  run_pipeline.py \
  fruitguard_ml.py

if [[ "${SKIP_TRAINING}" == "1" ]]; then
  echo
  echo "1/4 Skipping SageMaker training because SKIP_TRAINING=1"
else
  echo
  echo "1/4 Launching SageMaker XGBoost training job..."
  "$PYTHON_BIN" launch_training.py
fi

if [[ "${SKIP_DEPLOY}" == "1" ]]; then
  echo
  echo "2/4 Skipping endpoint deploy/update because SKIP_DEPLOY=1"
else
  echo
  echo "2/4 Deploying/updating SageMaker endpoint from latest completed fruitfly-xgb job..."
  "$PYTHON_BIN" deploy_endpoint.py \
    --region "${AWS_REGION}" \
    --endpoint-name "${ENDPOINT_NAME}" \
    --update-existing \
    --wait
fi

if [[ "${SKIP_DASHBOARDS}" == "1" ]]; then
  echo
  echo "3/4 Skipping dashboard regeneration because SKIP_DASHBOARDS=1"
else
  echo
  echo "3/4 Regenerating dashboard files..."
  "$PYTHON_BIN" run_pipeline.py
  "$PYTHON_BIN" fruitguard_ml.py
fi

if [[ "${SKIP_SERVER}" == "1" ]]; then
  echo
  echo "4/4 Skipping app server because SKIP_SERVER=1"
  echo "Run this when ready:"
  echo "  SAGEMAKER_ENDPOINT_NAME=${ENDPOINT_NAME} AWS_REGION=${AWS_REGION} ./run_endpoint_app.sh"
  exit 0
fi

echo
echo "4/4 Starting endpoint-backed app server..."
echo "Open the Studio forwarded URL for port ${PORT}, then open /ml_dashboard.html"
echo "Press Ctrl+C to stop the app server."
echo

SAGEMAKER_ENDPOINT_NAME="${ENDPOINT_NAME}" \
AWS_REGION="${AWS_REGION}" \
"$PYTHON_BIN" fruitguard_endpoint_server.py \
  --host "${HOST}" \
  --port "${PORT}" \
  --region "${AWS_REGION}" \
  --endpoint-name "${ENDPOINT_NAME}"
