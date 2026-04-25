#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python3}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8765}"

echo "============================================================"
echo "FRUITGUARD — Build and Launch"
echo "============================================================"

echo "1/3 Regenerating dashboard data..."
"$PYTHON_BIN" run_pipeline.py

echo
echo "2/3 Running ML pathway forecast..."
"$PYTHON_BIN" fruitguard_ml.py

echo
echo "3/3 Starting local web server..."
echo "Main dashboard: http://${HOST}:${PORT}/fruitguard_live.html"
echo "ML pathways:    http://${HOST}:${PORT}/ml_dashboard.html"
echo "Press Ctrl+C to stop the server."
echo

if command -v open >/dev/null 2>&1; then
  open "http://${HOST}:${PORT}/fruitguard_live.html"
fi

if command -v lsof >/dev/null 2>&1 && lsof -nP -iTCP:"${PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Port ${PORT} is already in use, so I refreshed the generated files and opened the dashboard."
  echo "Refresh the browser if it was already open."
  exit 0
fi

"$PYTHON_BIN" -m http.server "$PORT" --bind "$HOST"
