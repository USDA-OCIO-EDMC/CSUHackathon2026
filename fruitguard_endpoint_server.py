#!/usr/bin/env python3
"""Serve FruitGuard static files and proxy prediction requests to SageMaker.

Use this instead of plain `python -m http.server` when you want the browser app
to call a live SageMaker endpoint.
"""

from __future__ import annotations

import json
import os
import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import boto3


HOST = "127.0.0.1"
PORT = 8765
REGION = "us-west-2"
ENDPOINT_NAME = ""


class FruitGuardHandler(SimpleHTTPRequestHandler):
    def _send_json(self, status_code, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/model-status":
            self._send_json(
                200,
                {
                    "endpoint_configured": bool(ENDPOINT_NAME),
                    "endpoint_name": ENDPOINT_NAME,
                    "region": REGION,
                },
            )
            return
        super().do_GET()

    def do_POST(self):
        if self.path != "/api/predict":
            self._send_json(404, {"error": "Unknown API route"})
            return

        if not ENDPOINT_NAME:
            self._send_json(
                503,
                {
                    "error": "SAGEMAKER_ENDPOINT_NAME is not set.",
                    "model_source": "batch-dashboard-fallback",
                },
            )
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(content_length)
            runtime = boto3.client("sagemaker-runtime", region_name=REGION)
            response = runtime.invoke_endpoint(
                EndpointName=ENDPOINT_NAME,
                ContentType="application/json",
                Body=body,
            )
            result = json.loads(response["Body"].read().decode("utf-8"))
            self._send_json(200, result)
        except Exception as exc:
            self._send_json(500, {"error": str(exc), "model_source": "endpoint-error"})


def parse_args():
    parser = argparse.ArgumentParser(description="Serve FruitGuard with an optional SageMaker endpoint proxy.")
    parser.add_argument("--host", default=os.environ.get("HOST", HOST))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", str(PORT))))
    parser.add_argument(
        "--region",
        default=os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or REGION,
    )
    parser.add_argument("--endpoint-name", default=os.environ.get("SAGEMAKER_ENDPOINT_NAME", ""))
    return parser.parse_args()


def main():
    global HOST, PORT, REGION, ENDPOINT_NAME
    args = parse_args()
    HOST = args.host
    PORT = args.port
    REGION = args.region
    ENDPOINT_NAME = args.endpoint_name

    print("FruitGuard endpoint app server")
    print(f"  URL      : http://{HOST}:{PORT}/fruitguard_live.html")
    print(f"  ML view  : http://{HOST}:{PORT}/ml_dashboard.html")
    print(f"  Endpoint : {ENDPOINT_NAME or '(not configured)'}")
    print("Press Ctrl+C to stop.")
    server = ThreadingHTTPServer((HOST, PORT), FruitGuardHandler)
    server.serve_forever()


if __name__ == "__main__":
    main()
