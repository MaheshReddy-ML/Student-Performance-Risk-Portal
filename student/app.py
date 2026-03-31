from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:
    from .model import PREDICTOR
except ImportError:
    from model import PREDICTOR


BASE_DIR = Path(__file__).resolve().parent
INDEX_FILE = BASE_DIR / "index.html"

PROJECT_FILES = [
    {
        "name": "index.html",
        "purpose": "Single frontend document",
        "details": "Contains the only HTML doctype in the project, the full UI, inline styling, and browser-side logic.",
    },
    {
        "name": "app.py",
        "purpose": "Backend server",
        "details": "Serves the frontend, exposes JSON APIs, and handles prediction requests locally with the trained model.",
    },
    {
        "name": "model.py",
        "purpose": "Prediction engine",
        "details": "Loads the CSV, trains the classifier, normalizes incoming form data, and generates advice with no inline comments.",
    },
    {
        "name": "student_habits.csv",
        "purpose": "Training dataset",
        "details": "Supplies the historical student habit records used to learn risk patterns.",
    },
    {
        "name": "requirements.txt",
        "purpose": "Python dependencies",
        "details": "Lists the packages needed to run the backend and train the model in a fresh environment.",
    },
    {
        "name": ".gitignore",
        "purpose": "Workspace hygiene",
        "details": "Keeps caches, virtual environments, logs, and machine-specific files out of version control.",
    },
]


class StudentPortalHandler(BaseHTTPRequestHandler):
    server_version = "StudentPortal/1.0"

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in {"/", "/index.html"}:
            self._send_html(INDEX_FILE.read_text(encoding="utf-8"))
            return
        if path == "/api/health":
            self._send_json(
                {
                    "status": "ok",
                    "app_name": "Student Performance Risk Portal",
                    "project_files": PROJECT_FILES,
                    **PREDICTOR.overview(),
                }
            )
            return
        self._send_json({"error": "Route not found."}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/api/predict":
            self._send_json({"error": "Route not found."}, status=HTTPStatus.NOT_FOUND)
            return

        try:
            payload = self._read_json_body()
            if not isinstance(payload, dict):
                raise ValueError("JSON body must be an object.")
            prediction = PREDICTOR.predict(payload)
        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON body."}, status=HTTPStatus.BAD_REQUEST)
            return
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        except Exception as exc:
            self._send_json(
                {"error": f"Prediction failed: {exc}"},
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            return

        self._send_json(prediction.to_dict())

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _read_json_body(self) -> Any:
        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length) if content_length else b"{}"
        return json.loads(raw_body.decode("utf-8"))

    def _send_html(self, html: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)


def run_server(host: str, port: int) -> None:
    server = ThreadingHTTPServer((host, port), StudentPortalHandler)
    print(f"Student Performance Risk Portal is running on http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
    finally:
        server.server_close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the student risk prediction portal.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind.")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_server(args.host, args.port)
