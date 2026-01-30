"""
Simple HTTP server for live plot visualization.
return
"""

import os
import json
import threading
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from functools import partial

_server_instance = None
_server_lock = threading.Lock()

PLOTS_DIR = os.path.abspath(os.path.join("saved_data", "plots"))
LIVE_PLOTS_DIR = os.path.join(PLOTS_DIR, "live_plots")
MANIFEST_PATH = os.path.join(LIVE_PLOTS_DIR, "manifest.json")
INDEX_PATH = os.path.join(LIVE_PLOTS_DIR, "index.html")
DEFAULT_PORT = 8765


class PlotRequestHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Suppress logging


def _run_server(port: int):
    os.makedirs(LIVE_PLOTS_DIR, exist_ok=True)
    handler = partial(PlotRequestHandler, directory=LIVE_PLOTS_DIR)
    server = HTTPServer(("127.0.0.1", port), handler)
    server.serve_forever()


def start_plot_server(port: int = DEFAULT_PORT, open_browser: bool = True) -> str:
    global _server_instance
    with _server_lock:
        if _server_instance is not None:
            return f"http://127.0.0.1:{port}/index.html"
        
        _ensure_manifest_exists()

        thread = threading.Thread(target=_run_server, args=(port,), daemon=True)
        thread.start()
        _server_instance = thread

        url = f"http://127.0.0.1:{port}/index.html"
        if open_browser:
            webbrowser.open(url)
        print(f"[PlotServer] Started at {url}")
        return url


def _ensure_manifest_exists():
    os.makedirs(LIVE_PLOTS_DIR, exist_ok=True)
    if os.path.exists(MANIFEST_PATH):
        return
    with open(MANIFEST_PATH, "w") as f:
        json.dump({"plots": {}}, f)


def update_manifest(plots: dict[str, list[str]]):
    """
    Update the manifest with new plots.
    plots: dict mapping category -> list of relative filepaths (from live_plots dir)
    return
    """
    os.makedirs(LIVE_PLOTS_DIR, exist_ok=True)
    existing: dict[str, list[str]] = {}
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r") as f:
            data = json.load(f)
            existing = data.get("plots", {})

    existing.update(plots)
    with open(MANIFEST_PATH, "w") as f:
        json.dump({"plots": existing, "timestamp": _get_timestamp()}, f, indent=2)


def _get_timestamp() -> str:
    import datetime
    return datetime.datetime.now().isoformat()


