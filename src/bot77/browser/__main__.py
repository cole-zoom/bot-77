"""uv run python -m bot77.browser [--port 8077]  -> the data browser at http://127.0.0.1:8077 (local only)."""
import argparse

import uvicorn

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8077)
ap.add_argument("--reload", action="store_true")
a = ap.parse_args()
uvicorn.run("bot77.browser.app:app", host="127.0.0.1", port=a.port, reload=a.reload)
