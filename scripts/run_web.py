"""
Launch the Smart Assignment chat web app (smart_assignment/webapp).

A convenience wrapper around uvicorn, for parity with scripts/run_local.py and
scripts/generate_page.py. Runs the deterministic pipeline fully offline -- no
API key required.

    python3 scripts/run_web.py                 # http://127.0.0.1:8000
    python3 scripts/run_web.py --port 9000 --reload

Equivalent to:  uvicorn smart_assignment.webapp.app:app
"""

from __future__ import annotations

import argparse
import os

# Importing the package is credential-free (root_agent is built lazily), so this
# is not needed just to start the app. It defaults the webapp to the
# credential-free "standard" backend as a convenience, so conversational (llm)
# mode activates as soon as you add a GOOGLE_API_KEY -- without also having to
# set the backend. An explicitly-set SMART_ASSIGNMENT_LLM_BACKEND still wins.
os.environ.setdefault("SMART_ASSIGNMENT_LLM_BACKEND", "standard")


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the Smart Assignment chat web app")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes")
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - guidance path
        raise SystemExit(
            "uvicorn is not installed. Install the web extra:\n" '    pip install -e ".[web]"'
        ) from exc

    uvicorn.run(
        "smart_assignment.webapp.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
