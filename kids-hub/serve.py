"""
Static file server for the kids' app hub.

Deliberately stdlib-only -- no Flask, no `serve`, no npm install. The hub is one
HTML file, so the server's whole job is "hand out index.html on the port Railway
gave us." Fewer dependencies means fewer things that can break a deploy months
from now, and the container starts in about a second.

Run locally:   python3 serve.py       (then open http://localhost:3000)
On Railway:    the Dockerfile's CMD, with $PORT injected by the platform.
"""

from __future__ import annotations

import http.server
import os
import socketserver
from functools import partial

DIRECTORY = os.path.dirname(os.path.abspath(__file__))
PORT = int(os.environ.get("PORT", "3000"))


class _Handler(http.server.SimpleHTTPRequestHandler):
    """Serves the hub directory, with caching turned off.

    No-cache matters here specifically: adding one of the kids' apps means
    editing `index.html` and redeploying, and a cached copy would leave them
    staring at the old list wondering why their new app didn't show up."""

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        super().end_headers()

    def log_message(self, fmt: str, *args) -> None:
        # One tidy line per request in the Railway logs (the default writes to
        # stderr with a timestamp Railway already stamps itself).
        print(f"{self.address_string()} {fmt % args}")


class _Server(socketserver.TCPServer):
    # Set on the class, not the instance: TCPServer binds inside __init__, so an
    # instance attribute would be assigned too late to have any effect. Without
    # it a redeploy can fail to bind while the old socket is in TIME_WAIT.
    allow_reuse_address = True


def main() -> None:
    handler = partial(_Handler, directory=DIRECTORY)
    with _Server(("0.0.0.0", PORT), handler) as httpd:
        print(f"kids-hub: serving {DIRECTORY} on port {PORT}")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
