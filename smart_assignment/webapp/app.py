"""
FastAPI app: a chat interface that visualizes the Smart Assignment workflow
live, the same way the published GitHub Pages Simulator does — but on any input.

Run it (Phase 1 is fully offline, no API key needed):

    pip install -e ".[web]"
    python3 scripts/run_web.py            # http://127.0.0.1:8000 (offline-ready)

or with uvicorn directly (the package imports offline with no credentials):

    uvicorn smart_assignment.webapp.app:app --reload

Endpoints
---------
* ``GET  /``             — the chat page (static HTML/CSS/JS).
* ``POST /api/recommend``— parse a chat message into intake fields, run the real
  pipeline, and return the Simulator visualization payload (the 5 step cards +
  the result card) or a clarifying question when required fields are missing.
* ``GET  /api/samples``  — the bundled sample prospects, as ready-to-send chat
  messages (parity with the Simulator's sample chips).

Every recommendation goes through ``run_slot_recommendation`` with the
deterministic reasoner and is rendered by ``build_workflow_payload`` — the exact
same functions behind ``scripts/run_local.py`` and the static page — so the live
UI can never drift from the published output.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load .env (if present) before any smart_assignment import below, so a backend
# choice or credentials set there are in os.environ before Config.from_env()
# resolves DEFAULT_CONFIG at import time. Covers running this module directly
# (e.g. `uvicorn smart_assignment.webapp.app:app`) without scripts/run_web.py.
load_dotenv()

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from smart_assignment.mock_customers import SAMPLE_CUSTOMERS
from smart_assignment.pipeline import run_slot_recommendation
from smart_assignment.reasoning import DeterministicReasoner
from smart_assignment.reporting.page import _STYLE, build_workflow_payload
from smart_assignment.shared.config import DEFAULT_CONFIG
from smart_assignment.webapp.llm_chat import LlmChatService, resolve_mode
from smart_assignment.webapp.parse import describe_slot, parse_intake

_STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="Smart Assignment — live agent visualization",
    description="Chat with the delivery-slot agent and watch its workflow run, step by step.",
)

# One conversational service for the process; sessions are keyed per browser by
# a client-supplied session_id. Only builds an ADK Runner when llm mode is
# actually used (lazy), so the deterministic path stays import-light and offline.
_chat_service = LlmChatService()


class RecommendRequest(BaseModel):
    """A chat turn. ``message`` is free text like
    '1200 McKinney St, Houston, TX 77010, 90 cases, TUE 07:00-10:00'."""

    message: str


class RecommendResponse(BaseModel):
    ok: bool
    # Populated on success: {name, address, steps[], resultHtml} — the Simulator payload.
    payload: Optional[dict] = None
    # A short agent line to show in the chat transcript (confirmation or question).
    reply: Optional[str] = None
    # True when we could not run yet and are asking the user for more detail.
    needs_input: bool = False


def _sample_message(customer) -> str:
    """Compose a ready-to-send chat message from a bundled sample prospect."""
    parts = [customer.address, f"{customer.order_quantity_cases} cases"]
    if customer.preferred_slot is not None:
        parts.append(describe_slot(customer.preferred_slot))
    return ", ".join(parts)


@app.get("/api/samples")
def samples() -> list[dict]:
    """Bundled sample prospects as clickable chips (name + prefilled message)."""
    return [{"name": c.name, "message": _sample_message(c)} for c in SAMPLE_CUSTOMERS]


@app.post("/api/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest) -> RecommendResponse:
    """Parse the message, run the workflow, and return the visualization payload."""
    parsed = parse_intake(req.message)
    if parsed.profile is None:
        return RecommendResponse(ok=False, reply=parsed.clarify, needs_input=True)

    try:
        result = run_slot_recommendation(
            parsed.profile,
            config=DEFAULT_CONFIG,
            reasoner=DeterministicReasoner(),
        )
    except ValueError as exc:
        # Intake rejected the profile (e.g. malformed customer number). Surface
        # the reason as an agent reply instead of a 500.
        return RecommendResponse(ok=False, reply=str(exc), needs_input=True)

    payload = build_workflow_payload(result, DEFAULT_CONFIG)
    slot_phrase = describe_slot(parsed.preferred_slot)
    reply = (
        f"Running the workflow for {parsed.address} — "
        f"{parsed.order_quantity_cases} cases, preferred slot: {slot_phrase}."
    )
    return RecommendResponse(ok=True, payload=payload, reply=reply)


class ChatRequest(BaseModel):
    """A conversational turn. ``session_id`` is a stable per-browser id so the
    ADK agent keeps context across turns."""

    session_id: str
    message: str


@app.get("/api/mode")
def mode() -> dict:
    """Which brain the app will actually serve — 'llm' (Phase 2) or
    'deterministic' (Phase 1) — plus why, so the client picks the right
    endpoint and can surface any downgrade notice."""
    return resolve_mode(DEFAULT_CONFIG)


def _fallback_frames(message: str):
    """Deterministic frames for when the LLM path is unavailable or errors mid
    turn — parse the message and either ask for missing info or visualize."""
    parsed = parse_intake(message)
    if parsed.profile is None:
        yield {"type": "message", "text": parsed.clarify}
        yield {"type": "done"}
        return
    try:
        result = run_slot_recommendation(
            parsed.profile, config=DEFAULT_CONFIG, reasoner=DeterministicReasoner()
        )
    except ValueError as exc:
        yield {"type": "message", "text": str(exc)}
        yield {"type": "done"}
        return
    yield {"type": "visualization", "payload": build_workflow_payload(result, DEFAULT_CONFIG)}
    yield {"type": "done"}


@app.post("/api/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    """Stream one conversational turn of the real ADK agent as Server-Sent
    Events. On any failure (no credentials, model/network error) it degrades to
    a deterministic result so the chat never dead-ends."""

    async def event_stream():
        try:
            async for frame in _chat_service.stream_turn(req.session_id, req.message):
                yield f"data: {json.dumps(frame)}\n\n"
        except Exception:  # noqa: BLE001 - any LLM/runtime failure -> deterministic
            for frame in _fallback_frames(req.message):
                yield f"data: {json.dumps(frame)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _asset_version(name: str) -> str:
    """Short content hash of a static asset, for cache-busting its URL. When the
    file changes the hash changes, so the browser is forced to refetch instead of
    serving a stale cached copy (the CSS/JS have no Cache-Control, so browsers
    otherwise heuristic-cache them and updates don't show up until a hard refresh)."""
    try:
        return hashlib.md5((_STATIC_DIR / name).read_bytes()).hexdigest()[:10]
    except OSError:
        return "0"


def _render_index() -> str:
    """Build the chat page, injecting the Simulator's own CSS (``_STYLE``) so the
    live step/result cards look exactly like the published page and can't drift.
    App CSS/JS get a ?v=<hash> query so a new build always busts the browser cache."""
    template = (_STATIC_DIR / "index.html").read_text(encoding="utf-8")
    html = template.replace("/*__SHARED_STYLE__*/", _STYLE)
    for asset in ("app.css", "app.js"):
        html = html.replace(f"/static/{asset}", f"/static/{asset}?v={_asset_version(asset)}")
    return html


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    # no-cache so the browser revalidates the HTML each load and always sees the
    # current ?v=<hash> asset URLs (otherwise a cached page keeps the old URLs).
    return HTMLResponse(_render_index(), headers={"Cache-Control": "no-cache"})


# Serve the static assets (app.css, app.js) under /static.
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
