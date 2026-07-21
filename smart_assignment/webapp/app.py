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
* ``GET  /frontend``     — a read-only sales-consultant "Choose a delivery slot"
  view of the slots the chat last produced for this browser session (the GitHub
  Pages Frontend tab, fed live). ``GET /api/frontend`` returns its data.

Every recommendation goes through ``run_slot_recommendation`` with the
deterministic reasoner and is rendered by ``build_workflow_payload`` — the exact
same functions behind ``scripts/run_local.py`` and the static page — so the live
UI can never drift from the published output.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load .env before any smart_assignment import below, so a backend choice or
# credentials set there are in os.environ before Config.from_env() resolves
# DEFAULT_CONFIG at import time. Load the repo-root .env by an ABSOLUTE,
# package-relative path first (so it's found no matter which directory the app
# was launched from -- a CWD-only search silently misses it and the agent then
# runs with no credentials), then fall back to a CWD search.
load_dotenv(Path(__file__).resolve().parents[2] / ".env")
load_dotenv()

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from smart_assignment.mock_customers import SAMPLE_CUSTOMERS
from smart_assignment.pipeline import run_slot_recommendation
from smart_assignment.reasoning import DeterministicReasoner
from smart_assignment.reporting.page import _FE_STYLE, _STYLE, build_workflow_payload
from smart_assignment.shared.config import DEFAULT_CONFIG
from smart_assignment.webapp.deterministic_chat import DeterministicChatService
from smart_assignment.webapp.llm_chat import LlmChatService, resolve_mode
from smart_assignment.webapp.parse import describe_slot, parse_intake

logger = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="Smart Assignment — live agent visualization",
    description="Chat with the delivery-slot agent and watch its workflow run, step by step.",
)

# One conversational service for the process; sessions are keyed per browser by
# a client-supplied session_id. Only builds an ADK Runner when llm mode is
# actually used (lazy), so the deterministic path stays import-light and offline.
_chat_service = LlmChatService()
# The offline brain: session-aware, so the chat stays conversational (remembers
# the address, accepts revisions like "try 20 cases") when the LLM path isn't
# available -- either because deterministic mode is configured or the LLM errored.
_det_chat_service = DeterministicChatService()

# The latest workflow payload per browser session, so the read-only customer
# view (`GET /frontend`) can show the delivery-slot options the chat just
# produced -- mirroring production, where a prospect flows Salesforce ->
# Smart Assignment -> the sales consultant's "Choose a delivery slot" view. This
# is a display cache only: the payload already came from the audited pipeline via
# the chat stream, so nothing here re-decides anything. Bounded (LRU) so a long-
# lived process can't grow it without limit.
_RESULT_CACHE_MAX = 256
_last_result: "OrderedDict[str, dict]" = OrderedDict()


def _remember_result(session_id: str, payload: Optional[dict]) -> None:
    """Cache the latest visualization payload for a session (most-recent last)."""
    if not payload:
        return
    _last_result[session_id] = payload
    _last_result.move_to_end(session_id)
    while len(_last_result) > _RESULT_CACHE_MAX:
        _last_result.popitem(last=False)


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


@app.post("/api/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    """Stream one conversational turn as Server-Sent Events.

    In ``llm`` mode this drives the real ADK agent; on any failure (no
    credentials, model/network error) it degrades to the **session-aware**
    deterministic brain so the chat stays conversational and never dead-ends. In
    ``deterministic`` mode it uses that same session-aware brain directly (no
    wasted ADK-runner build). Either way the chat remembers the conversation and
    accepts revisions like "try 20 cases" — matching ``adk web``."""

    def _emit(frame: dict) -> str:
        # Remember the result payload as it streams by, so the customer view can
        # render the same slots this turn produced. Display-only, no re-decision.
        if frame.get("type") == "visualization":
            _remember_result(req.session_id, frame.get("payload"))
        return f"data: {json.dumps(frame)}\n\n"

    async def event_stream():
        mode = resolve_mode(DEFAULT_CONFIG).get("mode")
        if mode == "llm":
            try:
                async for frame in _chat_service.stream_turn(req.session_id, req.message):
                    yield _emit(frame)
                return
            except Exception as exc:  # noqa: BLE001 - any LLM/runtime failure -> deterministic
                # Don't swallow it: log the full traceback so the real cause
                # (missing credentials, model/network error, ADK mismatch) is
                # diagnosable, and tell the user this is the deterministic
                # fallback -- not the agent -- so a silent downgrade never again
                # masquerades as "adk web parity".
                logger.exception(
                    "LLM chat turn failed for session %s; falling back to the "
                    "deterministic brain.",
                    req.session_id,
                )
                notice = (
                    f"⚠️ The conversational agent errored on this turn "
                    f"({type(exc).__name__}: {exc}). Showing a deterministic result "
                    f"instead. Check the server logs and the LLM backend/credentials "
                    f"(SMART_ASSIGNMENT_LLM_BACKEND)."
                )
                yield f"data: {json.dumps({'type': 'message', 'text': notice})}\n\n"
        async for frame in _det_chat_service.stream_turn(req.session_id, req.message):
            yield _emit(frame)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/api/frontend")
def frontend_result(session: str = "") -> dict:
    """The sales-consultant "Choose a delivery slot" view for a session's latest
    chat run. Returns ``{ok: false}`` until that session has produced a result, so
    the customer page can show an empty state and then the real options."""
    payload = _last_result.get(session)
    if not payload or not payload.get("frontendHtml"):
        return {"ok": False}
    return {
        "ok": True,
        "name": payload.get("name", ""),
        "address": payload.get("address", ""),
        "frontendHtml": payload["frontendHtml"],
    }


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


def _render_frontend() -> str:
    """Build the read-only customer view, injecting the Simulator's shared CSS
    plus the Frontend tab's own styles (``_FE_STYLE``) so the "Choose a delivery
    slot" cards render exactly like the published GitHub Pages Frontend tab."""
    template = (_STATIC_DIR / "frontend.html").read_text(encoding="utf-8")
    html = template.replace("/*__SHARED_STYLE__*/", _STYLE + _FE_STYLE)
    for asset in ("frontend.css", "frontend.js"):
        html = html.replace(f"/static/{asset}", f"/static/{asset}?v={_asset_version(asset)}")
    return html


@app.get("/frontend", response_class=HTMLResponse)
def frontend_page() -> HTMLResponse:
    """The sales-consultant "Choose a delivery slot" view — a read-only display of
    the slots the chat last produced for this browser session. The chat page
    (``/``) stays the primary, unchanged experience."""
    return HTMLResponse(_render_frontend(), headers={"Cache-Control": "no-cache"})


# Serve the static assets (app.css, app.js) under /static.
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
