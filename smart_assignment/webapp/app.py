"""
FastAPI app: a chat interface that visualizes the Smart Assignment workflow
live, the same way the published GitHub Pages Simulator does — but on any input.

Run it (Phase 1 is fully offline, no API key needed):

    pip install -e ".[web]"
    python3 scripts/run_web.py            # http://127.0.0.1:8000 (offline-ready)

or with uvicorn directly (set the credential-free backend, same as run_local.py):

    SMART_ASSIGNMENT_LLM_BACKEND=standard uvicorn smart_assignment.webapp.app:app --reload

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

from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from smart_assignment.mock_customers import SAMPLE_CUSTOMERS
from smart_assignment.pipeline import run_slot_recommendation
from smart_assignment.reasoning import DeterministicReasoner
from smart_assignment.reporting.page import _STYLE, build_workflow_payload
from smart_assignment.shared.config import DEFAULT_CONFIG
from smart_assignment.webapp.parse import describe_slot, parse_intake

_STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="Smart Assignment — live agent visualization",
    description="Chat with the delivery-slot agent and watch its workflow run, step by step.",
)


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


def _render_index() -> str:
    """Build the chat page, injecting the Simulator's own CSS (``_STYLE``) so the
    live step/result cards look exactly like the published page and can't drift."""
    template = (_STATIC_DIR / "index.html").read_text(encoding="utf-8")
    return template.replace("/*__SHARED_STYLE__*/", _STYLE)


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(_render_index())


# Serve the static assets (app.css, app.js) under /static.
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
