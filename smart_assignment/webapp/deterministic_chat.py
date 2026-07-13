"""
Deterministic (Phase 1) conversational brain — session-aware, no LLM.

The stateless parser (:mod:`smart_assignment.webapp.parse`) reads one message in
isolation, so a follow-up like *"try 20 cases"* forgets the address you already
gave. That reads nothing like the real ADK agent (``adk web``), which remembers
the conversation and accepts revisions.

``DeterministicChatService`` fixes that offline: it keeps the intake fields it
has understood so far **per session** and merges each new message into them, so
the chat accumulates context and supports revisions turn-to-turn — the same
conversational feel as the agent, without needing any credentials. Once it has
an address and an order quantity it runs the exact same deterministic pipeline
and emits the identical ``build_workflow_payload`` visualization the rest of the
app uses, so nothing can drift.

It yields the same SSE frame dicts as :class:`~smart_assignment.webapp.llm_chat.
LlmChatService`, so the browser (and ``/api/chat``) treat both brains the same:

    {"type": "message", "text": ...}          — an agent reply / clarifying ask
    {"type": "visualization", "payload": ...}  — the 5 step cards + result
    {"type": "done"}                           — turn finished
"""

from __future__ import annotations

from typing import AsyncGenerator, Optional

from smart_assignment.pipeline import run_slot_recommendation
from smart_assignment.reasoning import DeterministicReasoner
from smart_assignment.reporting.page import build_workflow_payload
from smart_assignment.shared.config import DEFAULT_CONFIG
from smart_assignment.shared.geo import Geocoder
from smart_assignment.shared.models import CustomerProfile, PreferredSlot
from smart_assignment.webapp.parse import describe_slot, parse_intake

# Messages that clear the accumulated intake and start a fresh prospect.
_RESET_WORDS = {"reset", "restart", "start over", "clear", "new", "new prospect"}


def _clarify_for(missing: list[str]) -> str:
    """The question to ask for whatever intake fields are still missing (same
    wording as the one-shot parser, but driven by the accumulated state)."""
    if missing == ["address"]:
        return "What's the delivery address? (street, city, state, ZIP)"
    if missing == ["order quantity (in cases)"]:
        return "How many cases is the order?"
    return (
        "I need a delivery address and an order quantity in cases to run the "
        "workflow. For example: "
        "“1200 McKinney St, Houston, TX 77010, 90 cases, TUE 07:00-10:00”."
    )


class _SessionIntake:
    """The intake fields understood so far for one conversation."""

    __slots__ = ("address", "cases", "slot", "ran")

    def __init__(self) -> None:
        self.address: Optional[str] = None
        self.cases: Optional[int] = None
        self.slot: Optional[PreferredSlot] = None
        self.ran: bool = False

    def missing(self) -> list[str]:
        gaps: list[str] = []
        if not self.address:
            gaps.append("address")
        if self.cases is None:
            gaps.append("order quantity (in cases)")
        return gaps


class DeterministicChatService:
    """Session-aware deterministic chat. ``geocoder`` is injectable so tests can
    run fully offline; ``None`` uses the pipeline's default geocoder."""

    def __init__(self, geocoder: Optional[Geocoder] = None) -> None:
        self._geocoder = geocoder
        self._sessions: dict[str, _SessionIntake] = {}

    def _run(self, profile: CustomerProfile):
        kwargs = {"config": DEFAULT_CONFIG, "reasoner": DeterministicReasoner()}
        if self._geocoder is not None:
            kwargs["geocoder"] = self._geocoder
        return run_slot_recommendation(profile, **kwargs)

    async def stream_turn(self, session_id: str, message: str) -> AsyncGenerator[dict, None]:
        text = (message or "").strip()

        if text.lower() in _RESET_WORDS:
            self._sessions.pop(session_id, None)
            yield {
                "type": "message",
                "text": "Okay, starting fresh — tell me the prospect's address and order size.",
            }
            yield {"type": "done"}
            return

        st = self._sessions.setdefault(session_id, _SessionIntake())

        # Merge whatever this message added/updated into the running intake, so
        # the conversation accumulates and later turns can revise earlier ones.
        parsed = parse_intake(text)
        changed = False
        if parsed.address:
            st.address, changed = parsed.address, True
        if parsed.order_quantity_cases is not None:
            st.cases, changed = parsed.order_quantity_cases, True
        if parsed.preferred_slot is not None:
            st.slot, changed = parsed.preferred_slot, True

        missing = st.missing()
        if missing:
            yield {"type": "message", "text": _clarify_for(missing)}
            yield {"type": "done"}
            return

        # Complete. Re-run when this turn changed something (e.g. "try 20 cases")
        # or it's the first time we can run; otherwise just invite a revision so a
        # stray "thanks" doesn't re-animate the whole workflow.
        if not (changed or not st.ran):
            yield {
                "type": "message",
                "text": (
                    "Anything you'd like to change? (e.g. “try 20 cases”, a new "
                    "address, or a preferred day/time)"
                ),
            }
            yield {"type": "done"}
            return

        st.ran = True
        yield {
            "type": "message",
            "text": (
                f"Running the workflow for {st.address} — {st.cases} cases, "
                f"preferred slot: {describe_slot(st.slot)}."
            ),
        }
        profile = CustomerProfile(
            name="New prospect",
            address=st.address,
            order_quantity_cases=st.cases,
            preferred_slot=st.slot,
        )
        try:
            result = self._run(profile)
        except ValueError as exc:
            # Intake rejected the profile (e.g. couldn't geocode the address).
            st.ran = False
            yield {"type": "message", "text": str(exc)}
            yield {"type": "done"}
            return

        yield {"type": "visualization", "payload": build_workflow_payload(result, DEFAULT_CONFIG)}
        yield {"type": "done"}
