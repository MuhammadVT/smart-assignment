"""
Plain-language narration for the live workflow steps.

While the agent runs, the chat shows one breadcrumb per pipeline tool call so the
user can follow *what it is doing right now*. This module owns the wording for
those breadcrumbs -- the short step label (the row heading) and a one-line,
plain-language description of what that step does.

These strings are deliberately **descriptive signposts, not data**: the verified
numbers, scores, proximity map, and slot timeline all render in the step cards
*below* the chat once the run finishes (see ``reporting.page._sim_steps``). So
nothing here is an actionable value or a computed figure that could drift from,
or duplicate, the audited result -- keeping the live view light and the evidence
in one authoritative place.

The one exception is the Intake line, which echoes the caller's *own* stated
inputs (order size, preferred day) back as a grounded confirmation when they are
present in the tool-call arguments. That is a read-back of what the user said,
never an invented value; it falls back to the generic description otherwise.

Centralising the copy here (rather than inline at the streaming site) keeps it
easy to find and revise, and lets every caller share the exact same phrasing.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

# Pipeline tool name -> short step label shown as the breadcrumb's heading.
STEP_LABELS = {
    "intake_customer": "Intake",
    "find_candidate_routes": "Geo-Lookup",
    "evaluate_and_score_routes": "Score & Rank",
    "recommend_or_escalate": "Recommend / Decide",
}

# Pipeline tool name -> one-line, plain-language description of what it does.
_STEP_DETAIL = {
    "intake_customer": "Reading the address, order size, and preferred window.",
    "find_candidate_routes": "Placing the address on the map and finding the nearest routes.",
    "evaluate_and_score_routes": "Scoring each open slot on distance, capacity, and timing.",
    "recommend_or_escalate": "Checking the top slot against the auto-assign bar and deciding.",
}


def step_label(tool_name: str) -> Optional[str]:
    """The breadcrumb heading for a pipeline tool, or None if it isn't a step."""
    return STEP_LABELS.get(tool_name)


def step_detail(tool_name: str, args: Optional[Mapping[str, Any]] = None) -> Optional[str]:
    """A plain-language line describing what the step is doing.

    For Intake, echo the caller's own stated inputs (order size, preferred day)
    when present -- a grounded read-back, never an invented value -- otherwise
    fall back to the generic description. Returns None for tools that aren't
    pipeline steps.
    """
    if tool_name == "intake_customer" and args:
        grounded = _intake_readback(args)
        if grounded:
            return grounded
    return _STEP_DETAIL.get(tool_name)


def _intake_readback(args: Mapping[str, Any]) -> Optional[str]:
    """Confirm the customer's own intake inputs back to them, if this call
    carried any. Returns None when the call has nothing worth echoing (e.g. an
    address-only first call), so the caller uses the generic description."""
    bits: list[str] = []

    cases = args.get("order_quantity_cases")
    if isinstance(cases, (int, float)) and not isinstance(cases, bool) and cases > 0:
        n = int(cases) if float(cases).is_integer() else cases
        bits.append(f"{n} cases")

    day = args.get("preferred_day")
    if isinstance(day, str) and day.strip():
        bits.append(f"prefers {day.strip().upper()}")

    if not bits:
        return None
    return "Reading your order — " + ", ".join(bits) + "."
