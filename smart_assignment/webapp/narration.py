"""
Plain-language narration for the live workflow steps.

While the agent runs, the chat shows one breadcrumb per pipeline STEP so the user
can follow *what it is doing right now*. This module owns the wording for those
breadcrumbs -- the short step label (the row heading), a one-line, plain-language
description of what that step does, and the declared step list behind each tool.

It owns the WORDING only. Whether a step actually ran is decided elsewhere, from
the tool's own result (``webapp/llm_chat``), so nothing here can imply that work
happened.

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
    # The handoff phase (see HANDOFF_STEPS): only reached on an escalation.
    "escalation_triage": "Briefing a specialist",
}

# Pipeline tool name -> one-line, plain-language description of what it does.
_STEP_DETAIL = {
    "intake_customer": "Reading the address, order size, and preferred window.",
    "find_candidate_routes": "Placing the address on the map and finding the nearest routes.",
    "evaluate_and_score_routes": "Scoring each open slot on distance, capacity, and timing.",
    "recommend_or_escalate": "Checking the top slot against the auto-assign bar and deciding.",
    "escalation_triage": "Summarizing why this needs a human, and what the options are.",
}

# What the decision step says once the tool has reported an escalation. This is a
# restatement of a real field on the tool's own result (``requires_human_review``),
# never an invented cause -- the *reason* for the escalation is the audited brief's
# job, not a breadcrumb's.
ESCALATION_DETAIL = "Escalating for human review."

# Steps belonging to the HANDOFF phase rather than the assignment pipeline. The
# assignment steps answer "which route and slot?"; these are the agent changing
# hands to a person, so the UI styles them apart (see step_phase). Declared here,
# beside the wording they belong to, rather than hardcoded at the streaming site.
HANDOFF_STEPS = frozenset({"escalation_triage"})
_PHASE_HANDOFF = "handoff"

# Each pipeline tool -> the ordered pipeline STEPS it may execute internally, where
# each step is named by the tool that canonically represents it (so it maps through
# STEP_LABELS / step_detail above with no new label copy). This decouples WHICH
# steps the breadcrumbs show from how many tools the agent actually called:
# recommend_or_escalate re-derives candidates, scores, and decides internally, so it
# lights up Geo-Lookup + Score & Rank + Recommend/Decide even as a single tool call.
# Callers dedupe across a turn, so a step already shown (e.g. Geo-Lookup from an
# on-demand find_candidate_routes) is not repeated.
#
# This is the tool's DECLARED step list, not a record of what ran. It says only
# which steps to put on screen; whether one succeeded is never read from here --
# the caller opens them as "running" on the tool call and settles them from the
# tool's own result (see webapp/llm_chat._tool_outcome). Keeping the two apart is
# the point: a static table can't know that a geocode failed, and a breadcrumb
# claiming a step finished is a claim about the audited run.
TOOL_STEPS = {
    "intake_customer": ["intake_customer"],
    "find_candidate_routes": ["find_candidate_routes"],
    "evaluate_and_score_routes": ["find_candidate_routes", "evaluate_and_score_routes"],
    "recommend_or_escalate": [
        "find_candidate_routes",
        "evaluate_and_score_routes",
        "recommend_or_escalate",
    ],
    # The consolidated batch tool runs the whole pipeline in one call.
    "assign_prospect": [
        "intake_customer",
        "find_candidate_routes",
        "evaluate_and_score_routes",
        "recommend_or_escalate",
    ],
    # The escalation-triage sub-agent composing the specialist brief. It is the
    # longest single call in an escalation turn, and without a step here the
    # stepper sits fully ticked while it runs. Present only when the agent
    # actually escalates AND Config.use_escalation_triage is on -- breadcrumbs
    # follow real tool calls, so nothing needs flag-gating here.
    "escalation_triage": ["escalation_triage"],
}


def step_phase(step_name: str) -> Optional[str]:
    """``"handoff"`` for a step that hands the prospect to a person, else None.

    Lets the display surface style the handoff apart from the assignment steps
    without knowing which step names those are."""
    return _PHASE_HANDOFF if step_name in HANDOFF_STEPS else None


def tool_steps(tool_name: str) -> list[str]:
    """The ordered pipeline steps a tool may execute internally (each named by the
    tool that canonically represents it -- pass each to step_label / step_detail).
    Empty for a tool that isn't a pipeline step.

    The live stepper shows one breadcrumb per step, so a single consolidated call
    (recommend_or_escalate, assign_prospect) still surfaces every underlying step --
    which steps appear is decoupled from the tool count. Their outcome is NOT: this
    returns the declared step list, never a claim that any of it ran."""
    return TOOL_STEPS.get(tool_name, [])


def step_label(tool_name: str) -> Optional[str]:
    """The breadcrumb heading for a pipeline step, or None if it isn't a step."""
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
