"""Golden eval cases for the conversational agent, built from the repo's own
deterministic mock fixtures (``smart_assignment.mock_customers``).

Each case pairs a natural-language intake message with the customer facts it
encodes and the *expected tool trajectory* -- the ordered pipeline the agent must
drive: ``intake_customer`` -> ``recommend_or_escalate``. Only ``intake_customer``
takes arguments; its expected args are the KNOWN ground-truth fields of the mock
customer (not invented), so the trajectory expectation is real, not a guess.
See ``_PIPELINE_AFTER_INTAKE`` below for which tools are deliberately left
unpinned and why.

What is deliberately NOT encoded here is the agent's final natural-language
response: that is the LLM's narration, which can only be captured faithfully by
running a real backend (see ``eval/capture.py``, Phase 2b).
Until then the dataset scores trajectory only (see ``eval/data/test_config.json``),
which catches the structural regressions -- a dropped/reordered tool, the
address-resolution branch firing when it shouldn't -- without asserting text we
cannot yet generate. ``expected_outcome`` is documentation for the 2b capture
(which cases should recommend vs. escalate), not scored in 2a.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from smart_assignment.mock_customers import SAMPLE_CUSTOMERS
from smart_assignment.shared.models import CustomerProfile

# The decision step the agent must reach on every successful intake. It takes no
# arguments (it reads accumulated session state), so its expected call carries
# empty args; ``intake_customer`` is handled separately because its args are the
# customer's known fields.
#
# Only what the agent is REQUIRED to do belongs here. Two other tool families are
# deliberately absent, for different reasons:
#
# * ``find_candidate_routes`` / ``evaluate_and_score_routes`` are OPTIONAL,
#   on-demand tools -- ``recommend_or_escalate`` geocodes, constraint-checks and
#   scores internally, so the default flow goes straight from intake to the
#   decision and calls neither (see smart_assignment/prompts.py, "Workflow"). They
#   were pinned here until the flow was optimized to skip them, which silently
#   made every case score 0.0; don't re-add them, or the eval starts asserting a
#   path the prompt explicitly tells the agent not to take.
# * On an ESCALATE case the agent additionally hands off to a human --
#   ``escalation_triage`` (when Config.use_escalation_triage is on, the default)
#   and/or ADK's ``adk_request_input``. Their only arguments are model-authored
#   prose (the triage `request`, the handoff `message`) which differs every run,
#   and the trajectory metric compares args exactly, so pinning them would make
#   the eval permanently flaky.
#
# eval/data/test_config.json therefore scores this metric with
# ``match_type: IN_ORDER``: every tool below must appear, in this order, with
# exactly these args, while any additional call -- an on-demand scoring tool the
# user asked for, or an escalation handoff -- is tolerated anywhere in the
# trajectory. Don't "tighten" that back to the EXACT default: it fails the two
# escalate cases, and any user-prompted on-demand call too.
_PIPELINE_AFTER_INTAKE: Tuple[str, ...] = ("recommend_or_escalate",)


@dataclass(frozen=True)
class GoldenCase:
    """One eval case: a user message + the fixture it encodes + the expected
    outcome (for the 2b capture) + the note explaining what branch it exercises."""

    eval_id: str
    query: str
    customer: CustomerProfile
    expected_outcome: str  # "recommend" | "escalate" -- narrative target for 2b
    note: str
    # The production decision this case was curated FROM, when it was curated at
    # all (see eval/case_source.candidate_to_case). It is the id a human's
    # feedback on that same decision carries, so it is what a judge verdict and a
    # human label join on in eval/judge_calibration.py -- without it the link
    # survives only as an 8-char prefix inside the minted eval_id. ``None`` for
    # the hand-written fixtures below: no human ever labeled them, so there is
    # nothing to join to.
    decision_id: Optional[str] = None


def intake_args(customer: CustomerProfile) -> Dict[str, Any]:
    """The ground-truth ``intake_customer`` arguments for a customer -- exactly the
    fields the agent should extract from the message, derived from the fixture so
    the expectation is real rather than invented."""
    args: Dict[str, Any] = {
        "address": customer.address,
        "order_quantity_cases": customer.order_quantity_cases,
    }
    if customer.name:
        args["name"] = customer.name
    slot = customer.preferred_slot
    if slot is not None:
        args["preferred_day"] = slot.day.name
        args["preferred_window_start"] = slot.window[0].strftime("%H:%M")
        args["preferred_window_end"] = slot.window[1].strftime("%H:%M")
    return args


def expected_trajectory(case: GoldenCase) -> List[Tuple[str, Dict[str, Any]]]:
    """The ordered ``(tool_name, args)`` sequence the agent must produce for a
    clean single-pass intake."""
    trajectory: List[Tuple[str, Dict[str, Any]]] = [
        ("intake_customer", intake_args(case.customer)),
    ]
    trajectory.extend((name, {}) for name in _PIPELINE_AFTER_INTAKE)
    return trajectory


def _by_name(name: str) -> CustomerProfile:
    for customer in SAMPLE_CUSTOMERS:
        if customer.name == name:
            return customer
    raise KeyError(f"No mock customer named {name!r}")  # pragma: no cover


# Natural-language intake messages authored to encode each fixture's facts. The
# four mock customers were chosen to exercise the full outcome range under the
# built-in mock routes (see mock_customers.py): two clean recommends, two
# escalations (low score, and out-of-range/over-capacity).
#
# ``expected_outcome``/``note`` describe that MOCK-data design intent, not a
# guarantee: this repo's default data source is "cache" (see
# integrations/route_capacity_client.py), so whenever a local data/dev/*.parquet
# cache snapshot is present, trajectory-scored fields (agent's tool calls) are
# unaffected, but the outcome (recommend vs. escalate) and Phase 2b's captured
# final_response instead reflect THAT real capacity snapshot at capture time --
# which can legitimately differ from the mock-data design intent below, and will
# drift as real capacity changes. bayou_city_bistro_recommend is a live example:
# under real cache data captured so far it escalates (see
# eval/data/captured_responses.json), not the clean recommend the mock data
# gives. Not a bug -- re-run eval.capture (see eval/README.md) to refresh
# captured responses against current real data.
GOLDEN_CASES: List[GoldenCase] = [
    GoldenCase(
        eval_id="bayou_city_bistro_recommend",
        query=(
            "New prospect Bayou City Bistro at 1200 McKinney St, Houston, TX 77010. "
            "About 90 cases a week. They'd like Tuesday mornings, 7 to 10am."
        ),
        customer=_by_name("Bayou City Bistro"),
        expected_outcome="recommend",
        note=(
            "Downtown, modest order, in the dense Central Houston route -> clean "
            "recommend on the MOCK routes. Under real cache data this can escalate "
            "instead (route capacity is a real, current fact, not scripted) -- see "
            "the module docstring above."
        ),
    ),
    GoldenCase(
        eval_id="galleria_grill_escalate_low_score",
        query=(
            "Set up Galleria Grill & Catering, 5085 Westheimer Rd, Houston, TX 77056. "
            "Large catering account, 400 cases. No particular day or time preference."
        ),
        customer=_by_name("Galleria Grill & Catering"),
        expected_outcome="escalate",
        note="Big order; only one nearby route can take it and it lands full -> escalate.",
    ),
    GoldenCase(
        eval_id="katy_prairie_escalate_out_of_range",
        query=(
            "New customer Katy Prairie Steakhouse, 5000 Katy Mills Cir, Katy, TX 77494. "
            "260 cases. Prefers Tuesday early morning, 6 to 8am."
        ),
        customer=_by_name("Katy Prairie Steakhouse"),
        expected_outcome="escalate",
        note="Far-west; nearest routes out of range or over capacity -> escalate.",
    ),
    GoldenCase(
        eval_id="woodlands_fresh_cafe_recommend",
        query=(
            "Onboard Woodlands Fresh Cafe at 1201 Lake Woodlands Dr, The Woodlands, TX 77380. "
            "150 cases. Thursday late morning, 9am to noon works."
        ),
        customer=_by_name("Woodlands Fresh Cafe"),
        expected_outcome="recommend",
        note="Lightly-booked North route fits well -> clean recommend.",
    ),
]
