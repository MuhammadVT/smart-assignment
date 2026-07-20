"""Golden eval cases for the conversational agent, built from the repo's own
deterministic mock fixtures (``smart_assignment.mock_customers``).

Each case pairs a natural-language intake message with the customer facts it
encodes and the *expected tool trajectory* -- the ordered pipeline the agent
must drive: ``intake_customer`` -> ``find_candidate_routes`` ->
``evaluate_and_score_routes`` -> ``recommend_or_escalate``. Only ``intake_customer``
takes arguments; its expected args are the KNOWN ground-truth fields of the mock
customer (not invented), so the trajectory expectation is real, not a guess.

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
from typing import Any, Dict, List, Tuple

from smart_assignment.mock_customers import SAMPLE_CUSTOMERS
from smart_assignment.shared.models import CustomerProfile

# The fixed deterministic pipeline the agent drives on every intake. These three
# tools take no arguments (they read accumulated session state), so their
# expected calls carry empty args; ``intake_customer`` is handled separately
# because its args are the customer's known fields.
_PIPELINE_AFTER_INTAKE: Tuple[str, ...] = (
    "find_candidate_routes",
    "evaluate_and_score_routes",
    "recommend_or_escalate",
)


@dataclass(frozen=True)
class GoldenCase:
    """One eval case: a user message + the fixture it encodes + the expected
    outcome (for the 2b capture) + the note explaining what branch it exercises."""

    eval_id: str
    query: str
    customer: CustomerProfile
    expected_outcome: str  # "recommend" | "escalate" -- narrative target for 2b
    note: str


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
# four mock customers were chosen to exercise the full outcome range (see
# mock_customers.py): two clean recommends, two escalations (low score, and
# out-of-range/over-capacity).
GOLDEN_CASES: List[GoldenCase] = [
    GoldenCase(
        eval_id="bayou_city_bistro_recommend",
        query=(
            "New prospect Bayou City Bistro at 1200 McKinney St, Houston, TX 77010. "
            "About 90 cases a week. They'd like Tuesday mornings, 7 to 10am."
        ),
        customer=_by_name("Bayou City Bistro"),
        expected_outcome="recommend",
        note="Downtown, modest order, in the dense Central Houston route -> clean recommend.",
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
