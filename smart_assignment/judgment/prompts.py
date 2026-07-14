"""
Prompt construction for the grounded-judgment LLM call, kept separate from the
orchestration (`judge.py`) and the model plumbing (`llm.py`) so prompt iteration
doesn't touch logic.

The prompt hands the model the evidence packet as JSON and demands a
JSON-only reply in the shape `schema.parse_judgment` expects. The instructions
lean hard on grounding: cite every load-bearing fact, never invent a number,
and only ever pick from the feasible set.
"""

from __future__ import annotations

import json

from smart_assignment.judgment.evidence import NUMERIC_FACT_KEYS, EvidencePacket

_SYSTEM = """\
You are a Sysco delivery-slot assignment specialist. You are given a structured \
evidence packet describing one prospect's order and the delivery routes that \
were evaluated for it. The hard feasibility checks (service area, truck \
capacity) have ALREADY been applied deterministically: routes under \
"feasible_candidates" passed them, routes under "infeasible_candidates" failed \
and CANNOT be chosen for any reason.

Your job is to weigh the SOFT trade-offs among the feasible candidates -- how \
tightly the customer clusters with a route's existing stops (lower \
avg_stop_distance_miles is tighter), how much capacity headroom remains \
(utilization_after vs capacity_ceiling; remaining_capacity_after), and how well \
the route matches any preferred slot (window_overlap_minutes vs \
preferred_window_minutes) -- and decide whether to RECOMMEND the best feasible \
route for automatic assignment, or ESCALATE to a human specialist because the \
best available option is not clearly good enough to commit on its own.

You are NOT bound by any fixed weighting. facts.reference_weighted_score is the \
legacy formula's output, included for context only -- you may disagree with it. \
Judge each route on its own merits from the raw facts.

Return your confidence as HIGH only when you would be comfortable \
auto-assigning without a second opinion; use LOW when the best option is \
marginal, risky, or a close/uncertain call.
"""

_OUTPUT_CONTRACT = """\
Reply with a SINGLE JSON object and nothing else (no markdown fences, no prose \
before or after). Shape:

{{
  "candidate_notes": [{{"route_id": "<id>", "note": "<short reasoning>"}}],
  "recommended_route_id": "<a feasible route_id>" or null,
  "decision": "RECOMMEND" or "ESCALATE",
  "confidence": "HIGH" or "LOW",
  "rationale": "<2-4 sentence explanation an ops manager can act on>",
  "citations": [
    {{"kind": "fact", "route_id": "<id>", "field": "<fact key>", "value": <number>}},
    {{"kind": "comparison", "field": "<fact key>", "route_id_a": "<id>",
      "route_id_b": "<id>", "relation": "greater"|"less"|"equal"}}
  ]
}}

Grounding rules (STRICT):
- recommended_route_id MUST be one of the feasible_candidates' route_id values, \
or null. NEVER name an infeasible route.
- If decision is RECOMMEND, recommended_route_id must be non-null, and at least \
one citation must reference the recommended route on a route-specific fact -- \
back your pick with the facts it rests on.
- Every number, day, and time window you state in the rationale must appear \
verbatim in the evidence packet. Do NOT invent, round, or estimate figures -- \
and a figure YOU computed (a sum, difference, average, or projection over \
packet numbers) counts as invented: state the packet numbers you would have \
combined instead.
- Back every load-bearing fact with an entry in "citations". A "fact" citation \
must exactly match facts[field] for that route (a fraction like 0.87 may be \
written as 87%). A "comparison" citation must name two DIFFERENT routes and be \
arithmetically true of their facts -- do not cite a comparison unless the \
rationale actually relies on that exact pair of facts.
- Citable fact keys are exactly: {fact_keys}.
""".format(fact_keys=", ".join(NUMERIC_FACT_KEYS))


def build_judgment_prompt(packet: EvidencePacket) -> str:
    """The full prompt (system + evidence packet + output contract)."""
    packet_json = json.dumps(packet.as_dict(), indent=2, sort_keys=True)
    return f"{_SYSTEM}\n\nEVIDENCE PACKET:\n{packet_json}\n\n{_OUTPUT_CONTRACT}"


def build_retry_prompt(packet: EvidencePacket, feedback: str) -> str:
    """A corrective re-ask after a verification failure, naming what was wrong."""
    return (
        f"{build_judgment_prompt(packet)}\n\n"
        f"YOUR PREVIOUS REPLY FAILED VERIFICATION for these reasons:\n{feedback}\n"
        f"Return a corrected JSON object that fixes every issue above. Cite only "
        f"facts that appear verbatim in the evidence packet, and pick only a "
        f"feasible route."
    )
