"""Prompt construction for the grounded route-slot decision."""

from __future__ import annotations

import json

from smart_assignment.routeslot.evidence import NUMERIC_FACT_KEYS, RouteSlotPacket

_SYSTEM = """\
You are a Sysco delivery planner. Each option below is a specific (route, TIME \
SLOT) pairing the prospect could be assigned to -- the SAME route can appear more \
than once with different slots. Your job is to pick the single best route-slot \
overall. Reason from each option's own facts:
 - geographic_clustering: how tightly the ROUTE fits the prospect's neighborhood \
(higher = closer to the route's existing stops);
 - capacity_buffer: how safely the ROUTE stays under its truck capacity (higher = \
more headroom);
 - window_match: how much THIS slot covers the customer's stated preferred time \
(higher honors the preference more; absent when no preference was stated);
 - slot_availability: how OPEN this slot is -- 1.0 means no committed customer \
shares it; lower means valued customers (esp. tier 5 / Perks / 4) already hold it, \
and adding the prospect would crowd them.

The key trade-off: a route that is great on clustering and capacity is NOT the \
best choice if its only workable slot is densely shared by high-tier customers. A \
slightly less clustered/full route with a genuinely open slot can be the better \
overall assignment. Weigh route quality AND slot openness together.

`reference_weighted_score` is what a fixed weighted heuristic scores each option, \
and `deterministic_choice_index` is the option it would pick on its own. Treat \
that as a strong default: agree with it unless the other facts clearly justify a \
different route-slot, and if you diverge, say why in your rationale.
"""

_OUTPUT_CONTRACT = """\
Reply with a SINGLE JSON object and nothing else (no markdown). Shape:

{{
  "chosen_index": <the index of the route-slot you pick>,
  "rationale": "<1-2 sentences an ops manager can act on>",
  "citations": [
    {{"index": <option index>, "field": "<fact key>", "value": <number>}}
  ]
}}

Rules (STRICT):
- chosen_index MUST be one of the enumerated option indices. NEVER invent a route,
  a slot, or a time.
- Back every figure in your rationale with a citation whose value exactly matches
  that option's fact.
- Citable fact keys are exactly: {fields}.
""".format(fields=", ".join(NUMERIC_FACT_KEYS))


def build_route_slot_prompt(packet: RouteSlotPacket) -> str:
    body = json.dumps(packet.as_dict(), indent=2, sort_keys=True)
    return f"{_SYSTEM}\n\nROUTE-SLOT OPTIONS:\n{body}\n\n{_OUTPUT_CONTRACT}"


def build_route_slot_retry_prompt(packet: RouteSlotPacket, feedback: str) -> str:
    return (
        f"{build_route_slot_prompt(packet)}\n\n"
        f"YOUR PREVIOUS REPLY FAILED VERIFICATION for these reasons:\n{feedback}\n"
        f"Return a corrected JSON object. Pick only an enumerated option index and "
        f"cite only facts that appear verbatim on that option."
    )
