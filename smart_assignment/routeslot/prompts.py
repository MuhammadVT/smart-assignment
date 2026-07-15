"""Prompt construction for the grounded route-slot decision."""

from __future__ import annotations

import json

from smart_assignment.routeslot.evidence import NUMERIC_FACT_KEYS, RouteSlotPacket

# The sage generic agent is conversational: unless told forcefully, it narrates its
# reasoning as prose ("I have analyzed the options... I recommend...") or tries to
# call a tool, and either way the reply isn't the JSON this layer parses. This
# directive is placed FIRST and LAST in every prompt (primacy + recency) to pin the
# output to a bare JSON object with no prose and no tool call.
_JSON_ONLY = """\
CRITICAL OUTPUT FORMAT -- READ FIRST. Respond with ONE raw JSON object and NOTHING \
else: the first character of your reply MUST be `{` and the last MUST be `}`. Do not \
write any prose, preamble, greeting, analysis, or explanation before or after it, and \
do not wrap it in markdown fences. Do NOT call, invoke, or request any function, tool, \
or sub-agent -- put your ENTIRE answer inside the JSON object. Any text outside the \
single JSON object makes the reply unusable and it will be discarded.
"""

# Corrective note appended on a retry when the previous reply could not be parsed as
# JSON at all (prose or a tool call) -- distinct from a verification-failure retry.
_JSON_RETRY_NOTE = """\
YOUR PREVIOUS REPLY WAS UNUSABLE: it was not a single JSON object -- it contained \
prose, an explanation, or a tool/function call. Do not narrate and do not call any \
tool. Reply again with ONLY the JSON object specified above: first character `{`, \
last character `}`, nothing else.
"""

_SYSTEM = """\
You are a Sysco delivery planner. Each option below is a specific (route, TIME \
SLOT) pairing the prospect could be assigned to -- the SAME route can appear more \
than once with different slots. Your job is to pick the single best route-slot \
overall AND explain the reasoning an ops manager needs to trust the call. Reason \
from each option's own facts:
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
overall assignment. Weigh route quality AND slot openness together, and make that \
trade-off explicit in your answer -- name what the winner gives up and why that is \
acceptable.

`reference_weighted_score` is what a fixed weighted heuristic scores each option, \
and `deterministic_choice_index` is the option it would pick on its own. Treat \
that as a strong default: agree with it unless the other facts clearly justify a \
different route-slot, and if you diverge, say why.
"""

_OUTPUT_CONTRACT = """\
Reply with a SINGLE JSON object and nothing else (no markdown). Shape:

{{
  "chosen_index": <index of the route-slot you pick>,
  "decision_summary": "<one action line: assign <route> · <day> · <window>>",
  "primary_reasons": [
    "<geographic_clustering: your read of the neighborhood fit, WITH its number>",
    "<capacity_buffer: how much truck headroom is left, with its number>",
    "<window_match: how well the slot honors the stated preference, with its number \
-- INCLUDE this line only when the option has a window_match fact>",
    "<slot_availability: how open this slot is (who else holds it), with its number>"
  ],
  "key_tradeoff": "<what the winner gives up vs. the next-best option and why it is \
still the better overall pick -- reference BOTH options' numbers>",
  "runner_up": {{
    "index": <index of the next-best option>,
    "why_not": "<the specific fact that tips the pick away from it>"
  }},
  "vs_deterministic_default": {{
    "verdict": "AGREE" | "DIVERGE",
    "note": "<if DIVERGE, why the weighted default is wrong here; may be empty on AGREE>"
  }},
  "citations": [
    {{"index": <option index>, "field": "<fact key>", "value": <number>}}
  ]
}}

Rules (STRICT):
- chosen_index MUST be one of the enumerated option indices. NEVER invent a route,
  a slot, or a time.
- `primary_reasons` MUST comprehensively cover EVERY scored factor the chosen
  option carries -- geographic_clustering, capacity_buffer, slot_availability,
  and window_match (only when the option has a window_match fact) -- one short line
  each, in that order, each citing that factor's own value. Never drop
  slot_availability, and never drop window_match when a preference was stated.
- verdict is AGREE only when chosen_index == deterministic_choice_index, else DIVERGE
  (and then `note` must justify the divergence).
- When two or more options are offered, `key_tradeoff` and `runner_up` are REQUIRED
  and runner_up.index must be a real option other than your pick. With a single
  option, use key_tradeoff to say why it is the clear choice and set runner_up to null.
- EVERY number you state in any field must be a real fact from the option you
  attribute it to, and must appear in `citations`. Do not state a number you cannot
  cite -- and a figure YOU computed (a sum, difference, average, or projection over
  option facts) counts as a number you cannot cite: state the option's own numbers
  instead. Days and time windows must be quoted verbatim from the options.
- Whenever you name a route (in decision_summary, key_tradeoff, or anywhere),
  write it as "<route_id> - <route_name>" using that option's own route_id and
  route_name -- always both together, never one alone.
- Citable fact keys are exactly: {fields}.
""".format(fields=", ".join(NUMERIC_FACT_KEYS))


def build_route_slot_prompt(packet: RouteSlotPacket) -> str:
    body = json.dumps(packet.as_dict(), indent=2, sort_keys=True)
    return (
        f"{_JSON_ONLY}\n{_SYSTEM}\n\nROUTE-SLOT OPTIONS:\n{body}\n\n"
        f"{_OUTPUT_CONTRACT}\n\n{_JSON_ONLY}"
    )


def build_route_slot_retry_prompt(packet: RouteSlotPacket, feedback: str) -> str:
    return (
        f"{build_route_slot_prompt(packet)}\n\n"
        f"YOUR PREVIOUS REPLY FAILED VERIFICATION for these reasons:\n{feedback}\n"
        f"Return a corrected JSON object. Pick only an enumerated option index, keep the "
        f"verdict consistent with that pick, include the trade-off and runner-up when more "
        f"than one option exists, and cite every number you state."
    )


def build_route_slot_json_retry_prompt(packet: RouteSlotPacket) -> str:
    """Retry after a reply that could not be parsed as JSON (prose or a tool call)."""
    return f"{build_route_slot_prompt(packet)}\n\n{_JSON_RETRY_NOTE}"


# --- Grounded-escalation variant: the model also decides recommend-vs-escalate --

_ESCALATE_AUTHORITY = """\
Beyond picking the best route-slot, YOU also decide whether it is good enough to \
AUTO-ASSIGN or should go to a human specialist:
 - "decision": "RECOMMEND" -- the best option is a genuinely good assignment; \
auto-assign it.
 - "decision": "ESCALATE" -- NONE of these options are good enough to auto-assign \
(e.g. even the strongest still crowds tier 5 / Perks / 4 incumbents, fits the \
neighborhood poorly, or misses a firmly stated preference). A specialist will \
review it.
 - "confidence": "HIGH" or "LOW" -- how sure you are of that decision.
Even when you ESCALATE, still set chosen_index to the STRONGEST option (the \
specialist starts from it) and explain in your reasons why it is not good enough.
`auto_assign_threshold` and each option's `reference_weighted_score` / \
`meets_auto_assign_bar` are REFERENCE ONLY: a score at or above the threshold is \
what a fixed heuristic would auto-assign. They do NOT bind you -- you may escalate \
an option above the bar, or recommend one below it, when the facts justify it; just \
say why.
"""

_DECISION_OUTPUT_CONTRACT = """\
Reply with a SINGLE JSON object and nothing else (no markdown). Shape:

{{
  "decision": "RECOMMEND" | "ESCALATE",
  "confidence": "HIGH" | "LOW",
  "chosen_index": <index of the strongest route-slot (your pick, or the best one for
                   the specialist on an ESCALATE)>,
  "decision_summary": "<one action line: assign <route> · <day> · <window>, OR why
                        this is being escalated>",
  "primary_reasons": [
    "<geographic_clustering: your read of the neighborhood fit, WITH its number>",
    "<capacity_buffer: how much truck headroom is left, with its number>",
    "<window_match: how well the slot honors the stated preference, with its number \
-- INCLUDE this line only when the option has a window_match fact>",
    "<slot_availability: how open this slot is (who else holds it), with its number>"
  ],
  "key_tradeoff": "<what the pick gives up vs. the next-best option and why -- \
reference BOTH options' numbers>",
  "runner_up": {{
    "index": <index of the next-best option>,
    "why_not": "<the specific fact that tips the pick away from it>"
  }},
  "vs_deterministic_default": {{
    "verdict": "AGREE" | "DIVERGE",
    "note": "<if DIVERGE, why the weighted default is wrong here; may be empty on AGREE>"
  }},
  "citations": [
    {{"index": <option index>, "field": "<fact key>", "value": <number>}}
  ]
}}

Rules (STRICT):
- chosen_index MUST be one of the enumerated option indices. NEVER invent a route,
  a slot, or a time.
- `primary_reasons` MUST comprehensively cover EVERY scored factor the chosen
  option carries -- geographic_clustering, capacity_buffer, slot_availability,
  and window_match (only when the option has a window_match fact) -- one short line
  each, in that order, each citing that factor's own value. Never drop
  slot_availability, and never drop window_match when a preference was stated.
- verdict is AGREE only when chosen_index == deterministic_choice_index, else DIVERGE
  (and then `note` must justify the divergence).
- When two or more options are offered, `key_tradeoff` and `runner_up` are REQUIRED
  and runner_up.index must be a real option other than your pick. With a single
  option, use key_tradeoff to say why it is (or is not) good enough and set runner_up
  to null.
- EVERY number you state in any field must be a real fact from the option you
  attribute it to, and must appear in `citations`. Do not state a number you cannot
  cite -- a figure YOU computed (a sum, difference, average, or projection) counts as
  one you cannot cite: state the option's own numbers instead. Days and time windows
  must be quoted verbatim from the options.
- Whenever you name a route, write it as "<route_id> - <route_name>" using that
  option's own route_id and route_name -- always both together, never one alone.
- Citable fact keys are exactly: {fields}.
""".format(fields=", ".join(NUMERIC_FACT_KEYS))


def build_route_slot_decision_prompt(packet: RouteSlotPacket) -> str:
    body = json.dumps(packet.as_dict(), indent=2, sort_keys=True)
    return (
        f"{_JSON_ONLY}\n{_SYSTEM}\n{_ESCALATE_AUTHORITY}\n\n"
        f"ROUTE-SLOT OPTIONS:\n{body}\n\n{_DECISION_OUTPUT_CONTRACT}\n\n{_JSON_ONLY}"
    )


def build_route_slot_decision_retry_prompt(packet: RouteSlotPacket, feedback: str) -> str:
    return (
        f"{build_route_slot_decision_prompt(packet)}\n\n"
        f"YOUR PREVIOUS REPLY FAILED VERIFICATION for these reasons:\n{feedback}\n"
        f"Return a corrected JSON object. Keep your decision and confidence, pick only an "
        f"enumerated option index, keep the verdict consistent with that pick, include the "
        f"trade-off and runner-up when more than one option exists, and cite every number."
    )


def build_route_slot_decision_json_retry_prompt(packet: RouteSlotPacket) -> str:
    """Retry after a reply that could not be parsed as JSON (prose or a tool call)."""
    return f"{build_route_slot_decision_prompt(packet)}\n\n{_JSON_RETRY_NOTE}"
