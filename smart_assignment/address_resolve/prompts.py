"""Prompt construction for the grounded address-choice LLM call."""

from __future__ import annotations

import json

from smart_assignment.address_resolve.evidence import NUMERIC_ADDRESS_FIELDS, AddressPacket

_SYSTEM = """\
You are correcting a delivery address for a Sysco prospect. The address the user \
typed could not be geocoded exactly, so a geocoding service returned the \
ENUMERATED candidate addresses below (real, existing addresses). Your only job \
is to pick the candidate that best matches what the user INTENDED -- you do NOT \
write or invent an address, you only choose one of the candidates by index.

Reason from what the user typed versus each candidate:
 - a candidate that matches the street number, street name, and city the user \
typed -- allowing for a small typo or a missing ZIP/state -- is a strong match;
 - `similarity` is a rough token-overlap score (fraction of the user's words \
found in the candidate; higher is closer). It is REFERENCE ONLY -- a sanity \
check, not the answer, because it can't tell a typo ("mckiney") from a real \
mismatch. `deterministic_choice_index` names the candidate that score alone \
would pick.
 - use `components` (when present) to confirm the city/state make sense.

Pick the single best candidate. If several are plausible, prefer the one whose \
street number and street name match most closely. A human will confirm your \
pick before anything uses it, so choose the most likely intended address rather \
than refusing.
"""

_OUTPUT_CONTRACT = """\
Reply with a SINGLE JSON object and nothing else (no markdown, no prose around \
it). Shape:

{{
  "chosen_index": <the index of the candidate you pick>,
  "rationale": "<1-2 sentences: why this candidate matches what the user typed>",
  "citations": [
    {{"index": <candidate index>, "field": "<fact key>", "value": <number>}}
  ]
}}

Rules (STRICT):
- chosen_index MUST be one of the candidates' index values. NEVER invent an \
address or an index.
- Back every figure in your rationale with a citation whose value exactly \
matches that candidate's fact.
- Citable fact keys are exactly: {fields}.
""".format(fields=", ".join(NUMERIC_ADDRESS_FIELDS))


def build_address_prompt(packet: AddressPacket) -> str:
    body = json.dumps(packet.as_dict(), indent=2, sort_keys=True)
    return (
        f"{_SYSTEM}\n\nADDRESS THE USER TYPED: {packet.query!r}\n\n"
        f"CANDIDATES:\n{body}\n\n{_OUTPUT_CONTRACT}"
    )


def build_address_retry_prompt(packet: AddressPacket, feedback: str) -> str:
    return (
        f"{build_address_prompt(packet)}\n\n"
        f"YOUR PREVIOUS REPLY FAILED VERIFICATION for these reasons:\n{feedback}\n"
        f"Return a corrected JSON object. Pick only an enumerated candidate index "
        f"and cite only facts that appear verbatim on that candidate."
    )
