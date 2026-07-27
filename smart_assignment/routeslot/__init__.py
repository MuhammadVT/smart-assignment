"""
Route-slot scoring & decision.

Makes the decision unit the (route, slot) PAIR: slot availability (tier-weighted
openness) influences which route wins, not just which slot within an
already-chosen route. This is the single decision layer for step 5: the options
are always enumerated deterministically, and the grounded flags only control
whether an LLM reasons over that set, with the threshold decision as the
fallback.
"""

from __future__ import annotations

from smart_assignment.routeslot.decide import decide_route_slot

__all__ = ["decide_route_slot"]
