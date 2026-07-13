"""
Route-slot scoring & decision (Config.use_route_slot_scoring).

Makes the decision unit the (route, slot) PAIR: slot availability (tier-weighted
openness) influences which route wins, not just which slot within an
already-chosen route. Supersedes the two-stage judge+slotpick flow when on; the
prior route-only path is untouched and remains the rollback (flag off).
"""

from __future__ import annotations

from smart_assignment.routeslot.decide import decide_route_slot

__all__ = ["decide_route_slot"]
