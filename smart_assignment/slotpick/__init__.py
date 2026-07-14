"""
Grounded slot selection: let an LLM pick the FINAL delivery slot for the chosen
route from that route's deterministically-enumerated candidate menu.

This is the "constrained-option, grounded reasoning" pattern applied to the slot
decision:

  - the valid options are fixed upstream -- the candidate slots produced by
    `shared/slot_selection.identify_available_slots` + `select_candidate_slots`
    (already on `CandidateEvaluation.available_slots`);
  - the LLM picks one by INDEX (it cannot invent a time), with a rationale that
    must cite the candidates' own facts;
  - a deterministic verifier checks the index is valid and every cited figure is
    grounded, and any failure falls back to the deterministic pick.

So the model reasons over the trade-offs (fit vs. contention vs. preference)
without an arbitrary fixed weighting, but can never pick an invalid slot, change
the route, or move the score -- it only chooses which enumerated candidate is
presented. Opt-in via `Config.use_grounded_slot_selection`; off by default, the
deterministic blend stands.
"""

from __future__ import annotations

from smart_assignment.slotpick.selector import (
    DeterministicSlotSelector,
    GroundedSlotSelector,
    SlotPick,
    SlotSelector,
    default_slot_selector,
    refine_slot,
)

__all__ = [
    "DeterministicSlotSelector",
    "GroundedSlotSelector",
    "SlotPick",
    "SlotSelector",
    "default_slot_selector",
    "refine_slot",
]
