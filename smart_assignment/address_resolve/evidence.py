"""
The address evidence packet: the address the user typed, plus the geocoder's
candidate matches enumerated for the LLM to choose among, each with the facts it
may cite. Nothing here calls an LLM or a geocoder.

A deterministic token-overlap **similarity** is attached to every candidate and
the highest-similarity candidate is offered as the ``deterministic_choice_index``
-- the same "demote the heuristic to a grounded reference + fallback" move the
other layers use (see slotpick's ``blended_score``). It is a strong default the
model may agree with or, with justification, diverge from; it is also the pick
used when the LLM path fails.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from smart_assignment.shared.config import Config
from smart_assignment.shared.geo import AddressCandidate

# The numeric fact keys a citation may reference on a candidate.
NUMERIC_ADDRESS_FIELDS = ("similarity",)


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if t}


def similarity(query: str, formatted: str) -> float:
    """Deterministic 0-1 overlap: the fraction of the tokens the user typed that
    also appear in a candidate. Recall over the query tokens (not Jaccard) so a
    candidate isn't penalized for the extra tokens a canonical address adds
    (state, ZIP). A rough reference only -- the LLM does the smarter matching."""
    q = _tokens(query)
    if not q:
        return 0.0
    return round(len(q & _tokens(formatted)) / len(q), 4)


@dataclass
class AddressPacket:
    """JSON-safe view of the geocoder's candidate matches for the LLM, plus the
    original `AddressCandidate`s kept for mapping a chosen index back to a match."""

    query: str
    candidates: list[dict]
    # Index of the highest-similarity candidate -- the fallback pick, surfaced to
    # the LLM as a reference. None when the candidate set is empty.
    deterministic_choice_index: Optional[int] = None
    _candidates: list[AddressCandidate] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.candidates)

    def candidate_at(self, index: int) -> Optional[AddressCandidate]:
        if 0 <= index < len(self._candidates):
            return self._candidates[index]
        return None

    def candidate_facts(self, index: int) -> Optional[dict]:
        if 0 <= index < len(self.candidates):
            return self.candidates[index]["facts"]
        return None

    def as_dict(self) -> dict:
        return {
            "query": self.query,
            "deterministic_choice_index": self.deterministic_choice_index,
            "candidates": self.candidates,
        }


def build_address_packet(
    query: str, candidates: list[AddressCandidate], config: Config
) -> AddressPacket:
    """Enumerate the geocoder's candidate matches (with a per-candidate
    similarity fact) for the grounded address resolver. Order is preserved as the
    geocoder returned it; the deterministic reference is the highest-similarity
    candidate (ties broken by the earliest index)."""
    rows: list[dict] = []
    best_index: Optional[int] = None
    best_score = -1.0
    for i, cand in enumerate(candidates):
        score = similarity(query, cand.formatted)
        row = {
            "index": i,
            "formatted": cand.formatted,
            "facts": {"similarity": score},
        }
        if cand.components:
            row["components"] = cand.components
        rows.append(row)
        if score > best_score:
            best_score, best_index = score, i

    return AddressPacket(
        query=query,
        candidates=rows,
        deterministic_choice_index=best_index,
        _candidates=list(candidates),
    )
