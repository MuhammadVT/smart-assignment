"""
Grounded address resolution: turn an address that wouldn't geocode into a
*confirmable suggestion*, instead of a dead-end -- without inventing an address.

This is the "constrained-option, grounded reasoning" pattern (as in `judgment/`,
`slotpick/`, `routeslot/`) applied to address correction:

  - the valid options are fixed upstream -- the geocoder's own candidate matches
    (`Geocoder.suggest`, a provider-agnostic capability; see shared/geo.py);
  - an LLM picks one by INDEX (it cannot write a new address), with a rationale
    that must cite the candidates' own facts;
  - a deterministic verifier checks the index is valid and every cited figure is
    grounded, and any failure falls back to the deterministic highest-similarity
    candidate -- and if there are no candidates at all, the caller falls back to
    today's "ask the customer to double-check it."

The pick is only ever a *suggestion*: a human confirms it before anything acts
on it (see the ``resolve_address`` tool in tools/slot_recommendation.py and the
agent instruction). Gated by ``Config.use_address_resolution``.
"""

from __future__ import annotations

from smart_assignment.address_resolve.resolver import (
    ChoiceFn,
    ResolvedAddress,
    resolve_address,
    resolve_from_geocoder,
    suggest_addresses,
)

__all__ = [
    "ChoiceFn",
    "ResolvedAddress",
    "resolve_address",
    "resolve_from_geocoder",
    "suggest_addresses",
]
