"""
Grounded address resolution: given the geocoder's candidate matches for an
address that wouldn't resolve exactly, let an LLM pick the closest one BY INDEX
(it cannot invent an address), verify the pick against the enumerated set, and
fall back to the deterministic highest-similarity candidate on any failure.

Two entry points:
  - ``resolve_address(query, candidates, config, choice_fn)`` -- pure over a
    candidate list; fully offline-testable with a fake ``choice_fn``.
  - ``resolve_from_geocoder(address, geocoder, config)`` -- feature-detects a
    suggest-capable geocoder, fetches its candidates, and resolves them.

Both return ``None`` when there is nothing to choose among (no candidates), which
the caller turns into today's "ask the customer to double-check it" fallback.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

from smart_assignment.address_resolve.evidence import build_address_packet
from smart_assignment.address_resolve.llm import generate_address_choice
from smart_assignment.address_resolve.prompts import (
    build_address_prompt,
    build_address_retry_prompt,
)
from smart_assignment.address_resolve.schema import parse_address_choice
from smart_assignment.address_resolve.verifier import verify_choice
from smart_assignment.shared.config import Config
from smart_assignment.shared.geo import AddressCandidate, supports_suggestions

logger = logging.getLogger(__name__)

# A choice_fn turns (config, prompt) into a raw address-choice dict. Injectable
# so tests drive the resolver with a fake and no network/credentials.
ChoiceFn = Callable[[Config, str], dict]


@dataclass(frozen=True)
class ResolvedAddress:
    """The grounded pick among a geocoder's candidate matches, for a human to
    confirm. ``provenance`` is "llm" when the verified model pick was used, or
    "deterministic" when it fell back to the highest-similarity candidate."""

    chosen: AddressCandidate
    alternatives: list[AddressCandidate] = field(default_factory=list)
    provenance: str = "deterministic"
    rationale: Optional[str] = None


def resolve_address(
    query: str,
    candidates: list[AddressCandidate],
    config: Config,
    choice_fn: Optional[ChoiceFn] = None,
) -> Optional[ResolvedAddress]:
    """Pick the candidate that best matches ``query``, grounded + verified, with
    the deterministic highest-similarity candidate as the fallback. Returns
    ``None`` only when ``candidates`` is empty."""
    if not candidates:
        return None

    packet = build_address_packet(query, candidates, config)
    # deterministic_choice_index is non-None because candidates is non-empty.
    final_index: int = packet.deterministic_choice_index or 0
    provenance = "deterministic"
    rationale: Optional[str] = None

    choose = choice_fn or generate_address_choice
    try:
        choice = parse_address_choice(choose(config, build_address_prompt(packet)))
        result = verify_choice(choice, packet)
        if not result.ok:
            retry_prompt = build_address_retry_prompt(packet, result.as_feedback())
            choice = parse_address_choice(choose(config, retry_prompt))
            result = verify_choice(choice, packet)
        if result.ok:
            final_index, rationale, provenance = choice.chosen_index, choice.rationale, "llm"
        else:
            logger.warning(
                "Grounded address choice ungrounded after one retry (%s); using the "
                "highest-similarity candidate.",
                result.as_feedback(),
            )
    except Exception as exc:  # noqa: BLE001 - any backend/parse failure -> fallback
        logger.warning(
            "Grounded address resolution failed (%s: %s); using the highest-similarity "
            "candidate. Check SMART_ASSIGNMENT_LLM_BACKEND and its credentials.",
            type(exc).__name__,
            exc,
        )

    chosen = packet.candidate_at(final_index)
    if chosen is None:  # defensive; verifier already checked the range
        return None
    alternatives = [c for i, c in enumerate(candidates) if i != final_index]
    return ResolvedAddress(
        chosen=chosen,
        alternatives=alternatives,
        provenance=provenance,
        rationale=rationale,
    )


def suggest_addresses(
    address: str, geocoder: object, *, limit: int = 5
) -> list[AddressCandidate]:
    """Best-effort candidate matches from a suggest-capable geocoder. Returns
    ``[]`` when the geocoder can't suggest or the lookup fails (transport error,
    no match) -- so the caller cleanly falls back to asking for a corrected
    address rather than surfacing an error."""
    if not supports_suggestions(geocoder):
        return []
    try:
        return list(geocoder.suggest(address, limit=limit))
    except Exception as exc:  # noqa: BLE001 - suggestions are best-effort
        logger.warning(
            "Address suggest lookup failed (%s: %s); no candidates.",
            type(exc).__name__,
            exc,
        )
        return []


def resolve_from_geocoder(
    address: str,
    geocoder: object,
    config: Config,
    choice_fn: Optional[ChoiceFn] = None,
    *,
    limit: int = 5,
) -> Optional[ResolvedAddress]:
    """Fetch candidate matches for ``address`` from a suggest-capable geocoder and
    resolve them (grounded + verified). ``None`` when no candidate is available."""
    candidates = suggest_addresses(address, geocoder, limit=limit)
    return resolve_address(address, candidates, config, choice_fn=choice_fn)
