"""
Load eval cases from a *file* -- so curated production feedback runs as an eval
dataset without hand-copying anything into ``eval/golden_cases.py``.

The input is the candidate-cases JSON produced by ``scripts/curate_feedback.py``
(the vendor-free path, over the feedback log) or ``scripts/phoenix_curate.py``
(the Phoenix path) -- both emit the *same* schema, so one loader serves both. Each
candidate carries the decision ``context`` (name, address, order size, and, when
captured, the stated day/window) plus the human verdict; this module reconstructs
a :class:`~smart_assignment.shared.models.CustomerProfile` and wraps it in a
:class:`~eval.golden_cases.GoldenCase`, ready for ``eval.build_evalset``.

It is honest about what it can't replay: a case whose address is missing or
PII-redacted (feedback captured with scrub on) can't be geocoded, so it is
*skipped* with a reason rather than silently producing a broken case. The caller
gets both the usable cases and the skip report.
"""

from __future__ import annotations

import json
import logging
from datetime import time
from typing import Any, Dict, List, Optional, Tuple

from smart_assignment.shared.models import CustomerProfile, DayOfWeek, PreferredSlot

logger = logging.getLogger(__name__)

# Imported lazily-safe: GoldenCase is a light dataclass, no backend needed.
from eval.golden_cases import GoldenCase  # noqa: E402


class SkippedCase(Exception):
    """A candidate that can't be turned into a replayable eval case. Carries the
    reason so the loader can report it instead of failing the whole batch."""


def _parse_window(raw: str) -> Tuple[time, time]:
    """``"09:00-12:00"`` -> ``(time(9, 0), time(12, 0))``."""
    start_s, end_s = raw.split("-", 1)

    def _t(value: str) -> time:
        hh, mm = value.strip().split(":", 1)
        return time(int(hh), int(mm))

    return _t(start_s), _t(end_s)


def _preferred_slot(context: Dict[str, Any]) -> Optional[PreferredSlot]:
    """Reconstruct the stated preference from the captured day + window, or
    ``None`` when absent/unparseable (treated as no preference, not an error)."""
    day = context.get("preferred_day")
    window = context.get("preferred_window")
    if not (day and window):
        return None
    try:
        return PreferredSlot(day=DayOfWeek[str(day).strip().upper()], window=_parse_window(window))
    except (KeyError, ValueError):
        logger.debug("Ignoring unparseable preference %r / %r.", day, window)
        return None


def _customer_from_context(context: Dict[str, Any]) -> CustomerProfile:
    """Rebuild the intake profile from a candidate's decision context. Raises
    :class:`SkippedCase` when the address or order size is missing/redacted --
    those cases can't be geocoded or scored, so they're reported, not run."""
    address = (context.get("address") or "").strip()
    if not address or "[redacted]" in address:
        raise SkippedCase("address missing or PII-redacted (needs scrub-off feedback)")
    cases = context.get("order_quantity_cases")
    if cases is None:
        raise SkippedCase("order_quantity_cases missing")
    try:
        cases_int = int(cases)
    except (TypeError, ValueError):
        raise SkippedCase(f"order_quantity_cases not an int: {cases!r}")
    return CustomerProfile(
        name=(context.get("name") or "Curated prospect").strip() or "Curated prospect",
        address=address,
        order_quantity_cases=cases_int,
        preferred_slot=_preferred_slot(context),
    )


def _query_for(customer: CustomerProfile) -> str:
    """A natural-language intake message consistent with ``customer`` -- so the
    reconstructed query and the expected intake args can't disagree (the
    trajectory eval compares intake args exactly)."""
    parts: List[str] = [customer.name, customer.address, f"{customer.order_quantity_cases} cases"]
    slot = customer.preferred_slot
    if slot is not None:
        window = f"{slot.window[0].strftime('%H:%M')}-{slot.window[1].strftime('%H:%M')}"
        parts.append(f"prefers {slot.day.name} {window}")
    return ", ".join(parts)


def _expected_outcome(candidate: Dict[str, Any]) -> str:
    """The narrative outcome target: the human-set ``suggested_expected_outcome``
    when present (a promoted negative), else the observed outcome. Not scored by
    the trajectory metric, but recorded so a later response/outcome eval has it."""
    suggested = candidate.get("suggested_expected_outcome")
    if suggested:
        return str(suggested)
    return str(candidate.get("observed_outcome") or "recommend")


def candidate_to_case(candidate: Dict[str, Any]) -> GoldenCase:
    """Reconstruct one :class:`GoldenCase` from a curated candidate dict.
    Raises :class:`SkippedCase` when it can't be replayed."""
    context = candidate.get("context") or {}
    customer = _customer_from_context(context)
    eval_id = str(candidate.get("eval_id") or "curated_case")
    note = candidate.get("note") or "Curated from production human feedback."
    return GoldenCase(
        eval_id=eval_id,
        query=_query_for(customer),
        customer=customer,
        expected_outcome=_expected_outcome(candidate),
        note=str(note),
    )


def load_curated_cases(path: str) -> Tuple[List[GoldenCase], List[Dict[str, str]]]:
    """Load a curated-candidates JSON file into eval cases.

    Returns ``(cases, skipped)`` -- the replayable :class:`GoldenCase` list and a
    report of every candidate that couldn't be replayed (``{eval_id, reason}``),
    so a scrub-on batch degrades to "these N couldn't be replayed" rather than a
    hard failure. Duplicate eval_ids are de-duplicated (first wins), keeping the
    ADK evalset's ids unique."""
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, list):
        raise ValueError(f"{path} must contain a JSON array of candidate cases")

    cases: List[GoldenCase] = []
    skipped: List[Dict[str, str]] = []
    seen: set[str] = set()
    for candidate in raw:
        eval_id = str((candidate or {}).get("eval_id") or "curated_case")
        try:
            case = candidate_to_case(candidate)
        except SkippedCase as exc:
            skipped.append({"eval_id": eval_id, "reason": str(exc)})
            continue
        if case.eval_id in seen:
            skipped.append({"eval_id": case.eval_id, "reason": "duplicate eval_id"})
            continue
        seen.add(case.eval_id)
        cases.append(case)
    return cases, skipped
