"""
Turn a free-text chat message into structured intake fields, without an LLM.

This is the deterministic (Phase 1) "brain" behind the chat box: it extracts
the three things the pipeline needs from whatever the user typed — an address,
an order quantity in cases, and an optional preferred slot (day of week + time
window) — and reports back what is still missing so the UI can ask a clarifying
question. It never computes geography, capacity, or scoring; it only assembles
a ``CustomerProfile`` for ``run_slot_recommendation`` to run on.

The day/time vocabulary and window formatting are reused from
``shared.models`` / ``shared.timeutils`` so parsing stays consistent with the
rest of the system. When Phase 2's LLM conversation is enabled, the model
supersedes this parser for genuine natural language.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import time
from typing import Optional

from smart_assignment.shared.models import (
    CustomerProfile,
    DayOfWeek,
    PreferredSlot,
    Window,
)
from smart_assignment.shared.timeutils import fmt_window

# Map the spellings a user might type to the canonical DayOfWeek. Sunday is
# intentionally absent — routes only run MON–SAT (see DayOfWeek).
_DAY_ALIASES: dict[str, DayOfWeek] = {}
for _d in DayOfWeek:
    _DAY_ALIASES[_d.value.lower()] = _d  # "tue"
_DAY_ALIASES.update(
    {
        "monday": DayOfWeek.MON,
        "mon": DayOfWeek.MON,
        "tuesday": DayOfWeek.TUE,
        "tues": DayOfWeek.TUE,
        "wednesday": DayOfWeek.WED,
        "weds": DayOfWeek.WED,
        "thursday": DayOfWeek.THU,
        "thurs": DayOfWeek.THU,
        "thur": DayOfWeek.THU,
        "friday": DayOfWeek.FRI,
        "fri": DayOfWeek.FRI,
        "saturday": DayOfWeek.SAT,
        "sat": DayOfWeek.SAT,
    }
)

# "400 cases", "90 case", "order of 150 cases" -> the integer before "case(s)".
_CASES_RE = re.compile(r"(\d+)\s*cases?\b", re.IGNORECASE)

# A time window like "07:00-10:00", "7:00 - 10:00", "07:00–10:00" (en dash),
# "7-10" or "7am-10am". Captures the two endpoints.
_WINDOW_RE = re.compile(
    r"(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\s*(?:-|–|—|to)\s*(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)",
    re.IGNORECASE,
)


@dataclass
class ParseResult:
    """Outcome of parsing one chat message into intake fields.

    ``profile`` is populated only when the required fields (address + a positive
    order quantity) are present; otherwise ``missing`` lists what to ask for and
    ``clarify`` is a ready-to-send agent prompt.
    """

    profile: Optional[CustomerProfile]
    missing: list[str]
    clarify: Optional[str]
    # Echoes of what was understood, for a friendly confirmation line.
    address: Optional[str] = None
    order_quantity_cases: Optional[int] = None
    preferred_slot: Optional[PreferredSlot] = None


def _parse_clock(token: str) -> Optional[time]:
    """Parse a single clock token: '07:00', '7', '7am', '10:30 pm'."""
    token = token.strip().lower()
    m = re.match(r"^(\d{1,2})(?::(\d{2}))?\s*(am|pm)?$", token)
    if not m:
        return None
    hour = int(m.group(1))
    minute = int(m.group(2) or 0)
    meridiem = m.group(3)
    if meridiem == "pm" and hour != 12:
        hour += 12
    elif meridiem == "am" and hour == 12:
        hour = 0
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    return time(hour, minute)


def _extract_window(text: str) -> Optional[Window]:
    m = _WINDOW_RE.search(text)
    if not m:
        return None
    start = _parse_clock(m.group(1))
    end = _parse_clock(m.group(2))
    if start is None or end is None or start >= end:
        return None
    return (start, end)


def _extract_day(text: str) -> Optional[DayOfWeek]:
    # Word-boundary match so "satisfied" doesn't read as SAT, etc.
    for token in re.findall(r"[a-z]+", text.lower()):
        day = _DAY_ALIASES.get(token)
        if day is not None:
            return day
    return None


def _extract_cases(text: str) -> Optional[int]:
    m = _CASES_RE.search(text)
    if not m:
        return None
    value = int(m.group(1))
    return value if value > 0 else None


def _extract_address(text: str, cases: Optional[int]) -> Optional[str]:
    """Best-effort address extraction from a comma-delimited message.

    A prospect message typically looks like
    ``"1200 McKinney St, Houston, TX 77010, 90 cases, TUE 07:00-10:00"``.
    We drop the segments that clearly encode the order quantity or the slot and
    treat the remaining street/city/state/zip segments as the address. Requires
    a leading street number to avoid mistaking a bare "Houston" for an address.
    """
    parts = [p.strip() for p in text.split(",") if p.strip()]
    kept: list[str] = []
    for part in parts:
        low = part.lower()
        if _CASES_RE.search(low):
            continue
        if _WINDOW_RE.search(low):
            continue
        # A segment that is only a day name ("TUE") is slot info, not address.
        if low in _DAY_ALIASES:
            continue
        # "TUE 07:00-10:00" with no comma between day and window: strip a
        # leading day token if the rest is empty afterwards.
        kept.append(part)
    candidate = ", ".join(kept).strip()
    if not candidate:
        return None
    # Require a street number so we don't accept vague free text as an address.
    if not re.match(r"^\s*\d", candidate):
        return None
    return candidate


def parse_intake(message: str, name: str = "New prospect") -> ParseResult:
    """Parse one chat message into a ``CustomerProfile`` (or a clarifying ask).

    Returns a :class:`ParseResult`. When required fields are missing, ``profile``
    is ``None`` and ``clarify`` holds a question to send back to the user.
    """
    text = (message or "").strip()
    cases = _extract_cases(text)
    address = _extract_address(text, cases)

    day = _extract_day(text)
    window = _extract_window(text)
    slot: Optional[PreferredSlot] = None
    if day is not None and window is not None:
        slot = PreferredSlot(day, window)

    missing: list[str] = []
    if not address:
        missing.append("address")
    if cases is None:
        missing.append("order quantity (in cases)")

    if missing:
        if missing == ["address"]:
            clarify = "What's the delivery address? (street, city, state, ZIP)"
        elif missing == ["order quantity (in cases)"]:
            clarify = "How many cases is the order?"
        else:
            clarify = (
                "I need a delivery address and an order quantity in cases to run "
                "the workflow. For example: "
                "“1200 McKinney St, Houston, TX 77010, 90 cases, TUE 07:00-10:00”."
            )
        return ParseResult(
            profile=None,
            missing=missing,
            clarify=clarify,
            address=address,
            order_quantity_cases=cases,
            preferred_slot=slot,
        )

    profile = CustomerProfile(
        name=name,
        address=address,
        order_quantity_cases=cases,
        preferred_slot=slot,
    )
    return ParseResult(
        profile=profile,
        missing=[],
        clarify=None,
        address=address,
        order_quantity_cases=cases,
        preferred_slot=slot,
    )


def describe_slot(slot: Optional[PreferredSlot]) -> str:
    """Human phrase for a preferred slot, e.g. 'TUE 07:00-10:00' or 'any'."""
    if slot is None:
        return "any"
    return f"{slot.day.value} {fmt_window(slot.window)}"
