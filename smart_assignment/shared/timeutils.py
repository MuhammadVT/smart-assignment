"""
Small helpers for reasoning about delivery windows (start, end time pairs).
Shared by the constraint checker and the scorer so the two never disagree
about what "overlap" means.
"""

from __future__ import annotations

from datetime import time
from typing import Optional, Union

from smart_assignment.shared.models import DayOfWeek, Window

_DAY_NAMES = {
    "MON": "Monday",
    "TUE": "Tuesday",
    "WED": "Wednesday",
    "THU": "Thursday",
    "FRI": "Friday",
    "SAT": "Saturday",
}


def day_label(day: Union[DayOfWeek, str]) -> str:
    """Full weekday name for a DayOfWeek (or its short code), for natural prose."""
    code = day.value if isinstance(day, DayOfWeek) else day
    return _DAY_NAMES.get(code, code)


def _minutes(t: time) -> int:
    return t.hour * 60 + t.minute


def duration_minutes(window: Window) -> int:
    return _minutes(window[1]) - _minutes(window[0])


def overlap_minutes(a: Window, b: Window) -> int:
    """Minutes of overlap between two windows (0 if they don't overlap)."""
    start = max(_minutes(a[0]), _minutes(b[0]))
    end = min(_minutes(a[1]), _minutes(b[1]))
    return max(0, end - start)


def best_overlapping_window(
    preferred: Optional[Window], available: list[Window]
) -> tuple[Optional[Window], int]:
    """
    Pick the available window that best serves the customer.

    - If the customer stated a preference, return the available window with
      the most overlap (and the overlap in minutes).
    - If no preference, return the earliest available window with overlap 0
      (there is nothing to overlap against).
    - If there are no available windows, return (None, 0).
    """
    if not available:
        return None, 0
    if preferred is None:
        earliest = min(available, key=lambda w: _minutes(w[0]))
        return earliest, 0
    best = max(available, key=lambda w: overlap_minutes(preferred, w))
    return best, overlap_minutes(preferred, best)


def fmt_time(t: time) -> str:
    return t.strftime("%H:%M")


def parse_time(value: str) -> time:
    """Parse a 24-hour ``"HH:MM"`` string into a ``time`` (inverse of `fmt_time`)."""
    hour_str, minute_str = value.strip().split(":", 1)
    return time(int(hour_str), int(minute_str))


def fmt_window(window: Optional[Window]) -> str:
    if window is None:
        return "n/a"
    return f"{fmt_time(window[0])}-{fmt_time(window[1])}"
