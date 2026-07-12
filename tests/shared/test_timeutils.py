"""Unit tests for the delivery-window time helpers (shared/timeutils.py)."""

from __future__ import annotations

from datetime import time

import pytest

from smart_assignment.shared.timeutils import (
    best_overlapping_window,
    overlap_minutes,
    window_midpoint,
)


@pytest.mark.parametrize(
    "window, expected",
    [
        ((time(7, 0), time(10, 0)), time(8, 30)),
        ((time(13, 0), time(15, 0)), time(14, 0)),
        ((time(8, 0), time(12, 0)), time(10, 0)),
        ((time(9, 0), time(9, 0)), time(9, 0)),  # zero-width window
        ((time(7, 0), time(10, 15)), time(8, 37)),  # odd total -> floor of minutes
    ],
)
def test_window_midpoint(window, expected):
    assert window_midpoint(window) == expected


def test_overlap_minutes_disjoint_is_zero():
    assert overlap_minutes((time(7, 0), time(10, 0)), (time(13, 0), time(15, 0))) == 0


def test_best_overlapping_window_unchanged_behavior():
    # Regression guard: the legacy helper is retained (still importable/tested)
    # even though build_context now uses the location-aware selector.
    avail = [(time(7, 0), time(10, 0)), (time(13, 0), time(15, 0))]
    assert best_overlapping_window((time(8, 0), time(9, 0)), avail)[0] == (time(7, 0), time(10, 0))
    assert best_overlapping_window(None, avail) == ((time(7, 0), time(10, 0)), 0)
    assert best_overlapping_window((time(8, 0), time(9, 0)), []) == (None, 0)
