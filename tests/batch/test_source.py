"""Tests for batch prospect sources (batch/source.py). Offline, no LLM."""

from __future__ import annotations

import json

from smart_assignment.batch.source import MockProspectSource, Prospect
from smart_assignment.shared.models import CustomerProfile, DayOfWeek


def test_from_samples_wraps_every_sample_with_a_stable_id():
    source = MockProspectSource.from_samples()
    prospects = list(source.prospects())

    assert len(prospects) == 4  # the built-in demo set
    assert all(isinstance(p, Prospect) for p in prospects)
    assert all(isinstance(p.profile, CustomerProfile) for p in prospects)
    ids = [p.prospect_id for p in prospects]
    assert ids == ["MOCK-001", "MOCK-002", "MOCK-003", "MOCK-004"]


def test_from_json_parses_intake_including_a_preferred_slot(tmp_path):
    records = [
        {
            "prospect_id": "SF-42",
            "name": "Test Bistro",
            "address": "1200 McKinney St, Houston, TX 77010",
            "order_quantity_cases": 90,
            "preferred_day": "tue",
            "preferred_window_start": "07:00",
            "preferred_window_end": "10:00",
        },
        {
            "name": "No Preference Cafe",
            "address": "5085 Westheimer Rd, Houston, TX 77056",
            "order_quantity_cases": 120,
        },
    ]
    path = tmp_path / "prospects.json"
    path.write_text(json.dumps(records), encoding="utf-8")

    prospects = list(MockProspectSource.from_json(path).prospects())

    assert [p.prospect_id for p in prospects] == ["SF-42", "ROW-002"]  # id, then synthesized

    first = prospects[0].profile
    assert first.order_quantity_cases == 90
    assert first.preferred_slot is not None
    assert first.preferred_slot.day == DayOfWeek.TUE  # normalized from "tue"

    # A record with no day/window is treated as no preference, never guessed.
    assert prospects[1].profile.preferred_slot is None
