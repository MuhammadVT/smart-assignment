"""
Tests for the deterministic chat-message parser (smart_assignment/webapp/parse.py).

The parser turns free text into intake fields without an LLM; these guard the
extraction of address, order quantity, and the optional day+time slot, plus the
clarifying-question path when required fields are missing.
"""

from __future__ import annotations

from datetime import time

from smart_assignment.shared.models import DayOfWeek
from smart_assignment.webapp.parse import describe_slot, parse_intake


def test_parses_full_message():
    res = parse_intake("1200 McKinney St, Houston, TX 77010, 90 cases, TUE 07:00-10:00")
    assert res.profile is not None
    assert res.profile.address == "1200 McKinney St, Houston, TX 77010"
    assert res.profile.order_quantity_cases == 90
    slot = res.profile.preferred_slot
    assert slot is not None
    assert slot.day == DayOfWeek.TUE
    assert slot.window == (time(7, 0), time(10, 0))


def test_slot_is_optional():
    res = parse_intake("5085 Westheimer Rd, Houston, TX 77056, 400 cases")
    assert res.profile is not None
    assert res.profile.preferred_slot is None
    assert res.missing == []


def test_city_is_kept_when_it_shares_a_segment_with_the_order_quantity():
    # Regression: "Houston. 90 cases" was one comma-segment, and dropping the
    # whole segment for containing "90 cases" also dropped the city, leaving a
    # bare "1200 McKinney St" that the live geocoder can't match.
    res = parse_intake("1200 McKinney St, Houston. 90 cases")
    assert res.profile is not None
    assert res.profile.address == "1200 McKinney St, Houston"
    assert res.profile.order_quantity_cases == 90


def test_order_of_lead_in_is_not_left_in_the_address():
    res = parse_intake("1200 Main St, order of 150 cases")
    assert res.profile is not None
    assert res.profile.address == "1200 Main St"
    assert res.profile.order_quantity_cases == 150


def test_missing_cases_asks_for_them():
    res = parse_intake("1200 McKinney St, Houston, TX 77010")
    assert res.profile is None
    assert "order quantity (in cases)" in res.missing
    assert res.clarify and "cases" in res.clarify.lower()


def test_missing_address_asks_for_it():
    res = parse_intake("90 cases, TUE 07:00-10:00")
    assert res.profile is None
    assert "address" in res.missing


def test_parses_am_pm_and_weekday_names():
    res = parse_intake("400 Louisiana St, Houston, TX 77002, 120 cases, Thursday 9am-12pm")
    assert res.profile is not None
    slot = res.profile.preferred_slot
    assert slot is not None
    assert slot.day == DayOfWeek.THU
    assert slot.window == (time(9, 0), time(12, 0))


def test_describe_slot():
    res = parse_intake("1 Main St, Houston, TX 77002, 10 cases, FRI 08:00-11:00")
    assert describe_slot(res.profile.preferred_slot) == "FRI 08:00-11:00"
    assert describe_slot(None) == "any"
