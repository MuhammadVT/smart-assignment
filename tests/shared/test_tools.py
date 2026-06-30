"""
Unit tests for shared/tools.py. These are deterministic, fast, no LLM
involved -- run on every commit. Distinct from eval/, which tests the
LLM's actual recommendation quality and trajectory.
"""

from __future__ import annotations

from datetime import time

from smart_assignment.shared.tools import (
    filter_feasible_slots,
    geocode_and_cluster_customer,
)


def test_open_route_is_feasible(sample_customer, open_route):
    feasible = filter_feasible_slots(sample_customer, [open_route])
    assert len(feasible) == 1
    assert feasible[0].route_slot.route_id == "RTE-TEST-1"


def test_full_route_is_excluded(sample_customer, full_route):
    feasible = filter_feasible_slots(sample_customer, [full_route])
    assert feasible == []


def test_temperature_incompatible_route_is_excluded(sample_customer, open_route):
    open_route.vehicle_temp_zone = "frozen"
    sample_customer.product_temp_zone = "mixed"
    feasible = filter_feasible_slots(sample_customer, [open_route])
    assert feasible == []


def test_mixed_truck_accepts_any_product_zone(sample_customer, open_route):
    open_route.vehicle_temp_zone = "mixed"
    sample_customer.product_temp_zone = "frozen"
    feasible = filter_feasible_slots(sample_customer, [open_route])
    assert len(feasible) == 1


def test_window_outside_driver_shift_is_excluded(sample_customer, open_route):
    # Window starts before shift start -- would require unpaid overtime.
    open_route.driver_shift_start = time(9, 0)
    open_route.available_arrival_windows = [(time(8, 0), time(10, 0))]
    feasible = filter_feasible_slots(sample_customer, [open_route])
    assert feasible == []


def test_no_available_windows_is_excluded(sample_customer, open_route):
    open_route.available_arrival_windows = []
    feasible = filter_feasible_slots(sample_customer, [open_route])
    assert feasible == []


def test_geocode_returns_zone_and_customer(sample_customer):
    result = geocode_and_cluster_customer(sample_customer)
    assert result["customer"] is sample_customer
    assert result["zone_id"].startswith("ZONE_")
