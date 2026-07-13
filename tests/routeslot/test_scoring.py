"""Route-slot scoring: tier harm, slot openness, and score_route_slot."""

from __future__ import annotations

from datetime import time

from smart_assignment.shared.config import (
    Config,
    FACTOR_SLOT_AVAILABILITY,
    FACTOR_WINDOW_MATCH,
)
from smart_assignment.shared.constraints import build_context
from smart_assignment.shared.scoring import (
    score_route_slot,
    slot_openness,
    tier_weighted_contention,
)

from .conftest import MORNING, customer, route, slot_option, stop


def test_tier_harm_ordering():
    c = Config()
    assert c.tier_harm_weight("5") == c.tier_harm_weight("Perks") == 1.0
    assert c.tier_harm_weight("4") == 0.6
    assert c.tier_harm_weight("Other") == 0.1
    assert c.tier_harm_weight(None) == 0.4  # unknown fallback
    # High-tier is protected more than low-tier.
    assert c.tier_harm_weight("5") > c.tier_harm_weight("4") > c.tier_harm_weight("Other")


def test_openness_is_full_when_no_committed_stop_shares_the_window():
    r = route(committed=[stop(29.75, -95.36, (time(13, 0), time(15, 0)), tier="5")])
    # A morning window doesn't overlap the afternoon committed stop -> fully open.
    assert slot_openness(MORNING, r, Config()) == 1.0


def test_openness_penalizes_high_tier_incumbents_more():
    high = route(committed=[stop(29.75, -95.36, MORNING, tier="5")])
    low = route(committed=[stop(29.75, -95.36, MORNING, tier="Other")])
    o_high = slot_openness(MORNING, high, Config())
    o_low = slot_openness(MORNING, low, Config())
    assert o_high < o_low                       # crowding a tier-5 is worse
    assert o_high == 1.0 / (1.0 + 1.0)          # 0.5
    assert abs(o_low - 1.0 / 1.1) < 1e-9        # ~0.909
    # And the raw harm sum reflects the tier weights.
    assert tier_weighted_contention(MORNING, high, Config()) == 1.0


def test_score_route_slot_drops_window_match_without_a_preference():
    r = route(committed=[stop(29.75, -95.36, MORNING, tier="Other")])
    cfg = Config()
    ctx = build_context(customer(pref=None), r, cfg)
    breakdown, total = score_route_slot(customer(pref=None), r, ctx, slot_option(MORNING), cfg)
    names = {fs.name for fs in breakdown}
    assert FACTOR_WINDOW_MATCH not in names            # no preference -> factor absent
    assert FACTOR_SLOT_AVAILABILITY in names
    assert 0.0 <= total <= 1.0


def test_score_route_slot_includes_window_match_with_a_preference():
    r = route(committed=[stop(29.75, -95.36, MORNING, tier="Other")])
    cfg = Config()
    cust = customer(pref=MORNING)
    ctx = build_context(cust, r, cfg)
    breakdown, _ = score_route_slot(cust, r, ctx, slot_option(MORNING), cfg)
    assert FACTOR_WINDOW_MATCH in {fs.name for fs in breakdown}
