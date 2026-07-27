"""
Weighted multi-factor scoring of (route, slot) pairs (spec step 4).

The decision UNIT is the (route, slot) pair: every candidate slot on every
feasible route is scored separately, so slot availability influences which
ROUTE wins -- not just which slot within an already-chosen route.

Like `constraints.py`, this is deliberately modular: each factor is a small
pure function returning a normalized 0.0-1.0 value plus a human-readable
`detail`. The pair's total is the weighted average over whichever factors are
active, with weights living in `Config.rs_weight_*`.

Factors:
  1. geographic_clustering — tightness of fit with existing stops on the route
                             (route-level, shared across that route's slots)
  2. capacity_buffer       — stays flat once safely under the capacity
                             ceiling; only decays as utilization approaches it
                             (route-level)
  3. window_match          — how much THIS candidate window covers the
                             customer's preferred slot (slot-level; omitted
                             entirely when no preference was stated)
  4. slot_availability     — how OPEN this candidate window is, tier-weighted
                             by who already holds it (slot-level)
"""

from __future__ import annotations

from typing import Optional

from smart_assignment.shared.config import (
    FACTOR_CAPACITY_BUFFER,
    FACTOR_GEO_CLUSTERING,
    FACTOR_SLOT_AVAILABILITY,
    FACTOR_WINDOW_MATCH,
    Config,
)
from smart_assignment.shared.constraints import EvalContext
from smart_assignment.shared.models import CustomerProfile, FactorScore, Route, SlotOption
from smart_assignment.shared.timeutils import day_label, duration_minutes, overlap_minutes


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def geographic_clustering(
    customer: CustomerProfile, route: Route, ctx: EvalContext, config: Config
) -> FactorScore:
    """Closer to the route's existing cluster of stops -> higher score."""
    value = _clamp01(1.0 - ctx.avg_stop_distance_miles / config.cluster_reference_miles)
    return FactorScore(
        name=FACTOR_GEO_CLUSTERING,
        weight=config.rs_weight_geo,
        value=value,
        detail=f"avg {ctx.avg_stop_distance_miles:.1f} mi to existing stops",
    )


def capacity_buffer(
    customer: CustomerProfile, route: Route, ctx: EvalContext, config: Config
) -> FactorScore:
    """
    Reward staying comfortably under the capacity ceiling, without endlessly
    rewarding emptiness beyond that.

    The score is flat at 1.0 as long as utilization stays under a safety
    margin below the hard ceiling (``capacity_buffer_safety_margin``, default
    15 percentage points below ``max_utilization_after_assignment``) -- extra
    headroom past that point buys no further score. Past the safe line, the
    score decays linearly to 0 exactly at the ceiling, since that is where the
    real risk of a future add overflowing the truck actually lives. This
    avoids the old formula's bias toward near-empty trucks: two routes that
    are both comfortably safe now score the same, and only a route that is
    genuinely getting full is marked down.
    """
    ceiling = config.max_utilization_after_assignment
    margin = config.capacity_buffer_safety_margin
    safe_utilization = ceiling - margin
    if ctx.utilization_after <= safe_utilization:
        value = 1.0
    else:
        value = _clamp01((ceiling - ctx.utilization_after) / margin)
    return FactorScore(
        name=FACTOR_CAPACITY_BUFFER,
        weight=config.rs_weight_capacity,
        value=value,
        detail=(
            f"{ctx.remaining_capacity_after} cases of headroom left, putting the truck at "
            f"about {ctx.utilization_after:.0%} full after this order (comfortably safe up "
            f"to {safe_utilization:.0%})"
        ),
    )


def tier_weighted_contention(window, route: Route, config: Config) -> float:
    """Sum of tier `harm` weights over the committed stops whose own window
    overlaps ``window`` -- how much adding the prospect here would crowd valued
    incumbents. An Other-tier stop barely counts; a tier-5/Perks stop counts a
    lot (see Config.tier_harm_weight)."""
    return sum(
        config.tier_harm_weight(s.customer_tier)
        for s in route.committed_stops
        if s.delivery_time_window is not None
        and overlap_minutes(window, s.delivery_time_window) > 0
    )


def slot_openness(window, route: Route, config: Config) -> float:
    """Openness of a candidate window in (0, 1]: 1 / (1 + tier-weighted
    contention). A window no committed stop shares is 1.0 (fully open); one
    shared by valued incumbents decays toward 0. The single definition of
    openness -- `slot_availability` wraps this as a weighted factor."""
    return 1.0 / (1.0 + tier_weighted_contention(window, route, config))


def slot_availability(route: Route, slot: SlotOption, config: Config) -> FactorScore:
    """Slot-level factor: how open the candidate window is (few / low-tier
    committed stops already in it), tier-weighted so we avoid harming the most
    valued customers."""
    harm = tier_weighted_contention(slot.window, route, config)
    value = slot_openness(slot.window, route, config)
    return FactorScore(
        name=FACTOR_SLOT_AVAILABILITY,
        weight=config.rs_weight_availability,
        value=value,
        detail=(
            f"tier-weighted contention {harm:.2f} from committed stops sharing this "
            f"window ({slot.committed_overlap} overlap) -> openness {value:.2f}"
        ),
    )


def _slot_window_match(
    customer: CustomerProfile, route: Route, slot: SlotOption, config: Config
) -> Optional[FactorScore]:
    """Slot-level window_match: how much THIS candidate window covers the
    customer's preferred slot. Returns None when there is no stated preference --
    in the route-slot path the factor is simply dropped (no 0.6 neutral)."""
    pref = customer.preferred_slot
    if pref is None:
        return None
    day_ok = route.day == pref.day
    pref_minutes = max(1, duration_minutes(pref.window))
    overlap = overlap_minutes(pref.window, slot.window) if day_ok else 0
    value = _clamp01(overlap / pref_minutes) if (day_ok and overlap > 0) else 0.0
    if day_ok and overlap > 0:
        detail = (
            f"covers {overlap} of the {pref_minutes} preferred minutes "
            f"on {day_label(route.day)}"
        )
    elif day_ok:
        detail = f"on {day_label(route.day)} but this window misses the preferred hours"
    else:
        detail = f"route runs {day_label(route.day)}, not the preferred {day_label(pref.day)}"
    return FactorScore(
        name=FACTOR_WINDOW_MATCH, weight=config.rs_weight_window, value=value, detail=detail
    )


def score_route_slot(
    customer: CustomerProfile,
    route: Route,
    ctx: EvalContext,
    slot: SlotOption,
    config: Config,
) -> tuple[list[FactorScore], float]:
    """Score one (route, slot) pair. Route-level factors (geo, capacity) are
    shared across that route's slots; window_match and slot_availability are
    computed for THIS specific slot. window_match is present only when the
    customer stated a preference. The total is the weighted average over
    whichever factors are active."""
    breakdown: list[FactorScore] = [
        geographic_clustering(customer, route, ctx, config),
        capacity_buffer(customer, route, ctx, config),
    ]
    wm = _slot_window_match(customer, route, slot, config)
    if wm is not None:
        breakdown.append(wm)
    breakdown.append(slot_availability(route, slot, config))

    total_weight = sum(fs.weight for fs in breakdown) or 1.0
    total = sum(fs.weighted for fs in breakdown) / total_weight
    return breakdown, round(total, 4)
