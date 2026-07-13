"""
Weighted multi-factor scoring & ranking (spec step 4).

Like `constraints.py`, this is deliberately modular: each factor is a small
pure function registered in `SCORING_FACTORS`, each returns a normalized
0.0-1.0 value plus a human-readable `detail`. The final score is the
weight-dot-value across factors, with weights (and thus priorities) living in
`Config.factor_weights`. Add a factor by writing one function and giving it a
weight — nothing else changes.

Factors, in priority order (per spec):
  1. geographic_clustering — tightness of fit with existing stops on the route
  2. capacity_buffer       — stays flat once safely under the capacity
                             ceiling; only decays as utilization approaches it
  3. window_match          — how well the route's day and time fit the
                             customer's preferred slot
"""

from __future__ import annotations

from typing import Callable, Optional

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

FactorFn = Callable[[CustomerProfile, Route, EvalContext, Config], FactorScore]


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def geographic_clustering(
    customer: CustomerProfile, route: Route, ctx: EvalContext, config: Config
) -> FactorScore:
    """Closer to the route's existing cluster of stops -> higher score."""
    value = _clamp01(1.0 - ctx.avg_stop_distance_miles / config.cluster_reference_miles)
    return FactorScore(
        name=FACTOR_GEO_CLUSTERING,
        weight=config.factor_weights[FACTOR_GEO_CLUSTERING],
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
        weight=config.factor_weights[FACTOR_CAPACITY_BUFFER],
        value=value,
        detail=(
            f"{ctx.remaining_capacity_after} cases of headroom left, putting the truck at "
            f"about {ctx.utilization_after:.0%} full after this order (comfortably safe up "
            f"to {safe_utilization:.0%})"
        ),
    )


def window_match(
    customer: CustomerProfile, route: Route, ctx: EvalContext, config: Config
) -> FactorScore:
    """
    How well the route matches the customer's preferred **slot** (day + time).

    The day of week is a gate, not a source of partial credit: a route only
    earns any slot-match score once it lands on the customer's preferred day.
    From there, the score is simply how much of the preferred window the
    route actually covers:

        0.0                                   if route.day != preferred.day
        0.0                                   if the day matches but there is no time overlap at all
        overlap_minutes / preferred_minutes    otherwise

    A route on the wrong day, or on the right day with zero time overlap, is
    not a real match and scores 0 -- it shouldn't collect credit just for
    getting half of the slot right. With no stated preference, a neutral
    score is used.
    """
    slot = customer.preferred_slot
    weight = config.factor_weights[FACTOR_WINDOW_MATCH]
    if slot is None:
        return FactorScore(
            name=FACTOR_WINDOW_MATCH,
            weight=weight,
            value=config.window_neutral_score,
            detail="no stated preference (neutral score)",
        )
    day_ok = route.day == slot.day
    pref_minutes = max(1, duration_minutes(slot.window))
    time_frac = _clamp01(ctx.window_overlap_minutes / pref_minutes)
    value = time_frac if (day_ok and time_frac > 0) else 0.0
    if day_ok and time_frac > 0:
        detail = (
            f"the route runs on {day_label(route.day)}, matching the customer's preference, "
            f"and covers {ctx.window_overlap_minutes} of the {pref_minutes} minutes of their "
            f"preferred time"
        )
    elif day_ok:
        detail = (
            f"the route runs on {day_label(route.day)}, matching the customer's preference, "
            f"but its time window doesn't overlap their preferred hours at all, so this "
            f"doesn't count as a real match"
        )
    else:
        detail = (
            f"the route runs on {day_label(route.day)} rather than the "
            f"{day_label(slot.day)} the customer asked for, so this doesn't count as a "
            f"match regardless of the time overlap"
        )
    return FactorScore(
        name=FACTOR_WINDOW_MATCH,
        weight=weight,
        value=value,
        detail=detail,
    )


# Registry — the ordered set of scoring factors.
SCORING_FACTORS: list[FactorFn] = [
    geographic_clustering,
    capacity_buffer,
    window_match,
]


# ---------------------------------------------------------------------------
# Route-slot scoring (Config.use_route_slot_scoring)
#
# The decision unit becomes the (route, slot) PAIR. geo and capacity are
# route-level (shared across a route's slots and reused from the factors above);
# window_match and slot_availability are slot-level. This lets slot openness
# influence which ROUTE wins, not just which slot within an already-picked route.
# ---------------------------------------------------------------------------


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
    shared by valued incumbents decays toward 0."""
    return 1.0 / (1.0 + tier_weighted_contention(window, route, config))


def slot_availability(route: Route, slot: SlotOption, config: Config) -> FactorScore:
    """Slot-level factor: how open the candidate window is (few / low-tier
    committed stops already in it), tier-weighted so we avoid harming the most
    valued customers."""
    harm = tier_weighted_contention(slot.window, route, config)
    value = 1.0 / (1.0 + harm)
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
    """Score one (route, slot) pair. Route-level factors (geo, capacity) reuse
    the same value math as the route-only path but carry the route-slot weights;
    window_match and slot_availability are computed for THIS specific slot.
    window_match is present only when the customer stated a preference. The total
    is the weighted average over whichever factors are active."""
    geo = geographic_clustering(customer, route, ctx, config)
    cap = capacity_buffer(customer, route, ctx, config)
    breakdown: list[FactorScore] = [
        FactorScore(geo.name, config.rs_weight_geo, geo.value, geo.detail),
        FactorScore(cap.name, config.rs_weight_capacity, cap.value, cap.detail),
    ]
    wm = _slot_window_match(customer, route, slot, config)
    if wm is not None:
        breakdown.append(wm)
    breakdown.append(slot_availability(route, slot, config))

    total_weight = sum(fs.weight for fs in breakdown) or 1.0
    total = sum(fs.weighted for fs in breakdown) / total_weight
    return breakdown, round(total, 4)


def score_candidate(
    customer: CustomerProfile,
    route: Route,
    ctx: EvalContext,
    config: Config,
    factors: Optional[list[FactorFn]] = None,
) -> tuple[list[FactorScore], float]:
    """Return (per-factor breakdown, total weighted score in 0.0-1.0)."""
    fns = factors if factors is not None else SCORING_FACTORS
    breakdown = [fn(customer, route, ctx, config) for fn in fns]
    total_weight = sum(fs.weight for fs in breakdown) or 1.0
    total = sum(fs.weighted for fs in breakdown) / total_weight
    return breakdown, round(total, 4)
