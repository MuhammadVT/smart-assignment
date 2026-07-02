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
    FACTOR_WINDOW_MATCH,
    Config,
)
from smart_assignment.shared.constraints import EvalContext
from smart_assignment.shared.models import CustomerProfile, FactorScore, Route
from smart_assignment.shared.timeutils import day_label, duration_minutes

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
