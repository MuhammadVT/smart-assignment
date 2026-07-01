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
  2. capacity_buffer       — headroom left after adding this customer
  3. window_match          — how well the route's window fits the preference
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
from smart_assignment.shared.timeutils import duration_minutes

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
    """More remaining headroom after the add -> higher score (more resilient)."""
    value = _clamp01(ctx.remaining_capacity_after / route.vehicle_capacity_cases)
    return FactorScore(
        name=FACTOR_CAPACITY_BUFFER,
        weight=config.factor_weights[FACTOR_CAPACITY_BUFFER],
        value=value,
        detail=(
            f"{ctx.remaining_capacity_after} cases headroom "
            f"({ctx.utilization_after:.0%} utilized post-add)"
        ),
    )


def window_match(
    customer: CustomerProfile, route: Route, ctx: EvalContext, config: Config
) -> FactorScore:
    """
    How well the route matches the customer's preferred **slot** (day + time).

    The preferred slot always carries a day of week, so the score gives equal
    weight to the day matching and to the time-of-day overlap:

        0.5 * (route.day == preferred.day) + 0.5 * (overlap / preferred_minutes)

    A route on the right day covering the whole window scores 1.0; the right day
    but wrong time, or the wrong day but overlapping time, scores 0.5; neither
    scores 0.0. With no stated preference, a neutral score is used.
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
    value = 0.5 * (1.0 if day_ok else 0.0) + 0.5 * time_frac
    day_note = f"{route.day.value}={'✓' if day_ok else '✗'} vs pref {slot.day.value}"
    return FactorScore(
        name=FACTOR_WINDOW_MATCH,
        weight=weight,
        value=value,
        detail=(
            f"day {day_note}; {ctx.window_overlap_minutes}/{pref_minutes} min of "
            f"preferred time covered"
        ),
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
