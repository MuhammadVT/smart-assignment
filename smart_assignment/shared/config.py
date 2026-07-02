"""
Workflow configuration: tunable business thresholds and scoring weights,
centralized so ops can adjust rules via environment variables (or by passing
a `Config` instance) without touching workflow logic.

[ASSUMPTION] All defaults below are reasonable starting points, NOT validated
Sysco policy. Confirm real values (capacity buffer, serviceability radius,
confidence threshold, factor weights/priorities) with operations.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# Canonical names of the three weighted scoring factors (spec step 4).
FACTOR_GEO_CLUSTERING = "geographic_clustering"
FACTOR_CAPACITY_BUFFER = "capacity_buffer"
FACTOR_WINDOW_MATCH = "window_match"


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _default_weights() -> dict[str, float]:
    # Priority order per spec: geographic clustering > capacity buffer > window match.
    return {
        FACTOR_GEO_CLUSTERING: _float_env("SMART_ASSIGNMENT_WEIGHT_GEO", 0.45),
        FACTOR_CAPACITY_BUFFER: _float_env("SMART_ASSIGNMENT_WEIGHT_CAPACITY", 0.30),
        FACTOR_WINDOW_MATCH: _float_env("SMART_ASSIGNMENT_WEIGHT_WINDOW", 0.25),
    }


@dataclass
class Config:
    """All tunable knobs for the slot-recommendation workflow."""

    # --- Hard constraints ---
    # Max fraction of rated vehicle capacity a route may be filled to AFTER
    # adding the new customer (spec: "<= 90% post-add").
    max_utilization_after_assignment: float = 0.90
    # Safety upper bound on how far a customer can be from a route's service
    # center regardless of the route's own radius.
    max_service_distance_miles: float = 25.0

    # --- Candidate identification ---
    top_n_candidate_routes: int = 3  # spec step 2: "Top N candidate routes by proximity"

    # --- Scoring ---
    factor_weights: dict[str, float] = field(default_factory=_default_weights)
    # Distance (mi) at which geographic-clustering score decays to ~0.
    cluster_reference_miles: float = 15.0
    # Score assigned to window_match when the customer stated no preference.
    window_neutral_score: float = 0.6
    # Percentage points below max_utilization_after_assignment that still
    # count as fully safe for the capacity_buffer factor (default 15pp, i.e.
    # a 90% ceiling is "safe" up to 75%). Below that line, capacity_buffer is
    # flat at 1.0 -- more headroom than that buys no extra score. Above it,
    # the score decays linearly to 0 at the ceiling itself, since that's
    # where the real risk of a future add overflowing the truck actually is.
    capacity_buffer_safety_margin: float = 0.15

    # --- Decision / escalation ---
    confidence_threshold: float = 0.70  # below this -> escalate to a human
    # Score gap between #1 and #2 that counts as "clearly separated".
    confidence_separation_ref: float = 0.15

    # --- Optional LLM reasoning layer ---
    model: str = "gemini-flash-latest"

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            max_utilization_after_assignment=_float_env("SMART_ASSIGNMENT_MAX_UTILIZATION", 0.90),
            max_service_distance_miles=_float_env("SMART_ASSIGNMENT_MAX_SERVICE_MILES", 25.0),
            top_n_candidate_routes=_int_env("SMART_ASSIGNMENT_TOP_N", 3),
            factor_weights=_default_weights(),
            cluster_reference_miles=_float_env("SMART_ASSIGNMENT_CLUSTER_REF_MILES", 15.0),
            window_neutral_score=_float_env("SMART_ASSIGNMENT_WINDOW_NEUTRAL", 0.6),
            capacity_buffer_safety_margin=_float_env(
                "SMART_ASSIGNMENT_CAPACITY_SAFETY_MARGIN", 0.15
            ),
            confidence_threshold=_float_env("SMART_ASSIGNMENT_CONFIDENCE_THRESHOLD", 0.70),
            confidence_separation_ref=_float_env("SMART_ASSIGNMENT_CONFIDENCE_SEPARATION", 0.15),
            model=os.environ.get("SMART_ASSIGNMENT_MODEL", "gemini-flash-latest"),
        )


# Convenience default used when a caller doesn't inject its own Config.
DEFAULT_CONFIG = Config.from_env()
