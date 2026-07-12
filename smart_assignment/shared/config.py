"""
Workflow configuration: tunable business thresholds and scoring weights,
centralized so ops can adjust rules via environment variables (or by passing
a `Config` instance) without touching workflow logic.

[ASSUMPTION] All defaults below are reasonable starting points, NOT validated
Sysco policy. Confirm real values (capacity buffer, serviceability radius,
total-score threshold, factor weights/priorities) with operations.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

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


def _opt_float_env(name: str) -> Optional[float]:
    """A float env var that is genuinely optional: unset or blank -> None."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


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

    # --- Slot selection (location-aware delivery-window pick) ---
    # How many of a route's nearest committed stops get to "vote" on which
    # offered window a prospect belongs in ("a slot between adjacent stops").
    slot_neighbor_count: int = 3
    # Optional cap (miles): committed stops farther than this don't vote. None
    # means no cap -- every committed stop is eligible, ranked by distance.
    slot_neighbor_max_miles: Optional[float] = None
    # Length of the delivery window we actually RECOMMEND to the prospect, in
    # minutes. The route's historical windows can be any length; the pick is
    # normalized to this standard duration (default 180 = 3 hours), anchored at
    # the chosen window's start. Change this one value to widen/narrow every
    # recommended slot.
    slot_window_minutes: int = 180

    # --- Decision / escalation ---
    # The winning route's own total_score (see shared/scoring.score_candidate)
    # must meet this bar to auto-assign; below it, the agent escalates to a
    # human. A route's own merit is judged on its own -- this is intentionally
    # NOT a function of how close a runner-up scored (see reasoning.py).
    #
    # NOTE: this gate applies to the default *weighted-sum* decision path only.
    # When `use_grounded_judgment` is on, the LLM makes the recommend/escalate
    # call itself (see the `judgment` package) and this threshold is not used as
    # a gate -- the weighted score is demoted to a reference-only fact.
    total_score_threshold: float = 0.60

    # --- Grounded LLM judgment (optional, opt-in) ---
    # When True, the recommend/escalate decision is made by an LLM reasoning
    # over a structured *evidence packet* of the raw per-candidate facts
    # (see the `judgment` package), instead of by the fixed weighted-sum +
    # `total_score_threshold` gate. Hard constraints (constraints.py) still run
    # first and remain the only thing that can eliminate a candidate, so the
    # LLM can never pick an over-capacity or out-of-area route. Defaults to
    # False so the existing deterministic path is unchanged unless enabled.
    use_grounded_judgment: bool = False
    # Number of independent judgment samples to draw for an "escalation-side"
    # case (first sample is not a confident recommendation). k=1 disables
    # resampling. Confident recommendations always ship on a single call.
    judgment_sample_count: int = 3
    # How the k samples' decisions are combined to clear an escalation-side
    # case back to a recommendation: "unanimous" (default, precautionary --
    # every sample must recommend) or "majority".
    judgment_consensus: str = "unanimous"
    # Whether a first sample that recommends a route but with LOW confidence is
    # treated as "escalation-side" (True -> resample to confirm; the safe
    # default) or shipped as-is (False -> a pick is a pick). A hard ESCALATE
    # always resamples regardless of this flag.
    judgment_retry_on_low_confidence_recommend: bool = True

    # --- Escalation triage (optional sub-agent) ---
    # When True, root_agent exposes an `escalation_triage` AgentTool (see the
    # `triage` package) and, on any escalation, calls it to compose a specialist
    # brief (root cause + concrete remediation options + a question) before the
    # human handoff. It runs downstream of the deterministic decision and never
    # changes the route, score, or decision, so auditability is unaffected;
    # turning it off just reverts to a bare request_input handoff.
    use_escalation_triage: bool = True

    # --- LLM backend ---
    # "sage"     → enterprise-governed SageLlmRegistry (requires SAGE_CLIENT_ID,
    #              SAGE_CLIENT_SECRET, SAGE_ENVIRONMENT to be set).
    # "standard" → `model` below, used directly by Google ADK / genai
    #              (requires GOOGLE_API_KEY or Vertex credentials) -- unless
    #              it's a litellm-style "<provider>/<model>" string (e.g.
    #              "openai/gpt-4o-mini"), in which case shared/llm.py routes
    #              it through litellm instead (see that module's docstring).
    llm_backend: str = "sage"
    # Model name used when llm_backend == "standard" -- a bare Gemini name,
    # or a "<provider>/<model>" litellm string for any other provider.
    model: str = "gemini-2.5-flash"
    # Model name used when llm_backend == "sage" (Sage-prefixed identifier).
    sage_model: str = "sage-gemini-2.5-flash"

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
            slot_neighbor_count=_int_env("SMART_ASSIGNMENT_SLOT_NEIGHBORS", 3),
            slot_neighbor_max_miles=_opt_float_env("SMART_ASSIGNMENT_SLOT_NEIGHBOR_MAX_MILES"),
            slot_window_minutes=_int_env("SMART_ASSIGNMENT_SLOT_WINDOW_MINUTES", 180),
            total_score_threshold=_float_env("SMART_ASSIGNMENT_TOTAL_SCORE_THRESHOLD", 0.60),
            use_grounded_judgment=_bool_env("SMART_ASSIGNMENT_USE_GROUNDED_JUDGMENT", False),
            judgment_sample_count=_int_env("SMART_ASSIGNMENT_JUDGMENT_SAMPLE_COUNT", 3),
            judgment_consensus=os.environ.get("SMART_ASSIGNMENT_JUDGMENT_CONSENSUS", "unanimous")
            .strip()
            .lower(),
            judgment_retry_on_low_confidence_recommend=_bool_env(
                "SMART_ASSIGNMENT_JUDGMENT_RETRY_ON_LOW_CONFIDENCE", True
            ),
            use_escalation_triage=_bool_env("SMART_ASSIGNMENT_USE_ESCALATION_TRIAGE", True),
            llm_backend=os.environ.get("SMART_ASSIGNMENT_LLM_BACKEND", "sage"),
            model=os.environ.get("SMART_ASSIGNMENT_MODEL", "gemini-2.5-flash"),
            sage_model=os.environ.get("SMART_ASSIGNMENT_SAGE_MODEL", "sage-gemini-2.5-flash"),
        )


# Convenience default used when a caller doesn't inject its own Config.
DEFAULT_CONFIG = Config.from_env()
