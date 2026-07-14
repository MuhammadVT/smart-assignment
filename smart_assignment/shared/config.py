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
from dataclasses import dataclass, field, replace
from typing import Optional

# Canonical names of the weighted scoring factors (spec step 4).
FACTOR_GEO_CLUSTERING = "geographic_clustering"
FACTOR_CAPACITY_BUFFER = "capacity_buffer"
FACTOR_WINDOW_MATCH = "window_match"
# The route-slot scoring path (Config.use_route_slot_scoring) adds a fourth,
# slot-level factor: how OPEN the candidate window is (few/low-tier committed
# stops already in it). See shared/scoring.slot_availability.
FACTOR_SLOT_AVAILABILITY = "slot_availability"

# Canonical role names for per-task model selection (see Config.for_role). Each
# LLM-using surface passes its role so the right model can be assigned to the
# right task while the LLM backend stays global.
ROLE_ROOT_AGENT = "root_agent"  # the conversational LlmAgent
ROLE_TRIAGE = "triage"  # the escalation-triage sub-agent (AgentTool)
ROLE_JUDGMENT = "judgment"  # the grounded-judgment decision call
ROLE_REASONING = "reasoning"  # the LLM-narrated reasoning trace (LLMReasoner)
ROLE_SLOTPICK = "slotpick"  # the grounded slot selection over a route's candidate menu
ROLE_ADDRESS_RESOLVE = "address_resolve"  # grounded pick among geocoder address candidates

# role -> env var that overrides that role's model. A role whose env var is
# unset uses the global `model` / `sage_model`, so behavior is unchanged.
_ROLE_MODEL_ENV = {
    ROLE_ROOT_AGENT: "SMART_ASSIGNMENT_MODEL_ROOT_AGENT",
    ROLE_TRIAGE: "SMART_ASSIGNMENT_MODEL_TRIAGE",
    ROLE_JUDGMENT: "SMART_ASSIGNMENT_MODEL_JUDGMENT",
    ROLE_REASONING: "SMART_ASSIGNMENT_MODEL_REASONING",
    ROLE_SLOTPICK: "SMART_ASSIGNMENT_MODEL_SLOTPICK",
    ROLE_ADDRESS_RESOLVE: "SMART_ASSIGNMENT_MODEL_ADDRESS_RESOLVE",
}


def _role_models_from_env() -> dict[str, str]:
    models: dict[str, str] = {}
    for role, env in _ROLE_MODEL_ENV.items():
        raw = os.environ.get(env)
        if raw and raw.strip():
            models[role] = raw.strip()
    return models


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
    # How many of a route's nearest committed stops are considered as the
    # prospect's "adjacent stops" when placing a slot.
    slot_neighbor_count: int = 3
    # Optional cap (miles): committed stops farther than this are ignored. None
    # means no cap -- every committed stop is eligible, ranked by distance.
    slot_neighbor_max_miles: Optional[float] = None
    # Length of the delivery window we RECOMMEND to the prospect, in minutes.
    # Every candidate window is this long, CENTERED on its interpolated time.
    slot_window_minutes: int = 180
    # The nearest stops are grouped into temporal clusters (e.g. a morning
    # neighborhood vs. an afternoon one): consecutive reference times more than
    # this many minutes apart start a new cluster, and each cluster yields one
    # candidate slot centered on its proximity-weighted midpoint.
    slot_cluster_gap_minutes: int = 180
    # Top-N candidate slots kept per route (the menu handed to the recommender /
    # a future LLM). Any candidate that overlaps a stated customer preference is
    # always kept, even if it falls outside the top-N by quality.
    slot_candidate_count: int = 3
    # Blend weights for scoring/ranking candidate slots. Quality (used for the
    # top-N cut and the no-preference pick) blends fit + low-contention; when a
    # preference is stated, its overlap adds a third term. Need not sum to 1 --
    # they are normalized over whichever terms are active.
    slot_weight_fit: float = 0.5  # proximity-weight share of the slot's cluster
    slot_weight_contention: float = 0.2  # emptier (less committed overlap) is better
    slot_weight_preference: float = 0.3  # overlap with the customer's stated slot
    # When True, the FINAL recommended slot for the chosen route is picked by an
    # LLM reasoning over that route's candidate menu (see the `slotpick`
    # package), constrained to the enumerated candidates and grounded in their
    # facts -- instead of the deterministic blend above. It never changes the
    # route or the score, only which candidate slot is presented, and falls back
    # to the deterministic pick on any failure. Off by default.
    use_grounded_slot_selection: bool = False

    # --- Route-slot scoring (optional; supersedes route-only scoring) ---
    # When True, the decision unit becomes the (route, slot) PAIR: every
    # candidate slot on every feasible route is scored separately, so slot
    # availability influences which ROUTE wins -- not just which slot within an
    # already-chosen route (see the `routeslot` package). geo/capacity are
    # route-level (shared across a route's slots); window_match and
    # slot_availability are slot-level. When on, window_match is dropped entirely
    # for a prospect with no stated preference (instead of the 0.6 neutral), and
    # the grounded route-slot decision absorbs the separate slotpick pass. Off
    # reproduces the prior route-only behavior exactly.
    use_route_slot_scoring: bool = False
    # Route-slot factor weights (kept SEPARATE from factor_weights so the legacy
    # route-only path is byte-identical when the flag is off). Normalized over
    # whichever factors are active (window_match only when a preference exists).
    rs_weight_geo: float = 0.35
    rs_weight_capacity: float = 0.25
    rs_weight_window: float = 0.20
    rs_weight_availability: float = 0.20
    # Slot-openness "harm" weights: how costly it is to add the prospect to a
    # window already claimed by a committed stop of each tier. Higher = protect
    # more. Ordering per ops: tier 5 / Perks (most valued) > tier 4 > the prospect
    # itself > Other (lowest). So crowding an Other-tier stop is nearly free,
    # while crowding a tier-5/Perks stop is heavily penalized. openness =
    # 1 / (1 + sum of harm weights over overlapping committed stops).
    slot_tier_harm_high: float = 1.0  # tier "5" / "Perks"
    slot_tier_harm_mid: float = 0.6  # tier "4"
    slot_tier_harm_low: float = 0.1  # "Other"
    slot_tier_harm_unknown: float = 0.4  # tier not known (missing in data)
    # Auto-assign bar for the route-slot path (the chosen route-slot's own total
    # must meet it, else escalate). Deliberately a touch LOWER than the legacy
    # total_score_threshold: the new composition drops the 0.6 window neutral and
    # adds an availability term, shifting the score distribution, and ops asked to
    # err slightly toward recommending. See routeslot/decide.py.
    route_slot_score_threshold: float = 0.55

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

    # --- Address resolution (optional; grounded typo/ambiguity correction) ---
    # When True, if the geocoder can't resolve the prospect's address, the agent
    # can ask a suggest-capable geocoder for candidate matches and let an LLM
    # pick the closest one (constrained to that enumerated set, grounded +
    # verified -- see the `address_resolve` package) for the USER to confirm; it
    # never invents an address. Default ON (ops asked for it): a typo/ambiguous
    # address becomes a confirmable suggestion instead of a dead-end. On any
    # failure -- feature unavailable, no candidate found, LLM/verify error -- it
    # falls back to today's "ask the customer to double-check it." Turning it off
    # reproduces that prior behavior exactly.
    use_address_resolution: bool = True

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
    # Optional per-role model overrides (role -> model name; see the ROLE_*
    # constants and for_role). A role absent here uses the global model above,
    # so the default behavior is unchanged. Lets you assign a cheaper/faster
    # model to a lightweight task (e.g. triage or reasoning narration) and a
    # stronger one to the decision. The override value must match the ACTIVE
    # backend's naming (a Sage-prefixed id under sage; a bare/litellm name under
    # standard); the backend itself stays global.
    role_models: dict[str, str] = field(default_factory=dict)

    def tier_harm_weight(self, tier: Optional[str]) -> float:
        """Harm weight for crowding a committed stop of the given Sysco tier --
        how much to protect it when scoring slot openness. Unknown/absent tiers
        get the neutral fallback so the metric degrades gracefully where tier
        data is missing (mock/phase-A routes)."""
        key = (tier or "").strip().lower()
        if key in ("5", "perks"):
            return self.slot_tier_harm_high
        if key == "4":
            return self.slot_tier_harm_mid
        if key == "other":
            return self.slot_tier_harm_low
        return self.slot_tier_harm_unknown

    def for_role(self, role: str) -> "Config":
        """A copy of this config with the active model field overridden by the
        per-role model, if one is configured for ``role``; otherwise ``self``.

        Overrides ``sage_model`` under the sage backend and ``model`` otherwise,
        so the same override string is applied to whichever field
        ``shared.llm.get_llm`` / ``generate_text`` actually read."""
        override = self.role_models.get(role)
        if not override:
            return self
        if self.llm_backend == "sage":
            return replace(self, sage_model=override)
        return replace(self, model=override)

    def resolved_model(self, role: str) -> str:
        """The effective model name a given role will actually use (handy for
        logging and tests)."""
        scoped = self.for_role(role)
        return scoped.sage_model if scoped.llm_backend == "sage" else scoped.model

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
            slot_cluster_gap_minutes=_int_env("SMART_ASSIGNMENT_SLOT_CLUSTER_GAP", 180),
            slot_candidate_count=_int_env("SMART_ASSIGNMENT_SLOT_CANDIDATES", 3),
            slot_weight_fit=_float_env("SMART_ASSIGNMENT_SLOT_WEIGHT_FIT", 0.5),
            slot_weight_contention=_float_env("SMART_ASSIGNMENT_SLOT_WEIGHT_CONTENTION", 0.2),
            slot_weight_preference=_float_env("SMART_ASSIGNMENT_SLOT_WEIGHT_PREFERENCE", 0.3),
            use_grounded_slot_selection=_bool_env(
                "SMART_ASSIGNMENT_USE_GROUNDED_SLOT_SELECTION", False
            ),
            use_route_slot_scoring=_bool_env("SMART_ASSIGNMENT_USE_ROUTE_SLOT_SCORING", False),
            rs_weight_geo=_float_env("SMART_ASSIGNMENT_RS_WEIGHT_GEO", 0.35),
            rs_weight_capacity=_float_env("SMART_ASSIGNMENT_RS_WEIGHT_CAPACITY", 0.25),
            rs_weight_window=_float_env("SMART_ASSIGNMENT_RS_WEIGHT_WINDOW", 0.20),
            rs_weight_availability=_float_env("SMART_ASSIGNMENT_RS_WEIGHT_AVAILABILITY", 0.20),
            slot_tier_harm_high=_float_env("SMART_ASSIGNMENT_SLOT_HARM_HIGH", 1.0),
            slot_tier_harm_mid=_float_env("SMART_ASSIGNMENT_SLOT_HARM_MID", 0.6),
            slot_tier_harm_low=_float_env("SMART_ASSIGNMENT_SLOT_HARM_LOW", 0.1),
            slot_tier_harm_unknown=_float_env("SMART_ASSIGNMENT_SLOT_HARM_UNKNOWN", 0.4),
            route_slot_score_threshold=_float_env(
                "SMART_ASSIGNMENT_ROUTE_SLOT_SCORE_THRESHOLD", 0.55
            ),
            total_score_threshold=_float_env("SMART_ASSIGNMENT_TOTAL_SCORE_THRESHOLD", 0.60),
            use_grounded_judgment=_bool_env("SMART_ASSIGNMENT_USE_GROUNDED_JUDGMENT", False),
            judgment_sample_count=_int_env("SMART_ASSIGNMENT_JUDGMENT_SAMPLE_COUNT", 3),
            judgment_consensus=os.environ.get("SMART_ASSIGNMENT_JUDGMENT_CONSENSUS", "unanimous")
            .strip()
            .lower(),
            judgment_retry_on_low_confidence_recommend=_bool_env(
                "SMART_ASSIGNMENT_JUDGMENT_RETRY_ON_LOW_CONFIDENCE", True
            ),
            use_address_resolution=_bool_env("SMART_ASSIGNMENT_USE_ADDRESS_RESOLUTION", True),
            use_escalation_triage=_bool_env("SMART_ASSIGNMENT_USE_ESCALATION_TRIAGE", True),
            llm_backend=os.environ.get("SMART_ASSIGNMENT_LLM_BACKEND", "sage"),
            model=os.environ.get("SMART_ASSIGNMENT_MODEL", "gemini-2.5-flash"),
            sage_model=os.environ.get("SMART_ASSIGNMENT_SAGE_MODEL", "sage-gemini-2.5-flash"),
            role_models=_role_models_from_env(),
        )


# Convenience default used when a caller doesn't inject its own Config.
DEFAULT_CONFIG = Config.from_env()
