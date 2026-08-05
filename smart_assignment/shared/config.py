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

# Canonical names of the scoring factors. The decision unit is the (route, slot)
# PAIR: geo/capacity are route-level (shared across a route's slots), while
# window_match and slot_availability are slot-level. window_match is present only
# when the customer stated a preference -- there is no neutral stand-in.
FACTOR_GEO_CLUSTERING = "geographic_clustering"
FACTOR_CAPACITY_BUFFER = "capacity_buffer"
FACTOR_WINDOW_MATCH = "window_match"
# How OPEN the candidate window is (few/low-tier committed stops already in it).
# See shared/scoring.slot_availability.
FACTOR_SLOT_AVAILABILITY = "slot_availability"

# Canonical role names for per-task model selection (see Config.for_role). Each
# LLM-using surface passes its role so the right model can be assigned to the
# right task while the LLM backend stays global.
ROLE_ROOT_AGENT = "root_agent"  # the conversational LlmAgent
ROLE_TRIAGE = "triage"  # the escalation-triage sub-agent (AgentTool)
ROLE_JUDGMENT = "judgment"  # the grounded route-slot decision call
ROLE_ADDRESS_RESOLVE = "address_resolve"  # grounded pick among geocoder address candidates
# Not a product decision-layer role like the others above -- this is
# eval/test_quality.py's DeepEval G-Eval judge (Phase 3a, advisory, outside the
# product decision path). Included here anyway so it gets the same per-role
# override capability as everything else (e.g. a stronger judge model than the
# app's own operational model), via the same Config.for_role/generate_text seam
# eval/deepeval_llm.py's SmartAssignmentDeepEvalLLM reuses rather than
# reinventing a separate judge-model resolution path.
ROLE_QUALITY_JUDGE = "quality_judge"

# role -> env var that overrides that role's model. A role whose env var is
# unset uses the global `model` / `sage_model`, so behavior is unchanged.
_ROLE_MODEL_ENV = {
    ROLE_ROOT_AGENT: "SMART_ASSIGNMENT_MODEL_ROOT_AGENT",
    ROLE_TRIAGE: "SMART_ASSIGNMENT_MODEL_TRIAGE",
    ROLE_JUDGMENT: "SMART_ASSIGNMENT_MODEL_JUDGMENT",
    ROLE_ADDRESS_RESOLVE: "SMART_ASSIGNMENT_MODEL_ADDRESS_RESOLVE",
    ROLE_QUALITY_JUDGE: "SMART_ASSIGNMENT_MODEL_QUALITY_JUDGE",
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
    # When True (default), and the customer stated a preferred DAY that none of
    # the Top-N nearest routes runs on, the nearest route that DOES run on that
    # day is kept as an ADDITIONAL candidate -- however far down the proximity
    # ranking it sits. Without this, a purely distance-based cut can eliminate
    # the preferred day before any scoring happens, so `window_match` is 0 for
    # every option and a stated preference can only ever LOWER the totals (it
    # keeps its weight in the denominator whether or not it is satisfiable).
    #
    # This guarantees the preference is CONSIDERED, never that it wins: the added
    # route still faces the hard constraints and is scored like any other, so it
    # can be rejected or out-scored. Exactly one route is added, so the candidate
    # set is at most Top-N + 1.
    #
    # The search is capped at the service-area limit
    # (max_service_distance_miles, tightened by a route's own service_radius_miles
    # when it declares one -- see constraints.service_distance_limit): a route
    # beyond it would provably fail geographic_serviceability, so adding it would
    # only put a guaranteed-rejected route in front of a specialist. The cap is
    # on DISTANCE only -- an in-range preferred-day route that is too full is
    # still added and shows as rejected on capacity, which a human can act on.
    #
    # Off reproduces the prior behavior exactly: candidates are the N nearest,
    # day-blind.
    use_preferred_day_candidate: bool = True

    # --- Scoring ---
    # Distance (mi) at which geographic-clustering score decays to ~0.
    cluster_reference_miles: float = 15.0
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
    # Weights for ranking candidate slots into the top-N MENU: location fit +
    # low contention, normalized over the two. Preference deliberately plays no
    # part here -- a preference-overlapping candidate is kept unconditionally
    # (see select_candidate_slots), and preference is then weighed against slot
    # openness as the day-gated `window_match` factor when the (route, slot)
    # pair is scored. Need not sum to 1.
    slot_weight_fit: float = 0.5  # proximity-weight share of the slot's cluster
    slot_weight_contention: float = 0.2  # emptier (less committed overlap) is better

    # --- Route-slot scoring ---
    # The decision unit is the (route, slot) PAIR: every candidate slot on every
    # feasible route is scored separately, so slot availability influences which
    # ROUTE wins -- not just which slot within an already-chosen route (see the
    # `routeslot` package). Normalized over whichever factors are active
    # (window_match only when a preference exists).
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
    # Auto-assign bar (the chosen route-slot's own total must meet it, else
    # escalate). Set a touch low deliberately: the composition carries no window
    # neutral and adds an availability term, and ops asked to err slightly toward
    # recommending. See routeslot/decide.py.
    route_slot_score_threshold: float = 0.55
    # When True (default), the recommend-vs-escalate call on the route-slot path is
    # made by the LLM ITSELF over ALL feasible route-slots -- not gated by the
    # route_slot_score_threshold bar. The model reasons over the options (grounded +
    # verified, same as every other LLM call) and either RECOMMENDs the best one or
    # ESCALATEs when it judges none good enough; an escalation-side/low-confidence
    # call is resampled `judgment_sample_count` times and combined by
    # `judgment_consensus` before it may auto-assign. The threshold is DEMOTED to a
    # reference fact in the evidence packet, and the deterministic threshold decision
    # remains the FALLBACK on any LLM/verify/backend failure -- so it is never worse
    # than the bar-gated baseline. When False, the route-slot path uses the prior
    # logic exactly: the threshold gates recommend-vs-escalate and the LLM only picks
    # among the above-bar options. Non-feasible cases are always a deterministic
    # escalation regardless of this flag.
    use_grounded_route_slot_escalation: bool = True

    # --- Grounded LLM reasoning over the route-slot menu (optional, opt-in) ---
    # When True, the route-slot PICK is made by an LLM reasoning over the
    # deterministically enumerated (route, slot) options -- it chooses by index
    # from that set and every fact it cites is verified against the packet (see
    # the `routeslot` package). Hard constraints (constraints.py) still run first
    # and remain the only thing that can eliminate a candidate, so the LLM can
    # never pick an over-capacity or out-of-area route. Off by default, in which
    # case the highest-scoring route-slot is taken deterministically.
    #
    # NOTE: this gates the PICK. Whether the LLM also makes the
    # recommend-vs-escalate call is `use_grounded_route_slot_escalation` above.
    use_grounded_route_slot_pick: bool = False
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

    # --- Session memory (optional cross-prospect recall; off by default) ---
    # When True, the chat web app remembers free-form facts stated in EARLIER
    # prospects of the same browser session -- e.g. an aside the user made before
    # the conversation rotated to a new address. Purely additive: the deterministic
    # pipeline, the per-prospect rotation, and the decision are all unchanged; the
    # model simply gains RECALL of the prior transcript, it does not gain a new
    # actionable value. Mechanics (see webapp/llm_chat.py and agent.py): the app
    # wires an ADK ``InMemoryMemoryService`` into the Runner, folds a concluding
    # prospect's transcript into memory when it rotates to the next one, and
    # root_agent gains ADK's ``preload_memory`` tool, which auto-injects relevant
    # past-conversation snippets into each turn. Memory is scoped per browser
    # session (the browser session_id becomes the ADK user_id when this is on), so
    # one browser's facts never leak into another's. Off by default; flag-off wires
    # no memory service, adds no tool, and keeps the fixed webapp user_id -- i.e.
    # reproduces today's behavior exactly.
    use_session_memory: bool = False

    # --- LLM backend ---
    # "sage"     → enterprise-governed SageLlmRegistry (requires SAGE_CLIENT_ID,
    #              SAGE_CLIENT_SECRET, SAGE_ENVIRONMENT to be set) -- unless
    #              `use_sage_gateway` is on (see below).
    # "standard" → `model` below, used directly by Google ADK / genai
    #              (requires GOOGLE_API_KEY or Vertex credentials) -- unless
    #              it's a litellm-style "<provider>/<model>" string (e.g.
    #              "openai/gpt-4o-mini"), in which case shared/llm.py routes
    #              it through litellm instead (see that module's docstring).
    llm_backend: str = "sage"
    # Model name used when llm_backend == "standard" -- a bare Gemini name,
    # or a "<provider>/<model>" litellm string for any other provider.
    model: str = "gemini-3.5-flash"
    # Model name used when llm_backend == "sage" (Sage-prefixed identifier,
    # unless `use_sage_gateway` is on -- see below).
    sage_model: str = "sage-gemini-2.5-flash"
    # When True (and llm_backend == "sage"), the sage call is routed through
    # Sysco's enterprise LLM Gateway (the Sage SDK's `GatewayLlm`, an
    # OpenAI-compatible litellm proxy with OAuth2 token injection) instead of
    # SageLlmRegistry/SageLiteLlm's direct call to one registered SAGE agent
    # -- see shared/llm.py's module docstring. `sage_model` then names a
    # gateway-exposed model id (e.g. "gpt-4o"), not a sage-* agent name.
    # Requires LLM_GATEWAY_CLIENT_ID/LLM_GATEWAY_CLIENT_SECRET
    # (LLM_GATEWAY_ENV optional, defaults to "qa" in the SDK). Off by default
    # so the existing direct-agent path is unchanged.
    use_sage_gateway: bool = False
    # Optional per-role model overrides (role -> model name; see the ROLE_*
    # constants and for_role). A role absent here uses the global model above,
    # so the default behavior is unchanged. Lets you assign a cheaper/faster
    # model to a lightweight task (e.g. triage or reasoning narration) and a
    # stronger one to the decision. The override value must match the ACTIVE
    # backend's naming (a Sage-prefixed id under sage; a bare/litellm name under
    # standard); the backend itself stays global.
    role_models: dict[str, str] = field(default_factory=dict)

    # --- Backend compatibility (on by default) ---
    # How many times a single sage request may be ATTEMPTED before it fails, the
    # first try included (so 1 disables retrying and reproduces prior behavior
    # exactly). Applied by passing litellm's own `num_retries` (= attempts - 1)
    # through ADK's LiteLlm; litellm retries an APIConnectionError -- which is what
    # a sage request timeout surfaces as -- immediately, with no backoff.
    #
    # This exists because ADK's eval harness *intends* to retry (it registers a
    # plugin setting HttpRetryOptions(attempts=7)) but that is a google-genai
    # construct, and ADK's LiteLlm never reads it -- so on the sage path a request
    # that times out is simply lost, taking the whole agent turn with it. The
    # slowest call in this system (the triage agent writing its brief) sits close
    # enough to the timeout that a single transient spike kills a turn that would
    # otherwise succeed on a second try.
    #
    # Bounded on purpose: with SAGE_TIMEOUT=40 the worst case is 2 x 40s for one
    # call. If a second attempt also times out, the backend is genuinely unwell and
    # failing is the honest outcome.
    sage_request_attempts: int = 2
    # When True, a tool call whose arguments arrive wrapped in a JSON array -- an
    # intermittent sage-backend quirk that otherwise raises inside ADK and kills the
    # entire turn -- is repaired by taking the single argument object out of that
    # array (see _install_litellm_tool_args_repair in shared/llm.py).
    #
    # ON by default, unlike the opt-in flags above, because it provably cannot change
    # a healthy call: a well-formed arguments object is returned untouched, so the
    # repair only ever fires on a payload that would otherwise crash, and any shape it
    # cannot read with certainty is passed through unchanged so that failure stays
    # loud. Set to False for ADK's raw behavior.
    repair_tool_call_args: bool = True

    # --- Diagnostics (opt-in, off by default) ---
    # When True, wrap the Sage SDK's response extractor so that whenever it would
    # return its generic "Something went wrong" sentinel -- masking the model's real
    # reply -- the true agent_response (e.g. a tool/function call the grounded path
    # never offered) is logged. Purely diagnostic: it changes no decision, value, or
    # fallback; it only makes an opaque sage failure legible. Off by default.
    debug_sage_raw_response: bool = False

    # --- Observability (opt-in, off by default) ---
    # When True, LLM calls are wrapped in an OpenTelemetry span and exported to a
    # configured OTLP backend (e.g. a self-hosted Langfuse instance) -- see
    # shared/tracing.py. Purely additive: tracing observes, it never changes a
    # value a decision layer acts on, and every failure path (SDK missing, no
    # exporter, backend unreachable) degrades to a silent no-op. Off by default,
    # and flag-off imports no OpenTelemetry SDK and reproduces prior behavior
    # exactly. The exporter target comes from the environment (standard
    # OTEL_EXPORTER_OTLP_* vars, or the LANGFUSE_* trio), not from this flag.
    use_tracing: bool = False

    # --- Human feedback loop (opt-in, off by default) ---
    # When True, the app captures human quality judgments (a thumbs-up/down, an
    # optional score, and a freeform note) on a completed recommendation and
    # records them via the `feedback` package. Purely additive and observational:
    # feedback is written to a durable local log (the audit source of truth) and,
    # when tracing is on, emitted as a vendor-neutral OTLP span linked to the
    # decision's trace -- so ANY OTLP backend (Phoenix, Langfuse, Tempo, ...)
    # ingests it with only an endpoint change. It NEVER changes a route, score,
    # slot, or decision; any use of the labels (eval calibration, prompt tuning)
    # is a separate, offline, human-driven step. Off by default: flag-off hides
    # the UI, disables the endpoint, and imports nothing new.
    use_human_feedback: bool = False
    # When True (the default), a freeform feedback note and the captured decision
    # context are PII-scrubbed before they are written to the durable log, so an
    # off-network / shared deployment never persists customer identifiers. Turn
    # it OFF on a trusted company network, where the real customer PII is *wanted*
    # as part of the human feedback (who the account was, the actual address).
    # Categorical labels/scores are never PII and are unaffected either way; span
    # attributes never carry note text regardless (see feedback/emit.py).
    feedback_scrub_pii: bool = True
    # Absolute or relative path to the append-only JSONL feedback log -- the
    # durable, backend-independent record of every annotation (the curation
    # source of truth). Relative paths resolve against the process CWD.
    feedback_log_path: str = "feedback_data/annotations.jsonl"
    # When True, the decision span (webapp.recommendation) additionally carries the
    # decision's *input* (the intake) and *output* (the recommendation) as
    # OpenInference ``input.value`` / ``output.value`` attributes, so a
    # trace-backend-native dataset (Phoenix / Langfuse) built from those spans is
    # *replay-ready* -- not just filterable. It uses OPEN semantic conventions, so
    # it stays vendor-free while being natively understood by both backends.
    # Because the intake carries PII (name, address), this fires ONLY when
    # ``feedback_scrub_pii`` is also off (the same "PII allowed on the backend"
    # gate as the note-on-span behavior) -- so scrub-on always wins and no PII
    # reaches a trace. Off by default; a pure opt-in on top of tracing.
    use_trace_dataset_payloads: bool = False
    # --- Judge calibration (Phase 0; advisory, opt-in, off by default) ---
    # When True, the judge-calibration harness (``eval/judge_calibration.py``) is
    # available: it measures how well the automated LLM judges (brief_quality,
    # response_clarity) agree with the human labels being collected, so the auto
    # judges can be trusted (or not) before anything is gated on them. Purely
    # ADVISORY and OFFLINE -- it changes no decision and gates nothing; it only
    # reports agreement (Cohen's kappa, a dangerous-cell rate, a trust band).
    # Off by default; flag-off makes the CLI a no-op, so calibration never runs
    # unless explicitly turned on.
    use_judge_calibration: bool = False
    # Absolute or relative path to the append-only JSONL log of automated
    # JUDGE verdicts (see ``eval/judge_log.py``) -- the machine-verdict sibling of
    # ``feedback_log_path``'s human-label log, and the producer of the verdicts
    # ``scripts/calibrate_judges.py`` consumes. The path IS the switch: set it
    # empty to record nothing. It needs no ``use_*`` flag of its own because
    # recording is purely observational -- it changes no decision and cannot fail
    # a run (writes are swallowed; see judge_log.append_verdict) -- and gating it
    # off by default would mean an eval run still recorded nothing, which is the
    # problem this exists to solve. Defaults alongside the human log in the
    # gitignored feedback_data/, since judge scores are non-deterministic run
    # output, not source.
    judge_log_path: str = "feedback_data/judge_verdicts.jsonl"

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
            use_preferred_day_candidate=_bool_env(
                "SMART_ASSIGNMENT_USE_PREFERRED_DAY_CANDIDATE", True
            ),
            cluster_reference_miles=_float_env("SMART_ASSIGNMENT_CLUSTER_REF_MILES", 15.0),
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
            use_grounded_route_slot_escalation=_bool_env(
                "SMART_ASSIGNMENT_USE_GROUNDED_ROUTE_SLOT_ESCALATION", True
            ),
            use_grounded_route_slot_pick=_bool_env(
                "SMART_ASSIGNMENT_USE_GROUNDED_ROUTE_SLOT_PICK", False
            ),
            judgment_sample_count=_int_env("SMART_ASSIGNMENT_JUDGMENT_SAMPLE_COUNT", 3),
            judgment_consensus=os.environ.get("SMART_ASSIGNMENT_JUDGMENT_CONSENSUS", "unanimous")
            .strip()
            .lower(),
            judgment_retry_on_low_confidence_recommend=_bool_env(
                "SMART_ASSIGNMENT_JUDGMENT_RETRY_ON_LOW_CONFIDENCE", True
            ),
            use_address_resolution=_bool_env("SMART_ASSIGNMENT_USE_ADDRESS_RESOLUTION", True),
            use_escalation_triage=_bool_env("SMART_ASSIGNMENT_USE_ESCALATION_TRIAGE", True),
            use_session_memory=_bool_env("SMART_ASSIGNMENT_USE_SESSION_MEMORY", False),
            llm_backend=os.environ.get("SMART_ASSIGNMENT_LLM_BACKEND", "sage"),
            model=os.environ.get("SMART_ASSIGNMENT_MODEL", "gemini-3.5-flash"),
            sage_model=os.environ.get("SMART_ASSIGNMENT_SAGE_MODEL", "sage-gemini-2.5-flash"),
            use_sage_gateway=_bool_env("SMART_ASSIGNMENT_USE_SAGE_GATEWAY", False),
            role_models=_role_models_from_env(),
            repair_tool_call_args=_bool_env("SMART_ASSIGNMENT_REPAIR_TOOL_CALL_ARGS", True),
            sage_request_attempts=_int_env("SMART_ASSIGNMENT_SAGE_REQUEST_ATTEMPTS", 2),
            debug_sage_raw_response=_bool_env("SMART_ASSIGNMENT_DEBUG_SAGE_RESPONSE", False),
            use_tracing=_bool_env("SMART_ASSIGNMENT_USE_TRACING", False),
            use_human_feedback=_bool_env("SMART_ASSIGNMENT_USE_HUMAN_FEEDBACK", False),
            feedback_scrub_pii=_bool_env("SMART_ASSIGNMENT_FEEDBACK_SCRUB_PII", True),
            feedback_log_path=(
                os.environ.get("SMART_ASSIGNMENT_FEEDBACK_LOG_PATH")
                or "feedback_data/annotations.jsonl"
            ).strip()
            or "feedback_data/annotations.jsonl",
            use_trace_dataset_payloads=_bool_env(
                "SMART_ASSIGNMENT_USE_TRACE_DATASET_PAYLOADS", False
            ),
            use_judge_calibration=_bool_env("SMART_ASSIGNMENT_USE_JUDGE_CALIBRATION", False),
            # Unlike feedback_log_path above, an explicitly EMPTY value is
            # meaningful here (it disables recording) rather than falling back to
            # the default -- so only an unset var takes the default.
            judge_log_path=os.environ.get(
                "SMART_ASSIGNMENT_JUDGE_LOG_PATH", "feedback_data/judge_verdicts.jsonl"
            ).strip(),
        )


# Convenience default used when a caller doesn't inject its own Config.
DEFAULT_CONFIG = Config.from_env()
