# Architecture diagrams

Place a diagram of the agent's tool-calling flow here (e.g. `smart_assignment.png`).

The current architecture is a single ADK `LlmAgent` (`smart_assignment/agent.py`)
that talks to the user and calls one tool per step, in order:

```
intake_customer            (code — validate/merge address, cases, preferred slot)
find_candidate_routes      (code — geocode + Top-N nearest routes)
evaluate_and_score_routes  (code — HARD constraints, then weighted scoring)
recommend_or_escalate      (code — rank + total-score gate -> decision + reasoning)
  -> requires_human_review? -> agent calls request_input (ADK built-in, human input)
```

The agent (the LLM) decides *when* to call which tool and narrates the
result in conversation; it never computes a distance, a constraint check, or
a score itself -- every number comes back from the tool call. See
`smart_assignment/tools/slot_recommendation.py` for the tool implementations
and `smart_assignment/prompts.py` for the instruction that enforces this.

Reasoning (the natural-language trace on the final recommendation) is
produced deterministically inside `recommend_or_escalate` and then narrated
by the agent; the pipeline's own optional LLM-narrated reasoner
(`reasoning.LLMReasoner`, with a deterministic fallback) is a separate,
lower-level option used when calling `pipeline.run_slot_recommendation(...)`
directly (e.g. `scripts/run_local.py`), not by the conversational agent.

No image file is included in this package — generate one (e.g. via the
ADK Web UI's trace view, or any diagramming tool) and drop it here as
`smart_assignment.png` once available.

## Step 5, two ways: weighted-sum vs. grounded LLM judgment

Step 5 (recommend-or-escalate) has two interchangeable *decision strategies*
behind a common `Judge` protocol (`judgment/judge.py`), selected by
`Config.use_grounded_judgment` (env `SMART_ASSIGNMENT_USE_GROUNDED_JUDGMENT`):

```
                       hard constraints (constraints.py) -- ALWAYS run first,
                       the ONLY thing that can eliminate a candidate
                                     |
                       feasible / infeasible split (deterministic)
                                     |
             +-----------------------+------------------------+
             |  use_grounded_judgment=False  (DEFAULT)        |  =True
             v                                                v
   WeightedScoreJudge                               GroundedJudge
   rank feasible by weighted total_score            (judgment/ package)
   gate on total_score_threshold (0.60)             see the flow below
   -> today's behavior, unchanged
```

**Default path is unchanged.** `WeightedScoreJudge` is a thin wrapper over the
existing `pipeline.decide`, so with the flag off nothing about the current
behavior, tests, or offline demo changes.

### GroundedJudge flow (opt-in)

Instead of collapsing the soft factors into one weighted number and gating on a
fixed threshold, an LLM reasons over a structured **evidence packet** of the raw
per-candidate facts and makes the recommend/escalate call itself:

```
evaluate_candidates()                                    [UNCHANGED, deterministic]
  hard constraints -> feasible / infeasible split
        |
        v
build_evidence_packet (evidence.py)
  raw per-candidate facts (distance, clustering, utilization, headroom,
  window overlap) for feasible AND infeasible candidates; the legacy weighted
  total_score is included only as `reference_weighted_score` (NOT a gate)
        |
        v
Grounded Judgment call x1  (llm.py -> shared/llm.generate_text)
  structured JSON: decision (RECOMMEND|ESCALATE), confidence (HIGH|LOW),
  recommended_route_id (a feasible id or null), rationale, citations
        |
        v
Structured-Citation Verifier (verifier.py, deterministic -- no model call)
  1. pick must be in the feasible set (hard safety net on top of the schema)
  2. every fact/comparison citation must resolve + match the packet exactly
  3. tolerant prose scan: numbers/route-ids in the rationale must be grounded
        | pass                                    | fail
        v                                         v
  first sample confident recommend?          one corrective retry -> still
        |                                     fails -> DETERMINISTIC FALLBACK
   yes  |  no ("escalation-side":             (WeightedScoreJudge pick +
        |   ESCALATE, or LOW-confidence        DeterministicReasoner text) --
        |   recommend when the knob is on)     never worse than today
        v                          v
   SHIP on 1 call        resample up to k = judgment_sample_count
                         (env SMART_ASSIGNMENT_JUDGMENT_SAMPLE_COUNT)
                                   |
                                   v
                         consensus over the DECISION axis only
                         (judgment_consensus = unanimous|majority);
                         differing-but-good picks are NOT disagreement
                                   |
                         cleared -> RECOMMENDED ; otherwise -> ESCALATE,
                         surfacing all k reasoned takes to the specialist
                         (SlotRecommendation.alternative_takes)
```

**What stays invariant:** hard constraints run first and are the only thing that
eliminates a candidate; the LLM only ever chooses among the already-feasible set
(enforced by both the output schema and the verifier), so it can never place a
customer on an over-capacity or out-of-area route. Any mechanical failure
(unparseable/ungrounded output surviving one corrective retry, or a no-feasible
case) falls back to the exact deterministic result, so the grounded path is
never *worse* than the weighted one -- only, when it succeeds, better-reasoned.

**What changes on purpose:** the arbitrary `total_score_threshold` no longer
gates auto-assignment when grounded judgment is on. The escalate/recommend call
is the LLM's, made from the raw facts; "should a human look at this?" is
answered by the model's own confidence plus cross-sample agreement, not a fixed
0.60 cutoff.

### Config knobs (`shared/config.py`)

| Knob (env) | Default | Meaning |
|---|---|---|
| `SMART_ASSIGNMENT_USE_GROUNDED_JUDGMENT` | `false` | Master switch: grounded judgment vs. weighted-sum. |
| `SMART_ASSIGNMENT_JUDGMENT_SAMPLE_COUNT` | `3` | `k` — samples drawn for an escalation-side case (`1` disables resampling). |
| `SMART_ASSIGNMENT_JUDGMENT_CONSENSUS` | `unanimous` | How the `k` decisions clear back to a recommend: `unanimous` (precautionary) or `majority`. |
| `SMART_ASSIGNMENT_JUDGMENT_RETRY_ON_LOW_CONFIDENCE` | `true` | Whether a LOW-confidence *recommend* is escalation-side (resample) or ships as-is. A hard ESCALATE always resamples. |

**Where the flag takes effect.** `run_slot_recommendation(...)` honors
`use_grounded_judgment` whenever no explicit `judge=` is injected, so the master
switch reaches every surface — the offline demo (`scripts/run_local.py`), the
page generator, the web app, and the conversational tool alike. An explicitly
passed `judge=` always wins over the flag.

**Credentials required to see a difference.** Grounded judgment needs a working
LLM backend (`SMART_ASSIGNMENT_LLM_BACKEND` + its credentials). Without them the
judgment call fails and `GroundedJudge` deterministically falls back to the
weighted pick + `DeterministicReasoner` text — *byte-identical to the flag-off
output*. So on an offline/no-key run the two flag settings produce the same
text by design; a real LLM backend is what surfaces the grounded reasoning.
