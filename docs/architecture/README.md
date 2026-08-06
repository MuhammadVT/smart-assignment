# Architecture diagrams

Place a diagram of the agent's tool-calling flow here (e.g. `smart_assignment.png`).

The current architecture is a single ADK `LlmAgent` (`smart_assignment/agent.py`)
that talks to the user and calls one tool per step, in order:

```
intake_customer            (code — validate/merge address, cases, preferred slot)
find_candidate_routes      (code — geocode + Top-N nearest routes, plus the
                            nearest in-range preferred-day route when the Top-N misses it)
  -> address not found? -> agent calls resolve_address (grounded pick among the
                           geocoder's candidate matches) -> user confirms ->
                           intake_customer(confirmed) -> retry  [opt-in, default on]
evaluate_and_score_routes  (code — HARD constraints, then weighted scoring)
recommend_or_escalate      (code — rank + total-score gate -> decision + reasoning)
  -> requires_human_review? -> agent calls request_input (ADK built-in, human input)
```

Alongside those steps the interactive agent carries one **on-demand lookup**,
`geocode_prospect_address` (code -- the coordinates of the address on file, and
nothing else). It is a sibling of `find_candidate_routes`, not a split of it: it
answers a side question ("where is this customer?") with a single geocode instead
of running step 2's fetch-and-rank over the whole route set. It writes no session
state, is not a pipeline step (so it draws no live breadcrumb -- see
`webapp/narration.py`), and never substitutes for `recommend_or_escalate`. Batch
does not get it (see `_batch_agent_tools`): unattended runs ask no side questions.

The agent (the LLM) decides *when* to call which tool and narrates the
result in conversation; it never computes a distance, a constraint check, or
a score itself -- every number comes back from the tool call. See
`smart_assignment/tools/slot_recommendation.py` for the tool implementations
and `smart_assignment/prompts.py` for the instruction that enforces this.

## Batch (non-interactive) mode (`batch/` package)

The conversational chat is one way to run the workflow; **batch mode** is the
other. In production a prospect flows Salesforce -> Smart Assignment -> the
sales-consultant *Customer View* with **no human in the loop**: addresses come
from the CRM (there is nobody to confirm one), and the result is a recommendation
or an escalation the SC reviews later.

**Batch drives the SAME agent, non-interactively -- not a second decision brain.**
It runs the real agent architecture via `agent.build_batch_agent`: the same
`LlmAgent`, model role, and `escalation_triage` AgentTool as `root_agent`, so the
batch inherits the agent's natural-language reasoning and its triage brief. What
changes is the *shape* of the turn, tuned for one CRM-sourced prospect at a time
with no human present:

- **One consolidated tool, fewer round-trips.** Instead of the chat agent's four
  step-by-step tools (`intake` -> `find` -> `evaluate` -> `recommend`), the batch
  agent is given a single `assign_prospect` tool
  (`tools/slot_recommendation.py`) that runs intake -> geo -> evaluate -> score ->
  decide in one call and writes the same session state. With intake seeded from the
  CRM (below), a prospect resolves in ~2 model round-trips instead of ~5. The tool
  invents no decision logic -- it composes the exact same pipeline functions
  `recommend_or_escalate` already runs, so the decision is byte-identical.
- **A batch instruction** (`prompts.build_batch_instruction`) tells the agent to
  call `assign_prospect` once and present the result, with no conversational
  narration between steps.

`AgentBatchRunner` (`batch/agent_runner.py`) is the orchestrator. Per prospect it
seeds the CRM profile into a fresh ADK session, runs one non-interactive turn, and
maps the outcome to a `BatchRecord` -- reusing the same
`reporting.page.build_workflow_payload` the Customer View renders, so a batch
result renders with **no frontend change**. The decision the agent's tool made is
reused from the session snapshot (never re-sampled), exactly as the web app does.

```
ProspectSource        build_batch_agent (one turn/prospect)     build_workflow_payload   ResultSink
 (Salesforce / mock) -> assign_prospect: intake->geo->..->decide -> (unchanged renderer) -> (JSONL / API)
     |                    └ fallback: run_one (deterministic floor)                            |
     +----------------------------- one BatchRecord per prospect ------------------------------+
```

**Never worse than the deterministic baseline.** The deterministic pipeline
(`runner.run_one`) is the **floor**, not a separate mode: if the agent can't be
built (no credentials/backend) the whole run uses `run_one`; if a single agent turn
fails or produces no decision, *that* prospect degrades to `run_one`. So batch
still runs fully offline and a broken backend can never make it worse than the
deterministic result. A per-prospect failure becomes `needs_attention` and **never
aborts the batch**.

**The mode is an entry point, not a global config switch.** `scripts/run_batch.py`
sits beside `scripts/run_local.py` and `scripts/run_web.py`; nothing on the
interactive path builds the batch agent, so the conversational `root_agent` is
untouched with no flag to thread. Collaborators are injected exactly as the
pipeline's are -- a `ProspectSource` (mock/JSON now, a `SalesforceProspectSource`
behind the same one-method protocol later), a `ResultSink` (`JsonlResultSink` now,
an API/DB sink later), plus the geocoder, routes, and config.

**The two human-in-the-loop steps are replaced, each preserving the guarantees:**

| Human step in chat | Batch substitute | How the guarantee holds |
|---|---|---|
| Intake conversation (multi-turn Q&A) | **Seeded, not asked.** The CRM profile is written into the ADK session state before the turn, so the agent goes straight to `assign_prospect` | Same intake validation runs inside the tool; a bad profile still fails to a `needs_attention` |
| Address confirm (`resolve_address` -> user confirms) | **Trust Salesforce as-is.** A geocode miss (or intake reject) becomes a `needs_attention` record for a human to fix the source data -- the batch agent has no `resolve_address` tool | Never fabricates an address: batch never resolves one, so the "no fabricated actionable value" guarantee holds trivially |
| Escalation handoff (triage brief -> `request_input` pause) | **Captured, not paused.** The agent still composes its `escalation_triage` brief and calls `request_input`; the runner INTERCEPTS that call, captures the brief, and terminates the turn (no human reply) | Deterministic decision unchanged; triage stays advisory/read-only; the SC is the human, just asynchronous. On the deterministic floor the brief is `triage.compose_brief` instead |

**The output contract (`BatchRecord`, `batch/sink.py`)** carries the envelope
(`prospect_id`, `generated_at`, `outcome`) plus the pieces each outcome needs:

```
outcome = recommend       -> payload (build_workflow_payload: frontendHtml, resultHtml, map, ...)
outcome = escalate        -> payload + review_reason + triage_brief
outcome = needs_attention -> error (no payload; the address wouldn't geocode / intake rejected it)
```

**Grounded reasoning is config-gated, same as chat.** Batch honors
`use_grounded_route_slot_*` and `use_escalation_triage`, and every LLM call falls
back to the deterministic result when the backend/credentials are unavailable.
A per-prospect failure degrades to `needs_attention` and never aborts the batch.
Execution defaults to **sequential** (`concurrency=1`, the simplest, exact prior
behavior); because each prospect is fully independent (its own ADK session; routes
and the geocoder resolved once and read-only), raising `AgentBatchRunner`'s
`concurrency` (CLI `--concurrency`) fans the agent turns out for throughput,
bounded by a semaphore to at most that many in flight.

Run it: `python3 scripts/run_batch.py --mock-geocoder` (built-in demo prospects,
fully offline) or `--source prospects.json --out results.jsonl`.

## Candidate identification (`pipeline.geo_lookup`)

Step 2 geocodes the prospect and takes the **Top-N nearest** routes by distance
to each route's service center. That cut is *day-blind*, which has a sharp edge:
a customer who asks for Thursday can have every Thursday route eliminated by
proximity alone — before any constraint, score, or decision layer sees anything.
The damage is worse than a missed preference, because `window_match` keeps its
weight (`RS_WEIGHT_WINDOW`) in the denominator whether or not it is satisfiable:
with no same-day route in the set, *every* option scores 0 on it, so stating a
preference can only ever **lower** the totals.

So when `Config.use_preferred_day_candidate` is on (**default**), and none of the
Top-N runs on the stated day, the **nearest route that does** is kept as one
ADDITIONAL candidate:

```
ranked = sorted(routes, by distance)          # day-blind, as before
candidates = ranked[:top_n]
  + nearest route with route.day == preferred.day AND
    distance <= service_distance_limit(route),   when no candidate has that day
```

Deliberate choices, each of which the alternative would have cost something:

- **Additive, not displacing.** `top_n_candidate_routes` becomes a floor; the
  candidate set is at most `N + 1`. Displacing the N-th would trade a
  known-close route for a preference that may not survive the constraints.
- **Exactly one route.** The ranking is by distance, so a preferred-day route
  already inside the Top-N *is* the nearest one — "add the nearest preferred-day
  route" and "add one only when the day is missing" are the same rule, and it can
  never duplicate a candidate.
- **Capped at the service-area limit.** A route beyond it would *provably* fail
  `geographic_serviceability`, so adding it would only put a guaranteed-rejected
  route in front of a specialist — noise, not a diagnostic. The limit is
  `SMART_ASSIGNMENT_MAX_SERVICE_MILES`, tightened by a route's own
  `service_radius_miles` when it declares one, and it comes from the constraint's
  own helper (`constraints.service_distance_limit`) rather than a second copy of
  the rule — so the candidate filter and the hard constraint can never disagree
  about what "in range" means. The scan continues past an out-of-range route
  rather than giving up, since a farther route may declare a wider radius and
  still be serviceable.
- **The cap is on distance only.** An in-range preferred-day route that is too
  *full* is still added, fails `route_capacity`, and reaches the specialist as a
  rejected candidate. That one is actionable (split the order, move a stop), and
  unlike distance it depends on the order size rather than on geography alone.

This guarantees the preference is **considered**, never that it wins — the added
route is scored like any other and can lose. Flag off reproduces the prior
day-blind Top-N exactly. Pinned by `tests/test_geo_lookup.py`.

| Knob (env) | Default | Meaning |
|---|---|---|
| `SMART_ASSIGNMENT_TOP_N` | `3` | How many nearest routes to evaluate. |
| `SMART_ASSIGNMENT_USE_PREFERRED_DAY_CANDIDATE` | `true` | Also keep the nearest **in-range** route running on the stated preferred day, when the Top-N misses it. |
| `SMART_ASSIGNMENT_MAX_SERVICE_MILES` | `25.0` | Doubles as the cap on that extra candidate (tightened by a route's own radius). |

## Delivery-slot selection (`shared/slot_selection.py`)

The prospect should be delivered *when the truck is already in their
neighborhood*, inferred from the route's nearest committed stops. Two
deterministic steps:

```
identify_available_slots   nearest committed stops -> group by time (a morning
                           vs. an afternoon neighborhood) -> one candidate per
                           cluster, a fixed-length window CENTERED on the
                           cluster's inverse-distance-weighted midpoint time
                           (the slot "between the adjacent stops", pulled toward
                           the closer ones). No customer preference here.
select_candidate_slots     keep the top-N per route by quality (fit + low
                           contention), but ALWAYS keep any candidate that
                           overlaps a stated preference ON THE PREFERRED DAY
                           -> this is the menu.
```

There is deliberately **no "pick one slot" step**. This module only enumerates;
the winner is chosen by scoring every (route, slot) pair (below), where the
customer's preference is one weighted, day-gated factor among four. Preference
therefore influences the outcome through exactly one auditable weight
(`RS_WEIGHT_WINDOW`) rather than compounding across a menu blend and a score.

This replaced an earlier version that snapped the prospect to a route's nearest
*existing* window and anchored the slot at that window's start. The candidate
menu (`EvalContext.available_slots`, each `SlotOption` carrying its
`anchor_time`, `fit_score`, `committed_overlap`, `basis`) is exactly the set the
decision layer — deterministic or grounded — reasons over.

`EvalContext.window_overlap_minutes` is the best overlap *any* candidate in the
menu achieves with the preferred window. It is a reference fact cited in triage
briefs, never a decision input.

**The preference is day-gated once, up front.** A preference is always a
(day, window) pair, so `constraints.applicable_preferred_window` returns `None`
for a route running on a day the customer didn't ask for — and the menu's
always-keep rule and the overlap fact both then behave as "no preference". Only
same-day routes can earn preference credit, exactly as
`scoring._slot_window_match` already gates the `window_match` factor. Without
that gate a Wednesday route scored a time-of-day match against a Tuesday
preference.

**Phase A/B seam:** `stop_reference_time` is the single function that turns a
committed stop into a "when is the truck near here" clock value — today the TW1
window midpoint, later a real planned-arrival ETA (and, with a stop *sequence*,
the interpolation becomes true bracketing between the two sequential stops the
prospect is inserted between) — with no caller change. Knobs:
`SMART_ASSIGNMENT_SLOT_{NEIGHBORS,CLUSTER_GAP,WINDOW_MINUTES,CANDIDATES}` and
`SMART_ASSIGNMENT_SLOT_WEIGHT_{FIT,CONTENTION}`.

## Route-slot scoring (`routeslot/` package)

Ranking **routes** alone has a blind spot: a route can win on capacity and
clustering while its only workable slot is densely shared by high-value
customers, and a route-level ranker can't see that. So the **decision unit is the
(route, slot) pair**: every candidate slot on every feasible route is scored
separately, and slot availability influences which *route* wins.

Two factor levels (`shared/scoring.score_route_slot`):

| Factor | Level | Varies across a route's slots? |
|---|---|---|
| `geographic_clustering` | route | no — shared down to every slot |
| `capacity_buffer` | route | no — shared down to every slot |
| `window_match` | slot | yes — *this* slot's overlap with the preference |
| `slot_availability` | slot | yes — *this* slot's tier-weighted openness |

`window_match` is **dropped entirely** when there's no stated preference (rather
than the old 0.6 neutral); the total self-normalizes over whatever factors are
active. **Openness** = `1 / (1 + Σ harm(incumbent))` over committed stops sharing
the window, where `harm` protects valued tiers — 5/Perks `1.0` > 4 `0.6` > *the
prospect* > Other `0.1` (unknown `0.4`). So a window jammed with Other-tier stops
still scores open, while one shared by tier-5/Perks incumbents scores contended.

```
evaluate_candidates (flag on) -> per feasible route, score each candidate slot as
                                 its own (route, slot); fold the route's BEST
                                 scored slot back onto the evaluation so
                                 route-level ranking reflects the best route-slot
        |
        v
build_route_slot_packet   flatten all feasible route-slots into one indexed menu,
 (routeslot/evidence.py)  each with per-slot factor values + reference total; name
                          the deterministic best index
        |
        v
decide_route_slot         non-feasible cases ALWAYS escalate deterministically; for
 (routeslot/decide.py)    the feasible ones the recommend-vs-escalate call depends on
                          Config.use_grounded_route_slot_escalation. Branches:
                            · no feasible route            -> ESCALATED_NO_FEASIBLE_SLOT
                            · feasible route, no slot built -> ESCALATED_NO_FEASIBLE_SLOT
                                                              (distinct review_reason)
                            · feasible route-slots:
                               – flag ON (default): the LLM decides over ALL feasible
                                 route-slots (see below)
                               – flag OFF: the 0.55 bar gates it; none ≥ bar
                                 -> ESCALATED_LOW_SCORE (deterministic best proposed, NO
                                 llm call); ≥1 ≥ bar -> RECOMMENDED, LLM picks among the
                                 eligible (above-bar) menu when pick-grounding is on.
```

**LLM-decided escalation (`use_grounded_route_slot_escalation`, default on).** The
LLM makes the recommend-vs-escalate call *itself* over **all** feasible route-slots
— the `route_slot_score_threshold` bar does **not** gate it (it rides along in the
packet as a reference `auto_assign_threshold` + per-option `meets_auto_assign_bar`,
and stays the fallback). The output adds `decision` (RECOMMEND | ESCALATE) and
`confidence` (HIGH | LOW) to the same grounded, cited, verified choice; `chosen_index`
is always the strongest option (on an ESCALATE it's the best-but-insufficient one the
specialist reviews). Because the bar no longer binds it, the model **may escalate an
above-bar option** (it judges even the best isn't good enough) or **recommend a
below-bar one** — so the k-try guardrail matters: a confident RECOMMEND ships on one
verified call; an ESCALATE or a LOW-confidence RECOMMEND is resampled
`judgment_sample_count` times and must reach `judgment_consensus` (unanimous/majority)
to auto-assign, else it escalates (`ESCALATED_LOW_SCORE`, distinct review_reason,
strongest option proposed, all reasoned takes in `alternative_takes`). On
**any** mechanical/verification failure it falls back to the deterministic threshold
decision, so it is never worse than the bar-gated baseline. Set the flag off to gate
on the bar instead (the LLM then only picks among above-bar options, as the box's
flag-off branch describes).

This is the same **constrained-option + grounded + deterministic-fallback**
pattern as `triage/` and `address_resolve/`, applied to the route-slot unit; the
weighted total per route-slot is the reference and the deterministic fallback.

**Structured explanation (`routeslot/schema.py`).** A one-line rationale can't
carry the *trade-off* an ops manager needs to trust an auto-assign, so on a
RECOMMENDED pick the model returns a decomposed explanation rather than a
sentence: `decision_summary` (the action line), `primary_reasons[]` (a
comprehensive read — one line per scored factor, each with its number:
geographic fit, capacity headroom, preferred-window match when a preference was
stated, and slot openness — so no factor is silently dropped), `key_tradeoff`
(what the winner gives up vs. the
runner-up and why that's acceptable), `runner_up {index, why_not}`, and
`vs_deterministic_default {verdict, note}` (an explicit AGREE/DIVERGE against the
weighted blend). Only `chosen_index` is *actionable* — a real index from the
enumerated menu; every other field is grounded explanation. These land on
`SlotRecommendation` as their own fields, and `reasoning` is still set so existing
consumers keep working. `page.py` renders each as its own section, falling back to
the flat `reasoning` line when the structured fields are absent.

**Deterministic floor, grounded enrichment.** The structured fields are *always*
populated on a RECOMMENDED route-slot — first deterministically from the score
breakdown (`_apply_deterministic_narrative`: `decision_summary`, a `primary_reasons`
line for *every* scored factor in the breakdown's canonical order (not just the
top two), the score-ranked `runner_up`, and a
`key_tradeoff` naming the one factor the runner-up actually leads on), then, when
the grounded LLM produced a *verified* choice, overwritten by its reasoned prose
and AGREE/DIVERGE `default_comparison`. So the explanation is never a bare
one-liner: even with grounded reasoning off, credentials missing, or a
verification fallback, the user still gets the reasons and the trade-off; the LLM
only makes the prose better when it's available. (`default_comparison` is the one
field the deterministic floor leaves unset — a self-assessment against the default
only means something when an LLM actually diverged from it.) The conversational
agent narrates these fields directly — `prompts.py` step 4 tells `root_agent` to
lead with the summary, give the reasons, and state the trade-off vs. the runner-up
— so the web-app recommendation reads the same way the page does, not as a
one-sentence verdict.

These fields are populated for every recommendation: a deterministic structured
floor is always built from the score breakdown, and a verified grounded choice
replaces it with the model's own reasoned prose.

**Naming routes.** Everywhere a route is named to the user — the deterministic
narrative (`decision_summary`, `runner_up`, `reasoning`, the rejected/infeasible
lines via `_route_label`), the grounded route-slot prose, the `root_agent`
conversation, and the triage brief — it is written as `<route id> - <route name>`
(e.g. `RTE-4100 - Central Houston`), so the stable id and the human-readable name
always travel together. The deterministic sites enforce it; the prompts instruct
the LLM-authored ones.

The verifier (`routeslot/verifier.py`) gains two checks beyond the structured
citations: it rejects a **dishonest self-assessment** (verdict must match whether
the pick actually equals the deterministic default; a DIVERGE needs a note; the
trade-off and a valid, distinct runner-up are required whenever more than one
option is offered), and it runs a **tolerant prose scan** (mirroring
`triage/verifier.py`) so *every* number (including `"1,234"`-style thousands),
route-id or `"route N"` mention, day name, and HH:MM time stated in any free-text
field must be grounded in the packet — not just the values in the citation list.
Percent phrasings normalize only against fraction-scale facts and never for
unit-bearing tokens ("84 miles" can't launder through a stored 0.84), and small
integers carrying a unit or percent sign are checked. Any failure feeds the
single corrective retry, then the deterministic fallback: never worse than
before, only — on success — better explained.

The **one corrective retry covers two failure modes, not just one** (shared by
both the pick-only and grounded-escalation paths in `_grounded_choice`): a reply
that fails *verification*, **and** a reply that won't *parse* into a choice at all.
A parse/shape failure (`JSONDecodeError`/`RouteSlotChoiceParseError`) retries once
with a "return JSON only" corrective rather than dropping straight to the
deterministic floor. A *backend/credentials/transport* error is still **not**
retried (retrying missing creds only doubles the latency) — it falls back
immediately, logged.

**How the choice is obtained — a tool call, not a JSON string.** The direct SAGE
"generic agent" is a *conversational* agent: asked to "reply with a JSON object" it
reliably **narrates** ("I recommend route 6032…"), which a downstream `json.loads`
rejects, and the direct-agent API exposes no JSON/structured-output enforcement (its
`response_schema` only selects the SAGE *envelope* — text vs. function_call — not a
content schema). Prompt-nudging and the retry above lower the failure rate but can't
eliminate it. So `routeslot/llm.py` gets structured output the way the agent
actually cooperates: it offers **one function**, `submit_route_slot_decision`
(`ROUTE_SLOT_DECISION_TOOL`, a provider-agnostic `{name, description, parameters}`
mirroring the output contract), and uses the model's **call arguments** as the
choice. `shared/llm.py`'s `generate_tool_call` drives this through the ADK `BaseLlm`
(direct agent *and* LLM-Gateway sibling), returning `(call_args, text)`: on a tool
call `call_args` is the SDK-repaired arguments dict; if the model narrates anyway,
`call_args` is `None` and the prose is salvaged (strict → brace-slice →
`json_repair`) and, failing that, logged. Non-sage backends have no tool channel yet
and fall back to text + extraction — they gain real JSON mode when the project moves
to the gateway. Net: the grounded layer fires on the conversational SAGE backend
instead of silently degrading, with every guarantee (deterministic floor,
verification, retry, fallback) intact.

**Threshold.** `route_slot_score_threshold` defaults to `0.55`, a touch below the
route-only `0.60`: dropping the 0.6 window neutral and adding availability shifts
the score distribution, and ops asked to err slightly toward recommending. On the
mock accounts the natural separation sits between the designed escalation
(Galleria, a large order on a near-full route, ≈0.54) and the clean recommends
(≈0.77–0.83); `0.55` sits at the low end of that gap — auto-recommending as much as
possible while still catching the genuinely over-full route.

## Escalation-triage sub-agent (`triage/` package)

The first real multi-agent split. When `recommend_or_escalate` returns
`requires_human_review: true` and `Config.use_escalation_triage` is on (env
`SMART_ASSIGNMENT_USE_ESCALATION_TRIAGE`, default on), `root_agent` consults an
`escalation_triage` sub-agent — an `LlmAgent` exposed as an ADK `AgentTool` —
before the human handoff:

```
recommend_or_escalate -> requires_human_review?
        | yes
        v
  escalation_triage   (AgentTool: a second LlmAgent, consult-and-return)
     └─ get_escalation_context (reads session state: the profile + last
        recommendation; re-derives every feasible/infeasible route with its
        raw facts, the decision thresholds it was judged against, + any split
        model opinions)  -> composes a specialist brief
        in a fixed, scannable layout:
        SITUATION · ROOT CAUSE · OPTIONS (ranked, most-workable first, each with
        its state / action / trade-off) · RECOMMENDATION (advisory starting point)
        · DECISION NEEDED (the one question)
        |
        v
  root_agent -> request_input(message = the brief)   (root_agent still owns
                                                       the human-in-the-loop pause)
```

The brief is laid out for a fast human decision, not as a paragraph: a one-line
situation, the specific gate that tripped, then **ranked** remediation options
(option 1 = closest to workable) each showing the route's current state, the
concrete action, and the trade-off, followed by an advisory `RECOMMENDATION`
(which option to start with, and why — overridable, never a decision) and the
single `DECISION NEEDED`. It renders as multi-line text (the chat bubble keeps
newlines) so the specialist can scan and compare options at a glance. The
prompt-driven layout stays fully grounded — the same `verify_brief` prose scan
(below) still rejects any figure not in the escalation context.

**The `request` argument is a fixed label, not a payload
(`prompts.TRIAGE_REQUEST_LINE`).** ADK's `AgentTool` declares `request` as a
*required* string, so the model must always send one — but the triage agent
ignores it and loads every fact from session state via `get_escalation_context`.
Left unspecified, the model filled that hole differently from run to run:
measured live against sage over 15 escalations of the same prospect, it sent
four different values, and on 4 of the 15 it pasted the entire
`recommend_or_escalate` result (805 chars of escaped JSON). With the argument
pinned, the same 15 runs all sent the one 47-char line.
That paste is what makes the call fragile — the oversized nested
blob is where the array-wrapped tool-call arguments come from that ADK's
argument parser rejects (see `Config.repair_tool_call_args` below). Both
instructions therefore name one exact short line, and forbid pasting tool
output into `request`. The repair stays as the backstop; this removes the input
that triggers it.

**Why an `AgentTool` (consult-and-return), not a peer agent with control
transfer:** `root_agent` stays in control of the conversation and keeps
ownership of the `request_input` pause/resume; triage is a bounded call that
returns text. It runs strictly *downstream* of the deterministic decision and
is **read-only** — `get_escalation_context` never writes state, and the triage
agent has no tool to change the route, score, or decision. So the pipeline's
deterministic auditability is untouched; triage only turns the escalation into
a better-explained, more actionable handoff. Turning the flag off reverts to a
bare `request_input`.

Built lazily inside `root_agent`'s construction (`agent.py`), so importing the
package stays credential-free; the sub-agent resolves the LLM backend only when
`root_agent` itself is built.

**A non-agent sibling for batch (`triage/compose.py`).** The `AgentTool` above is
the *conversational* path -- it reads session state and owns the `request_input`
pause. Batch mode (above) has neither a session nor an agent loop, so
`compose_brief` composes the same brief with a single grounded `generate_text`
call (the `routeslot` pattern) instead: it reuses the identical deterministic
finalization (`normalize_brief` layout + `verify_brief` grounding scan, with one
corrective retry then an advisory caveat), and falls back to a deterministic-floor
brief built from the escalation context's raw facts on any failure. The pure
`build_escalation_context` / `escalation_context_from_recommendation` builders are
extracted from `get_escalation_context` so the agent (via the tool) and batch (via
`compose_brief`) reason over an identical context; the tool now delegates to them,
so the conversational brief is unchanged.

## Grounded address resolution (`address_resolve/` package)

When the geocoder can't resolve a prospect's address (a typo, or an ambiguous
one), the agent shouldn't dead-end — and it must **not invent** a corrected
address (an actionable value). This layer applies the same constrained-option,
grounded-reasoning pattern to address correction:

```
Geocoder.suggest(address)   provider-agnostic capability (shared/geo.py): return
 (integrations/*)           a ranked SET of real AddressCandidate matches, or []
                            (MockGeocoder ranks the demo addresses by token
                            overlap; CensusGeocoder maps its addressMatches).
build_address_packet        enumerate the candidates + a deterministic token-
 (address_resolve/          overlap `similarity` per candidate; the highest is the
  evidence.py)              `deterministic_choice_index` — the demoted heuristic,
                            offered as a reference AND used as the fallback.
resolve_address             the LLM picks a candidate BY INDEX with a cited
 (address_resolve/          rationale; a verifier (verifier.py) checks the index
  resolver.py)              is in the set and every citation matches a real fact;
                            one retry, then fall back to the deterministic pick.
```

`resolve_address` is a **`FunctionTool`** (not a sub-agent): the choice is a
constrained, verifiable, index-based selection whose output is checked
deterministically, so it belongs in the grounded-function family
(`routeslot`/`address_resolve`), not the `AgentTool` family (which is for the
free-form triage brief). The tool only ever returns a **suggestion**: on a hit it
returns `needs_confirmation` with the suggested address + alternatives, and the
instruction (`prompts.py`, `ADDRESS_RESOLUTION_GUIDANCE`) requires the agent to
get the **user's confirmation** — an intake-level pause — before adopting it via
`intake_customer`. The human is the verification step.

Guarantees preserved: the LLM selects from the geocoder's enumerated set and
never free-generates an address; the deterministic highest-similarity candidate
is the fallback on any LLM/verify failure; and when there are **no** candidates
at all, it falls back to today's "ask the customer to double-check it." Gated by
`Config.use_address_resolution` — **default on** (ops asked for it), and turning
it off reproduces the prior no-correction behavior exactly (the tool isn't even
registered, and the instruction doesn't mention it). The `suggest` capability is
feature-detected (`supports_suggestions`), so a provider without it simply yields
the double-check fallback — Census surfaces alternatives mainly for
*ambiguous-but-valid* input, while genuine-typo suggestions want a suggest-capable
provider (Google Places, Mapbox) behind the same protocol seam.

## Per-role model selection

Every LLM-using surface resolves its model through one place — `Config.for_role(role)`
(`shared/config.py`) — so you can assign the right model to the right task
without changing any call site's logic. Roles and their env overrides:

| Role | Surface | Env override |
|---|---|---|
| `root_agent` | the conversational `LlmAgent` | `SMART_ASSIGNMENT_MODEL_ROOT_AGENT` |
| `triage` | the escalation-triage sub-agent | `SMART_ASSIGNMENT_MODEL_TRIAGE` |
| `judgment` | the grounded route-slot decision (`routeslot/` package) | `SMART_ASSIGNMENT_MODEL_JUDGMENT` |
| `address_resolve` | the grounded address-candidate pick (`address_resolve/` package) | `SMART_ASSIGNMENT_MODEL_ADDRESS_RESOLVE` |

`for_role` returns a copy of the config with the *active* model field overridden
(`sage_model` under the sage backend, `model` otherwise); a role with no override
returns the config unchanged (same object), so leaving the vars unset keeps a
single model everywhere and behavior is identical to before. The LLM **backend**
(`sage` vs `standard`) stays global — only the model *tier* varies per role, so
each override value must match the active backend's naming. `resolved_model(role)`
returns the effective model name for a role (handy for logging/tests).

The two functions that actually talk to a backend (`shared/llm.get_llm` and
`generate_text`) are unchanged — each caller simply hands them
`config.for_role(<its role>)`.

`generate_text` is a **synchronous** API, but the sage backend it fronts is async
and its aiohttp `ClientSession` (inside the Sage SDK's process-global litellm
handler) is **bound to the first event loop that touches it** — under the web app
that is uvicorn's server loop, where the agent's own turns run. Two facts collide
there:

- ADK invokes a synchronous `FunctionTool` **inline on the server loop thread**, so
  the pipeline these tools drive blocks that loop while running.
- The sage coroutine `generate_text` must run has to execute **on that same loop**
  (its session lives there); a bare `asyncio.run()` raises `asyncio.run() cannot be
  called from a running event loop`, and running it on any *other* loop raises
  `loop <...> is not the running loop`.

You cannot both block the server loop and run a coroutine on it. The resolution is
a two-part cooperation:

1. **The tools offload their blocking body off the loop.** `agent.py` wraps each
   pipeline `FunctionTool` with `_offloaded_tool`, making it an `async` tool that
   runs its synchronous work in a worker thread via `offload_to_worker_thread`
   (`shared/llm.py`). That frees the server loop. The web app's
   `_visualization_from_state` re-run is offloaded the same way. `functools.wraps`
   keeps the tool's name/signature/declaration identical, so ADK's `tool_context`
   injection is unchanged.
2. **The grounded call hands its coroutine back to the server loop.**
   `offload_to_worker_thread` records the server loop in a `ContextVar` (which
   `asyncio.to_thread` copies into the worker thread); `_run_coro_blocking` then
   submits the sage coroutine to that recorded *host loop* via
   `asyncio.run_coroutine_threadsafe(...).result()` — so it runs where the session
   is bound. With no host loop recorded (the CLI/offline case) it just uses
   `asyncio.run()`. This keeps the grounded path working in both worlds instead of
   silently falling back to the deterministic result.

### Brief groundedness verification

The triage brief is free text, so — unlike the grounded-judgment layer, which
verifies structured citations — its numbers are checked by a prose scan
(`triage/verifier.py`, deterministic, no LLM). `verify_brief` confirms every
figure, route-id, day name, and HH:MM time in the brief is grounded in the
escalation context; `collect_grounding` stashes the groundable facts (numbers,
route-ids, days, windows, scrub-labels) in session state when
`get_escalation_context` runs. It's tolerant by design — route-ids, route
names, and the customer name (any of which may carry digits, e.g. a numeric
route-id `3170` or a name `BT149361-[…]`) are scrubbed first,
percent-vs-fraction is normalized (only against fraction-scale values, and
never for a unit-bearing figure like "84 miles"), small bare counts without a
unit are ignored — so faithful prose passes and only genuinely invented
figures are flagged.

Two enforcement points:

```
triage agent drafts brief
   ├─ (cooperative) calls check_brief_grounding(brief) -> revise
   │                bounded: MAX_GROUNDING_CHECKS (2) -- see below
   └─ (deterministic) after_model_callback (_finalize_brief):
        1. normalize_brief -> reflow the FINAL brief into the one canonical
           layout (headers/options/labels each on their own line)
        2. verify_brief -> if any figure/route is still ungrounded, append a
           caveat naming them ("⚠ Unverified — figures not found …")
```

**The decision thresholds are facts (`decision_thresholds`, `triage/context.py`).**
An escalation is *defined* by a bar it failed to clear, and the brief's ROOT CAUSE
section is explicitly asked to name that gate "with the exact numbers". Those bars
were originally absent from the escalation context, so the verifier flagged them
as invented — even though the context's own `review_reason` had handed the agent
the figure ("No route-slot cleared the **55%** auto-assign bar"). The instruction
was unwinnable: the only way to pass the check was to drop the number, and the
agent needed two or three **full brief rewrites** to discover that. The context now
publishes a `thresholds` block (auto-assign score bar, utilization ceiling, safe
utilization line) as fractions, exactly like every other ratio it carries, and
`collect_grounding` picks them up like any other fact. Nothing is loosened — a
fabricated bar ("63%") is still flagged.

**The revision loop is bounded (`MAX_GROUNDING_CHECKS`).** Every grounding round
costs a *full* regeneration of the brief — the agent passes the whole brief as the
tool's argument, then writes it again as its final answer — and a brief generation
is the only call in this system measured to reach the sage request timeout (~10s
median, with a tail past the 30s `SAGE_TIMEOUT`, versus ~2.5s for a root-agent tool
call). An unbounded loop therefore multiplies the chance the whole turn dies while
adding **no** guarantee, because `_finalize_brief` re-verifies the final brief and
caveats anything ungrounded regardless. After two checks the tool returns
`"stop": true` alongside the still-flagged items; `ok` stays honest about
groundedness, and `stop` says what to do about it. The budget resets in
`get_escalation_context`, so each triage invocation gets its own.

**Layout normalization (`triage/formatting.py`).** The brief is LLM-written, so
its formatting drifts turn to turn — one escalation comes back tidy and
multi-line, the next as a single run-on line. `normalize_brief` deterministically
reflows any brief into the canonical structure by putting the known section
headers, the `N)` option markers, and the `Action`/`Trade-off` labels on their own
lines. It only moves whitespace — never a word, number, or route — so a
well-formed brief is left materially unchanged, it's idempotent, and the grounding
scan (which sees identical figures) is unaffected. It runs at the source (the
callback above) *and* at the web-app display surface (`llm_chat` normalizes the
`request_input` message), so the specialist always sees the same scannable layout
even if an intervening agent reflowed the brief.

The callback always runs, so ungrounded figures are flagged for the specialist
even if the agent skipped the self-check. It only *annotates* (never silently
drops the brief), and is defensively wrapped so it can never break the agent —
triage is advisory and human-reviewed, so a visible caveat is the right
guarantee (vs. the route-slot decision layer, which hard-rejects + falls back
because it gates an auto-assign decision).

Reasoning (the natural-language trace on the final recommendation) is produced
inside the route-slot decision — a deterministic structured floor
(`_apply_deterministic_narrative`), replaced by the model's own reasoned prose
when a grounded choice verifies — and then narrated by the agent.

No image file is included in this package — generate one (e.g. via the
ADK Web UI's trace view, or any diagramming tool) and drop it here as
`smart_assignment.png` once available.

## Sage LLM Gateway sub-path (`shared/llm.py`, opt-in)

The Sage SDK ships two distinct ways to reach a model under `llm_backend =
"sage"`, and this repo can use either without touching any call site:

- **Direct-to-agent (default).** `SageLlmRegistry`/`SageLiteLlm` call one
  registered SAGE **agent** (by `sage_model`, a `sage-*` id) over the SAGE
  agent API, authenticated with `SAGE_CLIENT_ID`/`SAGE_CLIENT_SECRET`/
  `SAGE_ENVIRONMENT`.
- **LLM Gateway (`Config.use_sage_gateway = True`).** The SDK's `GatewayLlm`
  — itself an ADK `LiteLlm` — routes the call through Sysco's enterprise LLM
  Gateway instead: an OpenAI-compatible litellm proxy, with the SDK injecting
  an OAuth2 token it refreshes on a timer. Credentials are
  `LLM_GATEWAY_CLIENT_ID`/`LLM_GATEWAY_CLIENT_SECRET` (read directly by the
  SDK's `GatewayClient`, not this repo's `Config`); `LLM_GATEWAY_ENV` is
  optional (defaults to `"qa"`). Under this sub-path `sage_model` names a
  gateway-exposed model id (e.g. `"gpt-4o"`), not a SAGE agent — `GatewayLlm`
  wraps it as `"openai/{model}"` itself.

Both classes are lazily imported the same way (`shared/llm.py`'s
`_load_sage_registry` / `_load_sage_gateway_llm_cls`, which import the SDK
installed via the `sage` optional extra — `uv sync --extra sage` — and raise an
actionable `ModuleNotFoundError` if it is absent), and because
`GatewayLlm` is a plain ADK `LiteLlm`, it needs no new content-generation
logic — `get_llm()` and `generate_text()` dispatch to whichever sibling
`Config.use_sage_gateway` selects (`get_sage_llm()` vs.
`get_sage_gateway_llm()`), and everything downstream (`_generate_via_sage_async`,
the loop-binding dance below, the response diagnostic) is unchanged. The flag
is off by default, so the direct-agent path is reproduced exactly unless a
caller opts in.

### Request timeout (`SAGE_TIMEOUT`, set to 40s)

The Sage SDK applies `SAGE_TIMEOUT` (its own env var, read directly — not through
this repo's `Config`, same as `LLM_GATEWAY_*`) as an **aiohttp total-request
timeout**, defaulting to 30s. That default is too tight here, and the reason is
specific rather than general slowness: measurement showed the only call shape that
ever reaches the ceiling is the **escalation-triage agent writing its brief**
(~1000 characters of prose, ~10s median, tail reaching 30–31s). A root-agent tool
call is ~2.5s and never timed out; the root agent's own ~1000-character narration
is ~5.3s and never timed out — so it is not output length alone, the triage task
itself is heavier to reason through.

A timeout there does not degrade one call; it aborts the whole agent run. `.env`
therefore sets `SAGE_TIMEOUT=40`, clearing the observed tail while still failing a
genuinely wedged request promptly. This raises the ceiling — it does not make
anything faster; the latency work itself is the triage changes above.

### Retrying a transient request failure (`Config.sage_request_attempts`, default 2)

Raising the timeout only helps a call that is *slow*. A call that genuinely blows
past 40s is lost — and it takes the whole agent turn with it, because nothing
downstream re-issues it.

ADK's eval harness *intends* otherwise: it registers a plugin that sets
`HttpRetryOptions(attempts=7, …)` on the request. But that is a **google-genai**
construct, and ADK's `LiteLlm` never reads `retry_options` or `http_options` and
passes no `num_retries` to litellm — so on the sage path the retry is configured
and silently ignored. Verified against the installed google-adk: zero references
to either field in `lite_llm.py`.

litellm's own retry does work here, because a sage request timeout surfaces as
`litellm.APIConnectionError`, which subclasses `openai.APIError` — one of the
three types litellm's async wrapper retries once `num_retries` is set. And
`LiteLlm` merges its `_additional_args` into the litellm call, so setting the key
there is enough: no wrapping, no patching.

`_apply_request_retries` (`shared/llm.py`) does exactly that, translating
*attempts* (what a human reasons about) into litellm's *retries* (`attempts - 1`).
`sage_request_attempts = 1` disables it and reproduces prior behavior exactly.
The retry is immediate — litellm picks `constant_retry` with no backoff for an
`APIError`, which is the right shape for a latency spike (a rate-limit error gets
exponential backoff instead). Bounded on purpose: with `SAGE_TIMEOUT=40` the worst
case is 2 × 40s on one call; if a second attempt also times out, the backend is
genuinely unwell and failing is the honest outcome.

### Array-wrapped tool-call arguments (`Config.repair_tool_call_args`, on by default)

A tool call's arguments are a *named mapping* — ADK builds a genai `FunctionCall`
from them, and pydantic requires a `dict`. The sage backend intermittently emits
them wrapped in a JSON array beside a stray sibling; observed verbatim on an
`escalation_triage` call:

```
[{"request": "{...the decision JSON, including rejected_alternatives...}"},
 ["RTE-4200 …", "RTE-4110 - Downtown / Midtown (WED): infeasible — truck capacity"]]
```

The first element is the complete, correct argument object; the second is a
fragment of the escaped JSON *inside* it that leaked to the top level (those
strings duplicate the `rejected_alternatives` array within `request`, so nothing
is lost by dropping them). ADK's own `_parse_tool_call_arguments` repairs several
malformed payloads, but this one is **valid JSON** — it parses cleanly to a
`list`, is handed to `types.Part.from_function_call(args=<list>)`, and pydantic
raises. That exception escapes the entire agent run: in eval the case is silently
dropped (a green suite that scored fewer cases than it appears to), on a live turn
the turn dies. No retry helps — a `ValidationError` is not a retryable API error.

`shared/llm.py`'s `_install_litellm_tool_args_repair` wraps that one ADK function.
The repair is deliberately narrow, because a bare array carries no parameter names
and so cannot be a valid argument set for *any* tool — it is provably debris, not
data:

| Payload | Behavior |
|---|---|
| A well-formed object | Returned untouched — the same object, not a copy |
| An array with exactly one object | That object is used; the discarded siblings are logged |
| Anything else (no object, or several) | Passed through unchanged → ADK raises, loudly, as before |

An argument value is never synthesized, and the wrapper never raises: any
unexpected failure inside it falls through to the value ADK would have used.
Unlike the opt-in flags elsewhere in this document it defaults **on**
(`SMART_ASSIGNMENT_REPAIR_TOOL_CALL_ARGS=false` to disable), because it provably
cannot change a healthy call — it only ever fires on a payload that would
otherwise crash.

### Recovering from a failed model or tool call (`Config.recover_from_agent_errors`, on by default)

The repair above closes the *one* payload shape it can read with certainty. Every
other shape — and every unrelated model failure — still reached ADK, and ADK
re-raises: `base_llm_flow` re-raises the model error, `functions` re-raises the
tool error, and either one unwinds the whole Runner. A single malformed reply
therefore destroyed an entire turn *whose deterministic pipeline had already
succeeded*, and the web app replaced the agent's real answer with a deterministic
result that could contradict what the user had just been shown.

`agent_callbacks.py` installs ADK's two error hooks on `root_agent` and the batch
agent. They answer differently on purpose:

| Hook | Returns | Effect |
|---|---|---|
| `on_model_error_callback` | an `LlmResponse` | The turn ends with a plain reply instead of an exception. It has no function calls, so `Event.is_final_response()` is True and the flow's loop terminates — no retry semantics, no spin |
| `on_tool_error_callback` | `{"ok": false, "error": …}` | The same result shape every pipeline tool already returns, so `webapp.llm_chat._tool_outcome` marks that step failed with the real reason and the conversation continues |

The model-error response deliberately carries **both** `content` and
`error_code`. Content is required because the web app only emits a chat frame for
an event with `content.parts` — an error-code-only response would render a blank
turn, worse than the failure it replaces. The error code is required because
`Event` subclasses `LlmResponse`, so it is readable downstream: both
`webapp/llm_chat.py` and `batch/agent_runner.py` check it before capturing text,
so a recovery notice is shown to the user but **never** recorded as the agent's
reasoning for a decision it did not explain.

**The triage sub-agent deliberately gets neither hook.** `AgentTool` returns the
sub-agent's last content as the tool result, and the root instruction relays the
brief *verbatim* to a specialist — a model-error callback there would hand that
specialist an apology dressed as an escalation brief. Letting it raise into
`root_agent`'s tool-error hook turns the same failure into an honest failed-tool
result instead.

**Recovery must not cost the deterministic floor.** Suppressing the exception also
stops it reaching `app.chat`, whose `except` clause is what runs the deterministic
brain. So `stream_turn` splits on whether the turn produced anything:

| Model fails… | Behavior | Why |
|---|---|---|
| **after** a decision | notice shown, agent's own result cards still render, no fallback | The pipeline result is real and audited; re-running would replace it with a second answer that could contradict it |
| **before** a decision | `AgentTurnUnavailable` is raised, nothing is emitted | The turn produced nothing, so `app.chat` answers from the deterministic brain exactly as before — otherwise the user would be told to retry where they used to get a real answer |

Verified live end-to-end against sage by corrupting one real reply into the
unrepairable two-object array shape and driving `/api/chat`: with the failure
*before* a decision the user gets deterministic result cards whether recovery is
on or off (no regression), and with it *after* a decision recovery keeps the
agent's own decision instead of discarding it for a deterministic re-run.

Safe for an unattended batch run: `batch/agent_runner._run_one_via_agent` keys its
outcome off the decision stored in session state, so a turn that ends early
without one still degrades to the deterministic pipeline exactly as before.

Defaults **on** for the same reason as the repair above: it can only fire on a
path that is already an unhandled exception. With
`SMART_ASSIGNMENT_RECOVER_FROM_AGENT_ERRORS=false` the agents are constructed with
no callbacks at all and ADK raises exactly as it used to. The exception is always
logged with its traceback; no decision is suppressed and no value is invented.

## Tracing & observability (`shared/tracing.py`, opt-in)

The grounded decision layers already produce an auditable *record* of every
choice (evidence packet in, cited choice out, verifier verdict, fallback
reason). This layer makes that record *observable at runtime* by emitting an
OpenTelemetry span per LLM call, exportable to a self-hosted trace backend
(the chosen stack is a self-hosted Langfuse instance) so a human can inspect
production decisions in a UI instead of only in logs.

It is deliberately the thinnest possible seam, and holds the repo's standard
guarantees:

- **Opt-in, default off.** Gated by `Config.use_tracing` (env
  `SMART_ASSIGNMENT_USE_TRACING`). With it off, no OpenTelemetry SDK is imported
  and behavior is byte-identical to before.
- **Never worse than the baseline.** Tracing *observes*; it never changes a value
  a decision layer acts on. Every failure path — SDK not installed, no exporter
  configured, backend unreachable, span machinery erroring — degrades to a silent
  no-op (`llm_span` yields a `_NoopSpan`), so a broken trace backend can never
  break a decision. A caller exception still propagates unchanged (and is recorded
  on the span when one is active).
- **Credential-free import.** The SDK, exporter, and instrumentor are imported
  lazily inside a once-per-process `_configure`, so importing the package needs
  neither the `observability` extra nor any credentials (the same discipline as
  the lazy Sage/backend construction elsewhere).

**Two span sources, one connected trace.** Setup is `configure_tracing(config)`
(idempotent, once per process), called from `_build_root_agent` (`agent.py`)
*before* the agent runs — the one entry point every agent-serving surface
(`adk web`/`adk deploy`, the web app) shares — and lazily by `llm_span` for
non-agent paths. It installs a **global** `TracerProvider` + OTLP exporter and
the Google ADK OpenTelemetry instrumentor, so both span sources land in one
trace tree:

```
configure_tracing(config)                          [shared/tracing.py]
  ├─ global TracerProvider + OTLP exporter   (non-clobbering: attaches to an
  │                                            existing provider if one is set)
  └─ GoogleADKInstrumentor().instrument()    (agent turns + tool calls)

root_agent turn  ─►  tool call (recommend_or_escalate, …)   [ADK spans]
                        └─ generate_text(config, prompt, role)     [shared/llm.py]
                             └─ llm_span(...)  backend/model/role/prompt_chars…
                                (nests UNDER the ADK tool span)    [our span]
        · flag off -> nullcontext(_NoopSpan)   (no import, no-op)
```

**Why a global provider (the Phase 0.5 promotion).** Phase 0 deliberately used a
*local* `TracerProvider` to avoid claiming the process-global one before the ADK
instrumentor existed. ADK's built-in tracing emits against the **global**
provider, so capturing agent/tool spans *and* connecting them to our
grounded-call spans requires sharing it. `_install_provider` claims the global
provider when none is set, and otherwise **attaches its exporter to whatever
provider is already there** rather than replacing it (OpenTelemetry forbids
re-setting a real provider, and clobbering would drop the other side's spans) —
robust in a deployment that already configured its own tracing.

**Generic spans only, by design.** It records backend, model, an optional
`role` label, prompt/response sizes, latency, and error status — but **not**
prompt or response text, which can carry customer PII (an evidence packet
contains an address). Richer, per-layer payloads are an intentional per-call-site
decision for a later phase, not a global default here.

**Exporter is vendor-neutral.** The target comes from the environment, not from
code: standard `OTEL_EXPORTER_OTLP_ENDPOINT` (+ `OTEL_EXPORTER_OTLP_HEADERS`)
takes precedence, keeping the backend swappable; as a convenience, the
`LANGFUSE_HOST`/`LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` trio is turned into an
OTLP endpoint + Basic-auth header (Langfuse ingests OpenTelemetry directly). So
pointing dev at `localhost:3000` vs. prod at a Cloud Run instance is pure config.
Install with the `observability` extra (`pip install -e ".[observability]"`).

## Human feedback loop (`feedback/` package, opt-in)

The tracing layer above makes a decision *observable*; this layer lets a human
*judge* it and feeds that judgment back into the eval machinery — the production
feedback flywheel (traces → human labels → dataset curation → calibrate evals →
tune), built to the repo's standard guarantees.

It is deliberately **vendor-free**. OpenTelemetry has no standard
annotation/score signal, and every vendor's annotation REST API is
vendor-specific — so a feedback item is represented the one portable way:
feedback arrives *after* the decision span has already closed (a human clicks
👎 seconds later, and an exported span can't be mutated), so each annotation is
emitted as its **own OTLP span**, `human_feedback`, **linked** to the decision's
span via a standard OpenTelemetry span link and the original `trace_id`. Any
OTLP backend — Phoenix now, Langfuse later, Tempo/Jaeger/anything after —
ingests it and correlates by trace id, with only the exporter *endpoint*
differing. This reuses the exact exporter/provider seam in `shared/tracing.py`;
no vendor SDK is imported.

```
decision runs INSIDE one span (webapp.recommendation)   [pipeline, unchanged]
        |  webapp/decision.traced_decision wraps the run and captures that span's
        |  trace/span ids WHILE it is live; webapp/decision.feedback_context pulls
        |  the recommend/escalate outcome + route/window/order from the result.
        |  both ride the payload as private `_trace` / `_decision` hints that
        |  app._attach_feedback consumes and strips (they never reach the browser),
        |  minting a stable decision_id.
        v
pick 👍/👎, add an optional note, click "Send"   [static/feedback.js — one shared
        |   widget on BOTH the Live-agent result card AND the Customer view
        |   (/frontend, the end-user surface); shown only when feedback is on.
        |   Rating + note submit together on the button, so a note is never lost.]
        v
POST /api/feedback  (webapp/app.py, flag-gated)
        v
record_feedback(config, record)              [feedback/capture.py]
   1. gate on use_human_feedback (off -> no-op, imports nothing further)
   2. validate deterministically (feedback/schema.py) -> 400 on a bad record
   3. scrub PII if feedback_scrub_pii (feedback/scrub.py; default ON)
   4. PERSIST FIRST to the append-only JSONL log (feedback/store.py) -- the
      durable audit source of truth, independent of any backend
   5. best-effort OTLP emit (feedback/emit.py) -- silent no-op if tracing off
        v
scripts/curate_feedback.py  ->  feedback/curate.py                 [OFFLINE]
   read HUMAN labels -> candidate eval cases aligned to eval/golden_cases.py
   (a human reviews + promotes; nothing auto-mutates the golden set or a prompt)
```

**How the guarantees hold.** *Opt-in, default off* — everything is gated by
`Config.use_human_feedback` (env `SMART_ASSIGNMENT_USE_HUMAN_FEEDBACK`); flag-off
hides the UI (advertised via `/api/mode`), disables the endpoint, and imports
nothing new. *Never worse than the baseline* — feedback is purely observational;
it touches no route, score, slot, or decision, and any *use* of the labels
(eval calibration, prompt tuning) is a separate, **offline, human-driven** step,
never a live loop that mutates what the system does. *No fabricated actionable
values* — a record carries a human judgment, never a value a downstream system
acts on, and a deterministic validator (`schema.validate_feedback`) rejects a
malformed one before anything persists it. *Auditable & durable* — the JSONL log
is the source of truth, written before the best-effort emit, so an annotation
survives a down trace backend. *Credential-free, defensive* — lazy imports, and
every persistence/emit failure degrades to a logged no-op (only a bad record
raises, as a 400), the same discipline as `shared/tracing.py`. *A layer changes
only what it owns* — `feedback/` only writes records.

**PII is a toggle, not a policy — and it's consistent across log and trace.**
Only the freeform `note` and free-text context values are scrub-eligible;
labels/route-ids always pass through. `feedback_scrub_pii` defaults **on** (safe
for an off-network / shared deployment): the note is redacted in the durable log
*and* the OTLP span carries only a `has_note` boolean, never the text. On a
trusted company network, where the real customer PII is *wanted* as part of the
feedback, set `SMART_ASSIGNMENT_FEEDBACK_SCRUB_PII=false`: records are stored
verbatim **and** the note text rides on the `human_feedback` span (visible in
Phoenix/Langfuse). So the one toggle governs PII everywhere, rather than the span
being unconditionally text-free.

**Real trace linkage, on every path.** A feedback item must link to a *real*
trace, but feedback arrives after the decision span closed. So `traced_decision`
(`webapp/decision.py`) runs the pipeline inside one explicit
`webapp.recommendation` span and reads its coordinates *while the span is live*,
threading them onto the payload — rather than best-effort-reading a span that may
already be gone at emit time. Both request paths use it (the streaming chat
services and `/api/recommend`), so the link is populated whenever tracing is on,
regardless of which brain served the turn. With tracing off the span is a no-op
and feedback falls back to the always-present `decision_id`. The
`webapp.recommendation` span also carries the decision's non-PII facts (outcome,
route, window, order size) as `smart_assignment.decision.*` attributes, so it's
informative in the trace backend even in offline deterministic mode (where there
are no child LLM/tool spans). The same helper module's `feedback_context` puts
the recommend/escalate **outcome** (plus route, window, order size) into the
curation snapshot on every path — so a 👎 on a streamed chat result carries the
same structured facts a `/api/recommend` one did. These travel as private
`_trace`/`_decision` payload hints that `app._attach_feedback` consumes and
always strips.

**Replay-ready trace datasets, still vendor-free (`use_trace_dataset_payloads`).**
Filtering thumbs-down spans is enough to *triage*, but to *curate a dataset
inside* Phoenix/Langfuse you need the case's input and output on the trace. When
`use_trace_dataset_payloads` is on **and** PII scrub is off, `DecisionSpan.record`
attaches the intake and the recommendation to the `webapp.recommendation` span as
OpenInference `input.value` / `output.value` (with `openinference.span.kind`).
These are *open* semantic-convention keys — Phoenix and Langfuse both read them
natively to build replay-able dataset examples — so the feature is backend-native
yet imports no vendor SDK. It's a pure opt-in on top of tracing, and scrub-on
always suppresses it (the payload carries name/address), so no PII reaches a trace
unless the operator opted into *both* flags. The vendor-free JSONL curation path
(`scripts/curate_feedback.py`) is unaffected and remains the portable default;
this just makes the *backend-native* curation path viable too.

**One neutral pipe for all annotators.** The schema's `annotator_kind ∈ {HUMAN,
LLM, CODE}` means an LLM-as-judge score or a deterministic code check can flow
through the *same* record, log, and OTLP span later — so the existing eval
judges (`eval/deepeval_llm.py`, `eval/sage_judge_llm.py`) can unify with human
ground truth without a second mechanism.

**The machine half of that ground truth (`eval/judge_log.py`).** The automated
judges now *record* what they scored, instead of discarding it: every verdict
from `eval/test_quality.py` and `eval/test_rationale_faithfulness.py` — pass and
fail — is appended to `Config.judge_log_path` (default
`feedback_data/judge_verdicts.jsonl`), one self-describing JSONL record per line,
deliberately mirroring `feedback/store.py`'s format and its defensive-write
discipline. The two logs are counterparts: human labels and machine verdicts on
the same quality dimensions, joining on `(decision_id, dimension)` — which is
exactly the pair `eval/judge_calibration.py` needs, and the input
`scripts/calibrate_judges.py` documents but nothing previously produced. They
stay *separate files* on purpose: the annotations log is the production audit
trail of judgments on real customer decisions, this is eval-run output, and
merging them would mix provenance domains and let machine rows reach human-label
curation. Each record separates `judge` (the resolved `ROLE_QUALITY_JUDGE` model
and backend) from `run` (the dataset + product model provenance
`eval/dataset.run_provenance` already stamps on captures), so a score change is
attributable to the agent, the judge, or the data rather than guessed. Recording
is observational — it changes no score or test result, needs no `use_*` flag
because it cannot regress behavior, and the path itself is the switch (empty
records nothing); a failed *write* is swallowed, while a failed *judge call*
still fails the eval. Curation (`feedback/curate.py`) only
reads HUMAN records — those are the ground truth the auto-judges calibrate
against — and emits *candidate* cases for a human to review and promote into
`eval/golden_cases.py`. A `suggested_expected_outcome` is filled in only when the
verdict cleanly implies one: a 👍 confirms the observed outcome as ground truth.
A 👎 is left for the human to decide — a thumbs-down says the decision was wrong,
not *how* (the reviewer may have wanted an escalation, or simply a different
feasible route/slot), so guessing `escalate` would encode a target they never
chose. The boundary is the point: human feedback feeds an offline, human-gated
loop.

**From candidates to a runnable eval — no hand-copying.** Both curation entry
points emit the *same* candidate-cases JSON: `scripts/curate_feedback.py`
(vendor-free, over the JSONL log) and `scripts/phoenix_curate.py` (the Phoenix
path — it joins the `human_feedback` and `webapp.recommendation` spans by trace id
so you don't do it by hand, and writes that same file, optionally also uploading a
Phoenix Dataset). `eval/case_source.py` loads either file, reconstructing a
`CustomerProfile` (including the stated day/window, now carried in
`feedback_context`) and a `GoldenCase` per candidate; it *skips* any case whose
address is missing or PII-redacted (a scrub-on capture can't be geocoded) and
reports why. `python3 -m eval.build_evalset --cases <file>` then turns those into a
standard ADK evalset JSON — so curated production feedback runs through the exact
same trajectory eval as the built-in `GOLDEN_CASES`, without editing
`golden_cases.py`. The committed golden dataset and its sync test are untouched
(the flag-less `build_evalset` still regenerates exactly that).

### Judge calibration — trusting the auto-judges (Phase 0, advisory)

The automated judges (`brief_quality`, `response_clarity`) are themselves LLMs, so
their scores are only worth gating on once benchmarked against human ground truth.
`eval/judge_calibration.py` measures that agreement — Cohen's κ (with a rubber-stamp
guard so a judge that passes everything on 👍-skewed labels scores ~0, not ~0.9), a
**dangerous-cell rate** (how often the judge passes what a human rejected), and a
trust band (`insufficient` / `distrust` / `advisory` / `gate`). It's purely
**advisory** and gated by `Config.use_judge_calibration` (default off): it changes
no decision and gates nothing; `scripts/calibrate_judges.py` is a no-op with the
flag off.

The crux is that human feedback is **holistic** (a thumb on the whole decision)
while judges are **dimensional**, so the harness never fabricates a per-judge label
from a thumb. It tiers the signal: an explicit per-dimension annotation (Tier 3)
wins; else a note-tag (Tier 2 — a transparent keyword map, plus an opt-in LLM
suggestion that degrades to keyword-only on any failure); else the thumb is routed
to the outcome-appropriate judge (Tier 1.5 — escalate→`brief_quality`,
recommend→`response_clarity`, the same split `test_quality.py` uses) and *only* that
one; and separately a Tier-1 **composite** predicts a thumb from all judge verdicts
and calibrates that against the holistic thumb. Every aligned pair is tagged with
its tier, so the sharp (dimensional) agreement reads separately from the coarse
(holistic) one. Dimension names are exactly the `deployment/phoenix/README.md`
vocabulary (and the judge names), so human label, Phoenix annotation, and judge
speak one language.

Human labels come through **one shape (`HumanLabel`) from any source** — vendor-free
(the JSONL log, where a Tier-3 annotation is a record with a `"<dimension>:<verdict>"`
label, no schema change), **Phoenix** (annotations on the decision trace, today),
or **Langfuse** (scores, later) — all normalized by the shared parser, with the live
client calls lazily imported and defensive. No replay and no data source: calibration
needs only the `(human_label, judge_verdict)` pairs that already exist.

**Where the judge half now comes from.** `verdicts_from_jsonl` reads the durable
judge log (`eval/judge_log.py`) the judges write as they run, so the harness's
verdict side is *produced by running the judges* rather than hand-authored:
`pytest eval/test_quality.py` then `scripts/calibrate_judges.py --verdicts
feedback_data/judge_verdicts.jsonl`. The CLI picks the reader by suffix, so the
precomputed `.json` mapping still works unchanged. Because the log is append-only,
the **latest line per `(decision_id, dimension)` wins** — the same "latest record
per decision" rule `feedback/curate.py` applies to the human log, so a case
re-judged five times weighs the same as one judged once rather than five times as
much. The join itself is `(decision_id, dimension)`, which is why a curated case
carries `GoldenCase.decision_id` end to end (`eval/case_source.py` lifts it from
the candidate's `provenance`, and the judge tests pass it to `measure_and_record`):
the minted `eval_id` only encodes its first 8 characters, so without the field the
link back to the human's label on that same decision would mean parsing an id out
of a name. A hand-written fixture has no `decision_id` — no human ever labeled it,
so there is nothing to join to, and it participates only as its own `eval_id`.

### Self-contained snapshot datasets — scoring the model, offline, in CI

Trajectory eval is world-independent, but scoring the *decision* (recommend vs.
escalate, and which route-slot) needs the world the decision saw. So a curated
golden dataset carries its own world — the file-backed analogue of the
code-defined `mock` world, and PII-free the same way. A **snapshot bundle** is one
directory:

```
eval/data/snapshots/<name>/
  routes.json    the world: Route/RouteStop with capacity, committed stops, windows, tiers
  geocode.json   {address -> {lat, lon}} for every case
  cases.json     the cases: intake + expected_outcome + expected_route_id/window
  manifest.json  provenance for visibility (source, model, config, counts)
```

`integrations/snapshot_data.py` owns the encoding; a **`snapshot` data source**
(`route_capacity_client`) serves `routes.json` and a **`SnapshotGeocoder`**
(`geocoding_client`) replays `geocode.json`, both pinned by
`eval/dataset.py` (which **auto-discovers** any bundle under `eval/data/snapshots/`
— dropping a directory registers a dataset, no code change — and hashes the bundle
bytes for a provenance `dataset_content_ref`). So replay is fully offline and
deterministic, exactly like `mock`.

**Two authoring on-ramps, one format, little manual work:**

```
 human feedback                          synthetic
 curate_feedback.py / phoenix_curate.py  eval/synthetic.py
   -> candidate-cases JSON                 designed world + prospects
          |                                        |
   eval/freeze_dataset.py                          |   (already PII-free)
   run each once vs the real world,                |
   capture its routes + coords, ANONYMIZE          |
          \________________________  _____________/
                                   \/
                    a self-contained snapshot bundle
                                   |
                    eval/outcome_scoring.py  ── run the CURRENT model vs the
                    (offline, deterministic)    frozen world; score outcome +
                                                route-slot vs the golden target
```

**Anonymization (the PII line).** Scoring depends on geometry and capacity, not
identities, so `freeze_dataset.py` keeps the coordinates / capacity / windows /
tiers / route-codes and drops the identifiers: the prospect's name and street
address become synthetic labels (the label keys the geocode map to the real
coordinates, so distance math is unchanged) and committed-stop customer numbers
become `STOP-*`. The result is PII-free *by construction* — safe to commit and run
in a shared CI. (Synthetic datasets are PII-free already, so they skip this.) One
shared world (the dedup union of every case's candidate routes), each prospect
evaluated against it — the `mock` pattern generalized. Golden targets come from
the human's corrected target on a promoted thumbs-down, else the decision captured
at freeze time (a regression baseline).

**Scoring, two paths, one toggle.** `eval/outcome_scoring.py` re-runs the current
model over a bundle and checks the recommend/escalate outcome and the route-slot
(route id + window) against the golden target. `path` (or
`SMART_ASSIGNMENT_EVAL_MODEL_PATH`) selects `deterministic` (weighted-sum, grounded
off — offline, no credentials, the **blocking self-contained CI gate** in the
`test` job) or `llm` (grounded judgment in the loop — advisory in the credentialed
`agent-eval` job). The scorer is side-effect-free (it restores the data-source /
geocoder env it pins). This closes the flywheel: production feedback (or a
synthetic design) → an anonymized, self-contained golden dataset → the current
model scored against it, automatically, in CI.

## Step 5: the route-slot decision, deterministic or grounded

Step 5 (recommend-or-escalate) always operates on the **deterministically
enumerated (route, slot) options** produced by step 4. What varies is only
*whether an LLM reasons over that set* — never what is in it.

```
                       hard constraints (constraints.py) -- ALWAYS run first,
                       the ONLY thing that can eliminate a candidate
                                     |
                       feasible / infeasible split (deterministic)
                                     |
                       every feasible (route, slot) pair scored
                       (shared/scoring.score_route_slot)
                                     |
                       routeslot/decide.decide_route_slot
                                     |
        +----------------------------+----------------------------+
        | grounded flags OFF          | grounded flags ON          |
        | highest total, gated on     | LLM picks / decides over   |
        | route_slot_score_threshold  | the SAME enumerated set,   |
        | (the reproducible floor)    | verified, with that floor  |
        |                             | as the fallback            |
        +----------------------------+----------------------------+
```

The LLM never free-generates a route, a window, or a score: it returns an
**index** into the enumerated menu plus citations, `routeslot/verifier.py`
checks the index is in range and every cited number matches the packet within
tolerance, and one corrective retry is allowed. Hard constraints have already
run, so it can never place a customer on an over-capacity or out-of-area route.
Any mechanical failure (unparseable output, an ungrounded claim surviving the
retry, a backend/credentials error) falls back to the exact deterministic
threshold result and logs why — so the grounded path is never *worse* than the
deterministic one, only better-reasoned when it succeeds.

### Config knobs (`shared/config.py`)

| Knob (env) | Default | Meaning |
|---|---|---|
| `SMART_ASSIGNMENT_USE_GROUNDED_ROUTE_SLOT_PICK` | `false` | Let the LLM pick the winning route-slot from the enumerated options instead of taking the top-scoring one. |
| `SMART_ASSIGNMENT_USE_GROUNDED_ROUTE_SLOT_ESCALATION` | `true` | Let the LLM make the recommend-vs-escalate call itself over **all** feasible route-slots, with the bar demoted to a reference fact. |
| `SMART_ASSIGNMENT_ROUTE_SLOT_SCORE_THRESHOLD` | `0.55` | The auto-assign bar. Gates the decision when escalation grounding is off; a reference fact (and the fallback) when it is on. |
| `SMART_ASSIGNMENT_JUDGMENT_SAMPLE_COUNT` | `3` | `k` — samples drawn for an escalation-side case (`1` disables resampling). |
| `SMART_ASSIGNMENT_JUDGMENT_CONSENSUS` | `unanimous` | How the `k` decisions clear back to a recommend: `unanimous` (precautionary) or `majority`. |
| `SMART_ASSIGNMENT_JUDGMENT_RETRY_ON_LOW_CONFIDENCE` | `true` | Whether a LOW-confidence *recommend* is escalation-side (resample) or ships as-is. A hard ESCALATE always resamples. |

**Where the flags take effect.** `run_slot_recommendation(...)` routes every
surface — the offline demo (`scripts/run_local.py`), the page generator, the web
app — through `decide_route_slot`, and the conversational tool
(`tools/slot_recommendation.recommend_or_escalate`) calls it directly. There is a
single decision path, so no surface can drift from another.

### Live breadcrumbs: which steps to show vs. whether they ran

While a turn streams, the chat shows a checklist of the pipeline steps
(`Intake` → `Geo-Lookup` → `Score & Rank` → `Recommend / Decide`). Two separate
questions decide what that checklist says, and they are answered by different
things on purpose:

- **Which steps appear** comes from `webapp/narration.TOOL_STEPS` — each tool's
  *declared* step list. This decouples the checklist from the tool count: the
  interactive flow goes straight from intake to `recommend_or_escalate`, which
  re-derives geo and scores internally, so that one call surfaces three steps.
  Steps are deduped per turn, so an on-demand `find_candidate_routes` before the
  decision doesn't show `Geo-Lookup` twice.
- **Whether a step ran** comes only from the tool's own result. A step opens as
  `"status": "running"` on the FunctionCall event, and is settled `done` or
  `failed` when the matching FunctionResponse arrives (paired by call id;
  `_tool_outcome` reads the `{"ok": ...}` dict the tools return, and relays the
  tool's own `error` text on a failure).

**The handoff phase.** An escalation turn has a second half the assignment steps
don't describe: `escalation_triage` composing the specialist brief, which is the
*longest* single call in the turn (~14s measured against the real agent, out of
~24s). Without a step of its own the panel sits fully ticked, captioned "Working
on it", for most of the turn. So the triage tool is a narrated step like any
other — and `narration.HANDOFF_STEPS` / `step_phase` mark it `"handoff"`, which
rides on its frames so the UI can style it as a change of hands (amber, matching
the `await` bubble it leads into) rather than a fifth pipeline step. The stepper
also captions itself *Waiting on a specialist* instead of *Done*, because an
escalated turn is parked on a human, not finished. Nothing is flag-gated: the
breadcrumb follows a real tool call, so it appears exactly when triage runs and
never when `Config.use_escalation_triage` is off.

Alongside it, a decision that reported `requires_human_review` closes its own step
with "Escalating for human review." — a restatement of a real field on the tool's
result (never an invented cause; the *reason* is the audited brief's job), so the
handoff row reads as a consequence rather than a surprise.

Keeping these apart matters: a static table cannot know that a geocode failed. If
completion were inferred from the call — or, in the browser, from "the next step
started" — the UI would mark `Geo-Lookup` and `Score & Rank` complete the instant
`recommend_or_escalate` was *requested*, hold them green through ~10s of work that
hadn't begun, and keep them green when the tool then returned an address error. A
breadcrumb saying a step finished is a claim about the audited run, so it is
grounded in a real fact the tool reported, exactly like every other value a user
sees. The browser mirrors this: `app.js`'s stepper keys rows by step name and
paints whatever status arrives, never inferring one step's outcome from another's.

### Deciding once per turn

The chat web app renders a turn twice: the agent calls `recommend_or_escalate`,
then `webapp/llm_chat._visualization_from_state` rebuilds the Simulator payload.
Steps 1–4 are deterministic, so re-deriving the candidates there keeps the
numbers drift-free — but **step 5 is not** once grounded reasoning is on, since
it samples and may resample for consensus. Re-deciding would therefore render a
second, independently-sampled outcome underneath the agent's narration of the
first, and record *that* one for feedback and tracing.

So the tool snapshots its decision into session state, bound to the exact profile
it was computed from (`SlotRecommendation.to_state_dict`, read back by
`tools.slot_recommendation.cached_decision_for`), and the visualization passes it
to `run_slot_recommendation(..., recommendation=...)`, which skips step 5
entirely. The snapshot is ignored — and the decision simply recomputed once —
whenever it is absent, belongs to a different profile (the prospect was revised
mid-conversation), or can't be parsed. Reuse is therefore config-independent: it
holds whether the decision was reached deterministically or by either grounded
path.

The rebuild is attempted only once the tool has **reported a successful
decision** — a call that failed (an address that won't geocode) never produced one
to render — and any error it still raises is caught and logged, yielding no result
cards. The visualization re-derives an answer the agent has already computed and
narrated; letting a failed rebuild take the turn down would discard that correct
reply and hand the user the deterministic fallback contradicting it.

**What changes on purpose when escalation grounding is on:** the fixed
`route_slot_score_threshold` no longer gates auto-assignment. The
escalate/recommend call is the LLM's, made from the raw facts; "should a human
look at this?" is answered by the model's own confidence plus cross-sample
agreement, not a fixed cutoff. The bar remains in the packet as a reference and
remains the deterministic fallback.

### Prospect isolation: the profile belongs to its address

`intake_customer` merges by design — a revision supplies only the fields that
changed. But the same merge once ran when a conversation moved on to a
**different customer** in the same session, so the previous prospect's unstated
fields followed the new one — observed live: a prospect who stated no delivery
preference was decided with the previous prospect's `TUE 07:00-10:00`, and a
phantom preferred day changes the candidate set, so it changes the decision.
`adk web`/`adk run` were worst off (one eternal session, no rotation layer at
all). The guards live in the tools every conversational surface shares
(`tools/slot_recommendation.py`), so no surface depends on a wrapper for
correctness:

- **Deterministic reset.** When intake receives an address that differs from
  the one on file (normalized compare) *and* that profile already produced a
  decision, the profile starts fresh from the passed fields and every
  prospect-scoped state key is cleared — the decision snapshot (else
  `cached_decision_for` could re-render the previous customer's outcome) and
  the triage grounding (else a brief could cite it). After a decision, a
  different address IS a different customer; no model judgment overrides this.
  Pre-decision address changes still merge, which is what the
  `resolve_address` confirmation flow relies on. Accepted trade-off:
  correcting an address *after* a recommendation re-asks for the order size —
  a visible re-ask over silently deciding with another customer's data.
- **`start_new_prospect`** — a no-argument tool the model calls when the user
  switches customers ("another one, 40 cases" is byte-identical to a revision
  at the tool level; only the model sees the words). It only *discards* state,
  so a spurious call costs a re-ask, never contamination — model judgment is
  never load-bearing for correctness. It is deliberately a separate tool, not
  an `intake_customer` parameter: the golden eval pins intake's argument dict
  exactly (an extra argument flaked it, measured live), while the `IN_ORDER`
  trajectory matcher tolerates extra tool *calls*. Interactive surfaces only —
  batch seeds a fresh session per prospect (its isolation rests on that, since
  `assign_prospect` reuses `intake_customer`'s merge internally) and keeps its
  tool surface byte-identical.

`tests/test_prospect_isolation.py` replays the leak through a real ADK Runner +
real tools with a scripted model (the `adk web` shape, no rotation anywhere);
neutering the guard makes it fail exactly as the pre-fix code did.

### Prospect rotation and session memory (opt-in)

A browser session can walk through many prospects in a row.
`webapp/llm_chat._maybe_rotate_prospect` starts a **fresh underlying ADK
conversation** whenever a new *address* arrives after the current prospect
concluded/escalated: it bumps a generation counter and suffixes the ADK session
id (`s1`, `s1#1`, …) while the browser's own `session_id` never changes. A
*revision* (no new address — "try 20 cases") carries none, so it stays in the
same conversation and keeps its context. This is why `adk web` (one eternal
session) remembers an earlier aside but the web app, by default, does not: the
rotation deliberately drops the prior transcript.

Rotation is **hygiene, not the correctness boundary**: its address parser
misses many natural phrasings ("new customer at `<address>` - 260 cases" does
not rotate), and that is fine because cross-prospect contamination is prevented
in the tools (see *Prospect isolation* above). What rotation still buys is a
bounded per-prospect transcript (the sage backend folds recent history into its
system prompt) and clean pending-escalation bookkeeping.

`Config.use_session_memory` (env `SMART_ASSIGNMENT_USE_SESSION_MEMORY`, **off by
default**) restores cross-prospect recall *without* touching rotation. It is
purely additive — the model gains recall, never a new actionable value:

- The app wires an ADK `InMemoryMemoryService` onto the `Runner`
  (`_get_memory_service`), and `root_agent` gains ADK's `preload_memory` tool
  (gated in `agent.py`), which auto-runs each turn, keyword-searches memory for
  the user's query, and injects the matches as `<PAST_CONVERSATIONS>` context.
- On rotation, `_ingest_current_into_memory` folds the concluding prospect's
  transcript into memory *before* the fresh session is minted, so its facts
  survive. That is the only ingest point, so the still-active prospect is never
  double-counted against what normal session replay already shows the model.
- Memory is keyed by `(app_name, user_id)`, so to scope recall to one browser
  (and never leak between browsers) the browser `session_id` becomes the ADK
  `user_id` while the flag is on (`_user_id_for`); off, the fixed `webapp_user`
  is used exactly as before.

With the flag off, no memory service is built (even an injected one is ignored),
no tool is added, and the fixed user id is used — flag-off reproduces today's
behavior exactly. If the runner has no memory service (e.g. a bare `adk web`
without a memory backend), `preload_memory` swallows the lookup and is a no-op.
