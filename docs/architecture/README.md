# Architecture diagrams

Place a diagram of the agent's tool-calling flow here (e.g. `smart_assignment.png`).

The current architecture is a single ADK `LlmAgent` (`smart_assignment/agent.py`)
that talks to the user and calls one tool per step, in order:

```
intake_customer            (code — validate/merge address, cases, preferred slot)
find_candidate_routes      (code — geocode + Top-N nearest routes)
  -> address not found? -> agent calls resolve_address (grounded pick among the
                           geocoder's candidate matches) -> user confirms ->
                           intake_customer(confirmed) -> retry  [opt-in, default on]
evaluate_and_score_routes  (code — HARD constraints, then weighted scoring)
recommend_or_escalate      (code — rank + total-score gate -> decision + reasoning)
  -> requires_human_review? -> agent calls request_input (ADK built-in, human input)
```

The agent (the LLM) decides *when* to call which tool and narrates the
result in conversation; it never computes a distance, a constraint check, or
a score itself -- every number comes back from the tool call. See
`smart_assignment/tools/slot_recommendation.py` for the tool implementations
and `smart_assignment/prompts.py` for the instruction that enforces this.

## Delivery-slot selection (`shared/slot_selection.py`)

The prospect should be delivered *when the truck is already in their
neighborhood*, inferred from the route's nearest committed stops. Three
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
                           overlaps a stated preference -> this is the menu.
recommend_slot             pick one from the menu with a soft blend of
                           preference overlap + fit + low contention.
```

This replaced an earlier version that snapped the prospect to a route's nearest
*existing* window and anchored the slot at that window's start. The candidate
menu (`EvalContext.available_slots`, each `SlotOption` carrying its
`anchor_time`, `fit_score`, `committed_overlap`, `basis`) is exactly the set a
future recommendation LLM would reason over to pick the best slot.

**Phase A/B seam:** `stop_reference_time` is the single function that turns a
committed stop into a "when is the truck near here" clock value — today the TW1
window midpoint, later a real planned-arrival ETA (and, with a stop *sequence*,
the interpolation becomes true bracketing between the two sequential stops the
prospect is inserted between) — with no caller change. Knobs:
`SMART_ASSIGNMENT_SLOT_{NEIGHBORS,CLUSTER_GAP,WINDOW_MINUTES,CANDIDATES,WEIGHT_*}`.

## Grounded slot selection (`slotpick/` package)

The deterministic `recommend_slot` blend above is auditable but its weights
(`SLOT_WEIGHT_{FIT,CONTENTION,PREFERENCE}`) are hand-picked and can't adapt to a
situation the weights didn't anticipate. When `Config.use_grounded_slot_selection`
is on (env `SMART_ASSIGNMENT_USE_GROUNDED_SLOT_SELECTION`, default **off**), an LLM
picks the final window instead — but only from the *same* deterministically
enumerated menu, and only by index. It reasons over the evidence; it never
generates a window.

The weighted blend is **demoted, not removed**. Its per-candidate `blended_score`
and the index it would pick on its own (`deterministic_choice_index`) ride along
in the evidence packet as *reference* — a strong default the model is told to
agree with unless the other facts clearly justify diverging (and to say why in
its rationale when it does). And it stays the **fallback**: any parse/verify
failure or backend error reverts to exactly that blended pick. So the weights go
from being the sole, opaque decider to one grounded input among several, with the
LLM doing the reasoning over the valid set — while the auditable heuristic remains
the floor.

This is the same **constrained-option + grounded + deterministic-fallback**
pattern as the judgment and triage layers, applied to the slot pick:

```
route already chosen (route/score/decision are FINAL and untouched)
        |
        v
build_slot_packet   enumerate the winning route's available_slots -> for each,
 (evidence.py)      index + window + anchor_time + basis + numeric facts
                    (fit_score, committed_overlap, preference_overlap_minutes,
                    blended_score) + deterministic_choice_index
        |
        v
generate_slot_choice  LLM returns {chosen_index, rationale, citations[]}
 (prompts + llm.py)    -- "pick by index from this menu, cite the facts"
        |
        v
parse_slot_choice   strict shape (schema.py); one retry on parse/verify failure
verify_choice       chosen_index in range AND every cited (index, field, value)
 (verifier.py)      matches the packet within tolerance -> no fabricated numbers
        |
   ok / fail
        v
refine_slot         on OK: rewrite recommendation.recommended_window / _basis /
 (selector.py)      _window_rationale to the model's pick.
                    on fail / backend error: keep the deterministic blended pick
                    (logs a warning). Never worse than the flag being off.
```

`GroundedSlotSelector` wraps the call/parse/verify/fallback; `DeterministicSlotSelector`
is the trivial "return the blended `chosen_window`" default. `refine_slot(recommendation,
evaluations, customer, config, selector=None)` locates the winning route by
`recommended_route_id` and *only re-orders that route's already-computed candidate
slots* — it never changes the route, score, or decision, and is a no-op when there
is no recommended route (e.g. a no-feasible escalation) or the flag is off.

It's deliberately lightweight — a single grounded call + verify + fallback, no
resampling (unlike escalation-side judgment) — because a slot pick from a small
valid menu is low-stakes: the worst case is a suboptimal-but-feasible window, and
the deterministic pick is always there as the floor. Wired in two places behind the
flag: `pipeline.run_slot_recommendation` and the conversational
`tools/slot_recommendation.recommend_or_escalate` (which also serializes
`recommended_window_rationale`).

## Route-slot scoring (`routeslot/` package)

The layers above have a blind spot: scoring ranks **routes**, and slot contention
only enters *after* a route is chosen (slotpick). So a route can win on capacity
and clustering while its only workable slot is densely shared by high-value
customers — and the route ranker can't see that. When
`Config.use_route_slot_scoring` is on (env `SMART_ASSIGNMENT_USE_ROUTE_SLOT_SCORING`,
default **off** in code, on in `.env.example`), the **decision unit becomes the
(route, slot) pair**: every candidate slot on every feasible route is scored
separately, so slot availability influences which *route* wins.

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
                               – flag OFF (rollback): the 0.55 bar gates it; none ≥ bar
                                 -> ESCALATED_LOW_SCORE (deterministic best proposed, NO
                                 llm call); ≥1 ≥ bar -> RECOMMENDED, LLM picks among the
                                 eligible (above-bar) menu. ABSORBS the slotpick pass.
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
strongest option proposed, all reasoned takes in `alternative_takes`). This is the
same sampling+consensus machinery as `judgment/`, applied to the route-slot unit. On
**any** mechanical/verification failure it falls back to the deterministic threshold
decision, so it is never worse than the bar-gated baseline. Set the flag off to
reproduce the prior threshold-gated behavior exactly (the LLM then only picks among
above-bar options, as the box's rollback branch describes).

This is the same **constrained-option + grounded + deterministic-fallback**
pattern as judgment/triage/slotpick, applied to the route-slot unit; the weighted
total per route-slot is the reference and the fallback. The prior route-only path
(scoring, `judgment`, `slotpick`) is **untouched** and remains the rollback when
the flag is off — flag-off reproduces prior output exactly.

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

Because this rides on `use_route_slot_scoring` (opt-in, default off), the flag-off
route-only path never populates these fields and its output is unchanged.

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
        raw facts + any split model opinions)  -> composes a specialist brief
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
(`judgment`/`slotpick`/`routeslot`), not the `AgentTool` family (which is for the
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
| `judgment` | the grounded recommend/escalate decision | `SMART_ASSIGNMENT_MODEL_JUDGMENT` |
| `reasoning` | the LLM-narrated reasoning trace (`LLMReasoner`) | `SMART_ASSIGNMENT_MODEL_REASONING` |
| `slotpick` | the grounded delivery-slot pick (`slotpick/` package) | `SMART_ASSIGNMENT_MODEL_SLOTPICK` |
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
   ├─ (cooperative) calls check_brief_grounding(brief) -> revise until ok
   └─ (deterministic) after_model_callback (_finalize_brief):
        1. normalize_brief -> reflow the FINAL brief into the one canonical
           layout (headers/options/labels each on their own line)
        2. verify_brief -> if any figure/route is still ungrounded, append a
           caveat naming them ("⚠ Unverified — figures not found …")
```

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
guarantee (vs. the judgment layer, which hard-rejects + falls back because it
gates an auto-assign decision).

Reasoning (the natural-language trace on the final recommendation) is
produced deterministically inside `recommend_or_escalate` and then narrated
by the agent; the pipeline's own optional LLM-narrated reasoner
(`reasoning.LLMReasoner`, with a deterministic fallback) is a separate,
lower-level option used when calling `pipeline.run_slot_recommendation(...)`
directly (e.g. `scripts/run_local.py`), not by the conversational agent.

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
`_load_sage_registry` / `_load_sage_gateway_llm_cls`, sharing the
`_ensure_sage_sdk_on_syspath` local-workshop fallback), and because
`GatewayLlm` is a plain ADK `LiteLlm`, it needs no new content-generation
logic — `get_llm()` and `generate_text()` dispatch to whichever sibling
`Config.use_sage_gateway` selects (`get_sage_llm()` vs.
`get_sage_gateway_llm()`), and everything downstream (`_generate_via_sage_async`,
the loop-binding dance below, the response diagnostic) is unchanged. The flag
is off by default, so the direct-agent path is reproduced exactly unless a
caller opts in.

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
decision runs (recommend/escalate + slot)   [pipeline, unchanged]
        |  webapp stamps the visualization payload with a stable decision_id +
        |  best-effort trace coords (shared.tracing.current_trace_context)
        v
👍 / 👎 (+ optional note) on the result card   [static/app.js, shown only when
        |                                        Config.use_human_feedback is on]
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

**PII is a toggle, not a policy.** Only the freeform `note` and free-text
context values are scrub-eligible; labels/scores/route-ids always pass through,
and span attributes never carry note text regardless (matching tracing's
no-PII-on-spans stance). `feedback_scrub_pii` defaults **on** (safe for an
off-network / shared deployment); on a trusted company network, where the real
customer PII is *wanted* as part of the feedback, set
`SMART_ASSIGNMENT_FEEDBACK_SCRUB_PII=false` and records are stored verbatim.

**One neutral pipe for all annotators.** The schema's `annotator_kind ∈ {HUMAN,
LLM, CODE}` means an LLM-as-judge score or a deterministic code check can flow
through the *same* record, log, and OTLP span later — so the existing eval
judges (`eval/deepeval_llm.py`, `eval/sage_judge_llm.py`) can unify with human
ground truth without a second mechanism. Curation (`feedback/curate.py`) only
reads HUMAN records — those are the ground truth the auto-judges calibrate
against — and emits *candidate* cases (with a `suggested_expected_outcome` only
where the verdict cleanly implies one, e.g. a 👎 on a `recommend` → `escalate`)
for a human to review and promote into `eval/golden_cases.py`. The boundary is
the point: human feedback feeds an offline, human-gated loop.

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
  1. pick must be in the feasible set (hard safety net on top of the schema),
     and a RECOMMEND must be backed by >=1 citation on a route-varying fact of
     the picked route (no citation-padding via other routes / shared constants)
  2. every fact citation must resolve + match the packet (percent form allowed
     only for fraction-valued fields, so a figure can't shift magnitude 100x);
     every comparison must name two different routes and be arithmetically true
  3. tolerant prose scan: numbers (incl. "1,234" thousands), route-ids,
     "route N" mentions, day names, and HH:MM times in the rationale must all
     be grounded; unit-bearing figures ("84 miles") can't launder through
     percent normalization, and small counts with a unit ("5 cases") are checked
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
