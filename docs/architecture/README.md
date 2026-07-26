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

## Two ways to run the same workflow

The conversational agent above is one *entry point*, not the workflow itself. The
workflow is `pipeline.run_slot_recommendation`, and there are two ways in:

```
INTERACTIVE (agent.py + tools/)          HEADLESS (service.py)
a person, over several turns             a complete prospect record, one call
  root_agent picks the next tool           plain Python calls the steps in order
  ~5 model round-trips to sequence         0 model round-trips to sequence
  address confirmation, triage handoff     no conversation to have
        \                                       /
         \_____ pipeline.run_slot_recommendation
                  steps 1-4 deterministic
                  step 5 -> routeslot.decide_route_slot   <- IDENTICAL both ways
```

**Step 5 is the same call under the same `Config` on both paths**, so the
recommend-vs-escalate decision, the grounded reasoning, the verifier, the
resampling, and the deterministic fallback behave identically. The headless path
is cheaper because the LLM stops *sequencing deterministic steps*, not because it
stops reasoning: `decide_route_slot` has no ADK dependency and is driven entirely
by config, so the whole step-5 matrix comes across unchanged.

`service.assign(customer, config=...)` decides one prospect;
`service.assign_many(...)` decides a batch, fetching the route world once and
returning one outcome per prospect **in input order, with a failed prospect
reported rather than aborting the run** — a queue has to survive a bad row.
`service.from_salesforce_record(record)` maps a flat CRM record to a
`CustomerProfile`, kept as its own named function because upstream field names are
the part most likely to change. `scripts/run_assign.py` drives all of it from the
command line (`--samples`, `--file prospects.jsonl`, `--json`, `--html DIR`).

What the headless path deliberately does **not** do: correct an address (a
production record is trusted, and there is no user to confirm a suggestion with,
so a geocoding failure is reported via `AssignmentOutcome.error_kind`), and
compose the escalation brief (that is a separate on-demand step, so nothing
open-ended sits on a decision's critical path).

### One event loop per process (`service._llm_host_loop`)

`shared.llm.generate_text` is synchronous over an async backend. Reached from
ordinary synchronous code it takes `_run_coro_blocking`'s "no loop here" branch,
which calls `asyncio.run` — a fresh event loop per call, **closed** afterwards.
That is harmless once, but the sage backend caches an aiohttp session
process-wide bound to the first loop that touched it, so the second call raises
`RuntimeError: Event loop is closed`. A batch is exactly that pattern N times.

So the service keeps one long-lived loop for the process and records it as the
host loop, which makes every grounded call submit back to that single loop. It
**defers to a host loop a caller already established** — the web app records
uvicorn's loop via `offload_to_worker_thread`, and the session is bound *there* —
so running inside an async server is unaffected.

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
ground truth without a second mechanism. Curation (`feedback/curate.py`) only
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

**What changes on purpose when escalation grounding is on:** the fixed
`route_slot_score_threshold` no longer gates auto-assignment. The
escalate/recommend call is the LLM's, made from the raw facts; "should a human
look at this?" is answered by the model's own confidence plus cross-sample
agreement, not a fixed cutoff. The bar remains in the packet as a reference and
remains the deterministic fallback.
