# smart-assignment

Agentic workflow for automated delivery **slot assignment** for new
customers (Sysco context), built on Google's
[Agent Development Kit (ADK)](https://google.github.io/adk-docs/) and verified
against **google-adk 2.3.0**.

Given a new customer (address, order quantity in cases, and an optional
preferred slot — a **day of week plus a time window**), the workflow finds the
delivery route+day that best serves them — or escalates to a human specialist
when nothing fits or the call is too close.

New customers are **prospects**: Salesforce/CRM has their address, but they
don't have a Sysco customer number yet. **Address is therefore the primary
identifier and the default way to run the workflow**; `customer_number` is an
optional placeholder field, only used/validated if an account already has one.

> 📊 **Overview page for product owners:** a visual, tabbed walkthrough
> published via GitHub Pages at **https://muhammadvt.github.io/smart-assignment/** —
> an **Overview** tab, an **Architecture** tab (agentic workflow diagram), and a
> **Simulator** tab that spells out the scoring math and lets viewers **run the
> agent interactively** on a mock customer's address. The page
> ([`docs/index.html`](docs/index.html)) is **generated from live workflow
> output** so it can't drift — regenerate it with
> `python3 scripts/generate_page.py` after changing mock data, rules, or config.

## The process (`slot_recommendation`)

```
1. Intake        collect address, order quantity (cases), optional slot (day+time)
2. Geo-Lookup    geocode the address, pick the Top-N nearest candidate routes
3. Constraints   drop routes failing any HARD constraint (deterministic code)
4. Score & Rank  weighted multi-factor scoring over the survivors
5. Recommend     output the top slot + full reasoning trace,
                 or ESCALATE (no feasible slot / low total score)
```

**Hard constraints (step 3) — objective, code-enforced, never LLM judgment:**

| Constraint | Rule |
|---|---|
| Geographic serviceability | customer within the route's service radius (and a global mileage cap) |
| Route capacity | utilization stays `<= 90%` after adding the order |

The customer's **preferred delivery slot — a day of week plus a time window —
is a soft preference, not a hard constraint**: it never eliminates a route; it
only feeds the `window_match` scoring factor below.

**Scoring factors (step 4) — weighted, in priority order:**

| Factor | Weight | Meaning |
|---|---|---|
| `geographic_clustering` | 0.45 | how tightly the customer clusters with the route's existing stops |
| `capacity_buffer` | 0.30 | flat (1.0) while utilization stays under a safety margin below the ceiling; decays linearly to 0 at the ceiling itself |
| `window_match` | 0.25 | how well the route matches the preferred **slot** — day of week + time window (soft preference) |

`capacity_buffer` is deliberately **not** a straight "more headroom always
scores higher" ratio — that biased recommendations toward near-empty trucks.
Instead it stays flat at 1.0 as long as utilization is at or below
`max_utilization_after_assignment - capacity_buffer_safety_margin` (90% − 15pp
= 75% by default), so two routes that are both comfortably safe score
identically; only a route that's genuinely approaching the ceiling is marked
down, decaying linearly to 0 exactly at the ceiling.

Constraints and scoring are **deterministic Python** — reproducible and
auditable. An LLM is *structurally unable* to place a customer onto a full
truck or outside the serviceable area, because those checks are code, not
prompting. The LLM's only (optional) job is turning the already-decided,
fully-quantified result into a fluent natural-language explanation.

## Architecture / decision flow

```
START
  -> intake_node                (intake + geocode + Top-N nearest routes)
  -> constraint_and_score_node  (hard constraints, then weighted scoring)
  -> route_on_feasibility       (conditional)
       NO_OPTIONS  -> escalate_no_feasible_slot     (human input)
       HAS_OPTIONS -> build_recommendation_node
                        -> total_score_gate           (conditional)
                             LOW_SCORE  -> escalate_low_score (human input)
                             HIGH_SCORE -> format_output
```

The winning route's own `total_score` from Step 4 **is** the gating number —
there's no separate "confidence" computed from how close a runner-up scored.
A route's own merit shouldn't be discounted just because another candidate
happened to score nearly as well: two routes tied at a high score both clear
the bar, and either is a safe pick; a route only gets flagged when *its own*
score is mediocre. Below `SMART_ASSIGNMENT_TOTAL_SCORE_THRESHOLD` (default
0.60) it escalates for a human sanity-check (a slot is still proposed, so the
reviewer has something to approve/override).

## Repo structure

```
smart-assignment/
├── smart_assignment/
│   ├── agent.py                        # ADK entry point (root_agent)
│   ├── mock_customers.py               # [MOCK] sample Sysco new-customer intakes
│   ├── shared/                         # cross-workflow, framework-agnostic core
│   │   ├── models.py                   # data contracts (CustomerProfile, Route, ...)
│   │   ├── customer.py                 # Sysco customer-number format (NNN-NNNNNN)
│   │   ├── geo.py                      # haversine + Geocoder protocol
│   │   ├── timeutils.py                # delivery-window overlap helpers
│   │   ├── constraints.py              # pluggable HARD constraints (step 3)
│   │   ├── scoring.py                  # pluggable weighted factors (step 4)
│   │   └── config.py                   # env-driven thresholds & weights
│   ├── workflows/slot_recommendation/
│   │   ├── pipeline.py                 # the 5-step orchestration (source of truth)
│   │   ├── reasoning.py                # total-score gating + pluggable reasoner (LLM default)
│   │   ├── prompts.py                  # LLM reasoning prompt text
│   │   ├── graph.py                    # ADK Workflow wrapper (delegates to pipeline) -- batch path
│   │   ├── nodes.py                    # ADK nodes (delegate to pipeline)
│   │   ├── conversational_agent.py     # ADK LlmAgent -- conversational path (default root_agent)
│   │   └── agent_tools.py              # tool wrappers around pipeline steps, for the LlmAgent
│   └── integrations/                   # [MOCKED] external systems
│       ├── route_capacity_client.py    # route/capacity data (TMS stand-in)
│       └── geocoding_client.py         # address -> lat/lng
├── scripts/run_local.py                # OFFLINE demo over the mock customers
├── tests/                              # fast, deterministic unit tests
└── deployment/ · eval/ · docs/
```

**Modularity by design:** each hard constraint and each scoring factor is a
small pure function in a registry (`HARD_CONSTRAINTS`, `SCORING_FACTORS`).
Add/remove/reweight one by editing a single list or a config weight — nothing
else changes. Every collaborator (route source, geocoder, reasoner, config) is
injected into `run_slot_recommendation(...)`, so pointing this at real systems
is a matter of passing different arguments, not editing logic.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

No API key or `.env` is needed for the demo or tests.

## Seeing the agent output on mock data

The demo runs the full pipeline over the mock Sysco customers and prints an
auditable trace for each — geocoding, the Top-N candidate routes, every
constraint outcome, the weighted score breakdown, and the final decision:

```bash
python3 scripts/run_local.py                        # all sample customers
python3 scripts/run_local.py --customer Westheimer   # just one, by address (default)
```

New customers are **prospects**, so they're identified by **address** by
default — a Sysco customer number (`NNN-NNNNNN`; see `shared/customer.py`) is
only matched if the `--customer` value happens to equal one already on file.
The four bundled customers exercise a different branch each:

| Customer | Address | Situation | Outcome |
|---|---|---|---|
| Bayou City Bistro (downtown) | 1200 McKinney St, Houston | sits in the dense Central route, well under the capacity safety margin | **RECOMMENDED** (~97%) |
| Galleria Grill & Catering | 5085 Westheimer Rd, Houston | large catering order — only one nearby route can still take it, and even that route's own score is mediocre (getting quite full) | **ESCALATE – low total score** (~57%) |
| Katy Prairie Steakhouse (far west) | 24600 Katy Fwy, Katy | all routes out of range / over capacity | **ESCALATE – no feasible slot** |
| Woodlands Fresh Cafe | 1201 Lake Woodlands Dr, The Woodlands | North route fits well but is getting full — the one demo route inside the capacity-buffer decay zone (81% utilized) | **RECOMMENDED** (~86%) |

### Reasoning: deterministic vs. LLM

Reasoning defaults to the **LLM layer** (`LLMReasoner`). When no
`GOOGLE_API_KEY` / Vertex credentials are present it transparently falls back
to a deterministic trace, so the demo always runs offline. To get real Gemini
narratives, `cp .env.example .env`, set `GOOGLE_API_KEY`, and re-run. To force
the deterministic reasoner in code, pass
`run_slot_recommendation(customer, reasoner=DeterministicReasoner())`.

## Talking to it conversationally with `adk run` / `adk web`

`smart_assignment/agent.py`'s `root_agent` — what `adk run`/`adk web` launch by
default — is a conversational ADK `LlmAgent`
(`workflows/slot_recommendation/conversational_agent.py`). It collects a
prospect's address, order quantity, and (optional) preferred slot over
multiple turns, then calls the exact same deterministic pipeline as **tools**
(`workflows/slot_recommendation/agent_tools.py`) — the model never computes a
distance, a constraint check, or a score itself, only when to call which tool
and how to narrate the result. Because it's LLM-driven end to end (not just
for narration), **this path requires `GOOGLE_API_KEY`**: `cp .env.example .env`
and set it first.

```bash
adk web smart_assignment          # serves http://127.0.0.1:8000, chat in the browser
adk run smart_assignment          # interactive terminal chat
```

A conversation walks through the same 5 steps as the batch pipeline, visibly:

```
you> New prospect at 5085 Westheimer Rd, Houston, TX 77056, 400 cases.
agent> [intake_customer] Got it — 5085 Westheimer Rd, 400 cases, no preferred slot yet.
agent> [find_candidate_routes] Nearest routes: RTE-4100 (5.9 mi), RTE-4200 (9.3 mi), RTE-4400 (13.3 mi).
agent> [evaluate_and_score_routes] Only RTE-4200 clears both hard rules; the others fail on
       capacity or service area.
agent> [recommend_or_escalate] RTE-4200 scores 57%, below the 60% auto-assign bar — I'd like a
       specialist to confirm before this goes out.
agent> [request_input] Pausing for a specialist's sign-off...
```
(Illustrative — exact wording depends on the model; the tool calls, numbers,
and decision are always real, straight from `agent_tools.py`.)

If the user then says "actually make it 140 cases", the agent calls
`intake_customer` again with just that field — the address is kept
automatically — and re-runs the scoring/recommendation steps, which now clear
the bar.

### The original deterministic batch path still works too

`workflows/slot_recommendation/graph.py`'s `root_agent` — the plain ADK
`Workflow` graph this repo started with — is unchanged: a one-shot, fully
offline path (LLM reasoning still falls back to the deterministic trace
without an API key) for a single free-text address in, one recommendation
out. Point `adk run`/`adk web` at it directly instead of the conversational
agent:

```bash
adk run smart_assignment/workflows/slot_recommendation "5085 Westheimer Rd, Houston, TX 77056"
adk web smart_assignment/workflows/slot_recommendation
```

See `scripts/run_local.py` above for the same batch behavior with zero ADK
runtime at all.

Both paths delegate to the same `pipeline.py`, so the conversational agent,
the batch graph, and the offline demo can never disagree on business logic.
`adk deploy` uses whichever `root_agent` you point it at.

## Testing

```bash
pytest tests/              # fast, deterministic unit tests — no LLM/network
```

Constraints, scoring, total-score gating, the end-to-end pipeline decisions,
the ADK routers, and the conversational tool wrappers (`agent_tools.py`,
called directly with a fake tool context -- no LLM needed) are all covered.

## [ASSUMPTIONS / MOCKS REQUIRING REPLACEMENT]

This is a **first-pass** on mock data. Highest-priority items to replace:

1. **Route/capacity source** (`integrations/route_capacity_client.py`) is
   mocked Houston data. Replace with a real Sysco TMS/routing integration —
   as long as it populates `Route`/`RouteStop`, nothing downstream changes.
2. **Geocoding** (`integrations/geocoding_client.py`) resolves a handful of
   demo addresses and otherwise returns a deterministic Houston-area point.
   Swap for a real geocoder implementing the `Geocoder` protocol.
3. **`geographic_clustering`** uses average distance to a route's committed
   stops as a proxy — real clustering quality should come from the routing
   engine's marginal stop-insertion cost / drive-time delta.
4. **Thresholds & weights** (90% utilization, 0.60 total-score threshold,
   service radius, factor weights) are starting points in `shared/config.py`,
   not validated Sysco policy — tune against real operational data.
5. **Human-input UX** — the conversational agent's escalation calls ADK's
   real `request_input` tool, which pauses and waits for a reply in whatever
   client is running the conversation (terminal, `adk web`, etc.); a
   dedicated ops reviewer surface (dashboard, Slack, etc.) is out of scope
   for this pass. The batch graph (`graph.py`) still just returns terminal
   text on escalation — it has no conversation to pause.
6. **Customer intake source** — new customers are prospects with no Sysco
   customer number yet, so their address (the primary lookup key) is assumed
   to be pulled from Salesforce/CRM. `customer_number` is an optional
   placeholder, matched only when this workflow is run for an account that
   already has one.
7. **Conversational guardrails are instruction-level, not code-level** — the
   conversational agent's system instruction tells the model to never state
   a number it didn't get from a tool call and to follow the 5 steps in
   order, but an LLM can still deviate. The business logic itself
   (`pipeline.py`) stays 100% deterministic Python regardless, so the worst
   case is a confused conversation, never a wrong decision silently computed
   by the model.
