# smart-assignment

Agentic workflow for automated delivery **slot assignment** for new
customers (Sysco context), built on Google's
[Agent Development Kit (ADK)](https://google.github.io/adk-docs/) and verified
against **google-adk 2.3.0**.

Given a new customer (address, order quantity in cases, and an optional
preferred slot — a **day of week plus a time window**), the workflow finds the
delivery route+day that best serves them — or escalates to a human specialist
when nothing fits or the call is too close.

> 📊 **Overview page for product owners:** a visual, tabbed walkthrough
> published via GitHub Pages at **https://muhammadvt.github.io/smart-assignment/** —
> an **Overview** tab, an **Architecture** tab (agentic workflow diagram), and a
> **Simulator** tab that spells out the scoring math and lets viewers **run the
> agent interactively** on a mock customer number. The page
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
│   │   ├── graph.py                    # ADK Workflow wrapper (delegates to pipeline)
│   │   └── nodes.py                    # ADK nodes (delegate to pipeline)
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
python3 scripts/run_local.py --customer 067-100002  # just one, by customer number
```

Customers are identified by a **Sysco customer number** in the form
`NNN-NNNNNN` (3-digit site/OpCo + 6-digit per-site number); see
`shared/customer.py`. The four bundled customers (all on mock site `067`) each
exercise a different branch:

| Customer number | Customer | Situation | Outcome |
|---|---|---|---|
| `067-100001` | Bayou City Bistro (downtown) | sits in the dense Central route, well under the capacity safety margin | **RECOMMENDED** (~97%) |
| `067-100002` | Galleria Grill & Catering | large catering order — only one nearby route can still take it, and even that route's own score is mediocre (getting quite full) | **ESCALATE – low total score** (~57%) |
| `067-100003` | Katy Prairie Steakhouse (far west) | all routes out of range / over capacity | **ESCALATE – no feasible slot** |
| `067-100004` | Woodlands Fresh Cafe | North route fits well but is getting full — the one demo route inside the capacity-buffer decay zone (81% utilized) | **RECOMMENDED** (~86%) |

### Reasoning: deterministic vs. LLM

Reasoning defaults to the **LLM layer** (`LLMReasoner`). When no
`GOOGLE_API_KEY` / Vertex credentials are present it transparently falls back
to a deterministic trace, so the demo always runs offline. To get real Gemini
narratives, `cp .env.example .env`, set `GOOGLE_API_KEY`, and re-run. To force
the deterministic reasoner in code, pass
`run_slot_recommendation(customer, reasoner=DeterministicReasoner())`.

## Running on the mock examples with `adk run` / `adk web`

`graph.py` wraps the same pipeline as an ADK `Workflow` (`root_agent`). Because
`adk run`/`adk web` send the agent a free-text message, the entry node accepts
a **customer number** (`NNN-NNNNNN`) and resolves it to one of the mock Sysco
customers. Names are never accepted — Sysco identifies customers by number.
Reasoning defaults to the LLM layer with a deterministic fallback, so no API
key is required to see output.

Valid inputs: `067-100001` … `067-100004` (unrecognized/blank input falls back
to the first customer).

**CLI (`adk run`)** — one-shot, prints the recommendation to the terminal:

```bash
adk run smart_assignment "067-100001"   # RECOMMENDED (~97%)
adk run smart_assignment "067-100002"   # ESCALATE - low total score (~57%)
adk run smart_assignment "067-100003"   # ESCALATE - no feasible slot
adk run smart_assignment "067-100004"   # RECOMMENDED (~86%)

# Omit the query for an interactive prompt (type a customer number, then Enter):
adk run smart_assignment
```

**Web UI (`adk web`)** — point it directly at the agent folder, open the URL,
pick `smart_assignment`, and type a customer number in the chat:

```bash
adk web smart_assignment          # serves http://127.0.0.1:8000
```

Sample `adk run` output (`067-100002`):

```
[smart_assignment_slot_recommendation]: Customer: Galleria Grill & Catering (067-100002)
Decision: ESCALATE -> human review (low total score)  |  total score 57%
Proposed slot: RTE-4200 (West Houston / Energy Corridor), WED, window 07:30-11:00
Score factors: geographic_clustering=0.67(w0.45)  capacity_buffer=0.39(w0.30)  window_match=0.60(w0.25)
Reasoning: For Galleria Grill & Catering, I recommend route RTE-4200 (West Houston / Energy
Corridor), delivering on Wednesday between 07:30-11:00. Geographically, it lines up reasonably
well with the stops already on this route — avg 5.0 mi to existing stops. The customer did not
name a preferred day or time, so I treated every option evenly on that front. On capacity, the
truck is getting quite full for this order — there's still some room, but not a lot of cushion —
150 cases of headroom left, putting the truck at about 84% full after this order (comfortably
safe up to 75%). It was also the only route that cleared every requirement, so there wasn't
anything else to weigh it against. Putting all of that together, this pick's total score comes
out to 57%, which falls short of the 60% bar I use before auto-assigning. Rather than commit on
my own, I'd like a specialist to take a quick look before this goes out.
```

Because every ADK node delegates to `pipeline.py`, the deployed graph and the
offline demo can never disagree on business logic. To get real Gemini
narratives instead of the deterministic fallback, set `GOOGLE_API_KEY` in
`.env` first. `adk deploy` uses the same `root_agent`.

## Testing

```bash
pytest tests/              # fast, deterministic unit tests — no LLM/network
```

Constraints, scoring, total-score gating, the end-to-end pipeline decisions,
and the ADK routers are all covered.

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
5. **Human-input UX** — the escalation nodes yield an ADK `RequestInput`; the
   client that surfaces it to an ops reviewer (dashboard, Slack, etc.) is out
   of scope for this pass.
