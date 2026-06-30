# smart-assignment

Agentic workflows for automated delivery day/time slot assignment for new
customers, built on Google's [Agent Development Kit (ADK)](https://google.github.io/adk-docs/).

Verified against **google-adk 2.3.0** (the current `pip install google-adk`
release as of this writing). Every import, the graph construction, and
the full test suite in this repo have been executed against the real
installed package — not just checked against documentation prose.

## Repo structure

This repo is structured to hold **multiple, architecturally different
agentic workflows** side by side, not just the one currently implemented.

```
smart-assignment/
├── smart_assignment/                # importable package
│   ├── agent.py                     # ADK entry point (root_agent)
│   ├── workflows/                   # one subpackage per agentic workflow
│   │   └── slot_recommendation/     # ← currently the only workflow
│   │       ├── graph.py             # the Workflow(edges=[...]) definition
│   │       ├── nodes.py             # function nodes + the one LLM Agent
│   │       └── prompts.py           # instruction text & output schemas
│   ├── shared/                      # cross-workflow code
│   │   ├── models.py                # data contracts
│   │   ├── tools.py                 # reusable deterministic functions
│   │   └── config.py                # env-driven thresholds/settings
│   └── integrations/                # external system clients
│       └── route_capacity_client.py # [MOCKED] route/capacity data source
├── deployment/                      # deploy.py + optional terraform/
├── eval/                            # AgentEvaluator golden-dataset evals
├── tests/                           # fast, deterministic unit tests
├── docs/architecture/               # per-workflow diagrams
└── scripts/run_local.py             # manual smoke-test entry point
```

**Why `workflows/<name>/` instead of flat files at the package root:**
each workflow owns whatever internal shape its orchestration pattern
needs — a graph (what `slot_recommendation` uses), a `SequentialAgent`/
`LoopAgent` pipeline, or a multi-agent hierarchy with `sub_agents/`. Adding
a second, differently-architected workflow means adding a new folder
under `workflows/`, without touching `slot_recommendation/` or `shared/`.

**Why `shared/` is separate from any one workflow:** functions like
`filter_feasible_slots` and data contracts like `RouteSlot` aren't
specific to the graph pattern — a future workflow built as a sequential
pipeline could call the exact same constraint-filtering function. This is
what makes the structure genuinely reusable across architectures, not
just parallel folders that don't talk to each other.

> **Honesty note:** I have not found a published Google reference repo
> that contains multiple, architecturally different workflows side by
> side in one package — every `google/adk-samples` entry is one agent
> (single pattern) per repo. The `workflows/<name>/` convention is an
> extrapolation from how Google structures sub-agents *within* a single
> complex sample (e.g. `data-science`'s `sub_agents/`), generalized one
> level up. It's ADK-idiomatic and follows real Python packaging
> practice, but it is not itself a documented Google standard — validate
> it against your team's actual second and third workflows once they
> exist.

### A note on flat vs. `src/` layout

This repo uses a **flat layout** (the `smart_assignment/` package sits at
the repo root, not under `src/`), matching the convention used across
Google's own `adk-samples` repo (`llm-auditor/`, `customer-service/`,
etc., and the ADK CLI's own project scaffolding). General Python
packaging guidance increasingly recommends a `src/` layout for
distributable library code, since it prevents accidentally importing the
in-development copy of the code instead of the installed one. This repo
is closer to a deployed application than a distributed library, and
matching the ADK ecosystem's own convention makes it easier to use
standard ADK tooling (`adk run`, `adk web`, `adk deploy`) without extra
path configuration. If this ever needs to be published as an installable
library for others to depend on, moving to `src/smart_assignment/` would
be the standard next step.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate        # or .venv\Scripts\activate on Windows
pip install -e ".[dev]"
cp .env.example .env             # then fill in GOOGLE_API_KEY or Vertex AI config
```

## Running

```bash
# Manual smoke test
python3 scripts/run_local.py --workflow slot_recommendation

# ADK's own CLI / Web UI
adk run smart_assignment
adk web smart_assignment
```

## Testing

```bash
pytest tests/              # fast, deterministic unit tests — no LLM calls
pytest eval/test_eval.py   # AgentEvaluator-based trajectory/response eval
```

All 11 unit tests pass as of this writing. The eval suite currently
contains a **placeholder** dataset (`eval/data/slot_recommendation.test.json`)
— no real route/capacity data was available to build a genuine golden
dataset. Populate it with real captured trajectories (via the ADK Web
UI's trajectory recording) before relying on it for regression detection.

## Architecture: `slot_recommendation`

```
START
  -> geocode_and_cluster_customer        (code)
  -> fetch_candidate_slots_node          (code — calls route capacity system)
  -> filter_feasible_slots_node          (code — HARD constraints only)
  -> route_on_feasibility                (code, conditional)
       NO_OPTIONS  -> escalate_no_feasible_slot   (human input)
       HAS_OPTIONS -> recommend_slot_agent        (LLM — ranks feasible options)
                        -> confidence_gate          (code, conditional)
                             LOW_CONFIDENCE  -> escalate_low_confidence (human input)
                             HIGH_CONFIDENCE -> format_output            (code)
```

Capacity, driver hours, and temperature compatibility are objectively
checkable, so they're plain Python functions (`shared/tools.py`), not
agent reasoning — an LLM is structurally unable to assign a stop that
violates them, not just instructed not to. Exactly one LLM call happens,
and only over options that already passed every hard constraint, to make
the genuinely subjective call: which feasible slot best serves
reliability and customer satisfaction.

## Key ADK 2.0 mechanics this design relies on (verified against installed package)

- `Event.output` passes data to the *immediately next* node only — it does
  not accumulate across the graph.
- `Event.state` persists across the whole workflow run and is how the
  customer profile / feasible options survive the LLM hop.
- `RequestInput` (from `google.adk.events`) implements human-in-the-loop
  nodes without invoking a model.
- `Agent.input_schema` must be a `pydantic.BaseModel` subclass (not a bare
  type like `str`) in this installed version — confirmed by instantiating
  the agent and fixing a validation error that the official docs example
  did not surface.
- `Event(route=[...])` stores routing data on `event.actions.route`, not
  `event.route` directly — confirmed by inspecting the real `Event`
  pydantic model fields.
- `AgentEvaluator.evaluate()` auto-discovers eval criteria from a
  `test_config.json` file in the **same folder** as the `.test.json`
  dataset — it is not passed as an explicit config-path argument (an
  earlier draft of this repo assumed otherwise; fixed after inspecting
  the real method signature and source).

## [ASSUMPTIONS REQUIRING CORRECTION]

1. **Route/capacity data source** (`integrations/route_capacity_client.py`)
   is entirely mocked. Highest-priority integration point — replace with
   a real call into your TMS/routing system.
2. **Geocoding/zone assignment** (`shared/tools.py: _naive_zone_bucket`)
   is a placeholder lat/lng bucket, not real territory logic.
3. **Capacity utilization buffer** (`SMART_ASSIGNMENT_MAX_UTILIZATION`,
   default 90%) is a guess, not a stated operational policy — confirm
   the real "do not exceed X% of rated capacity" rule with ops.
4. **`available_arrival_windows`** on `RouteSlot` assumes some upstream
   system already computes open slack windows within a route.
5. **Geographic fit score** is a crude proxy (count of committed stops).
   Real clustering quality should come from actual stop-insertion
   marginal cost/drive-time delta from the routing engine.
6. **Confidence threshold** (`SMART_ASSIGNMENT_CONFIDENCE_THRESHOLD`,
   default 0.70) for escalating to a human is an arbitrary starting
   point — tune against real human-override rates once available.

## Production-readiness gaps

- **Eval dataset is a placeholder**, as noted above.
- **No retry/error handling** around the route data fetch — a failed or
  slow call to the real capacity system should have explicit fallback
  behavior, not crash the graph.
- **No monitoring/observability wiring.** ADK supports logging, metrics,
  and traces (https://google.github.io/adk-docs/observability/) — none
  configured here. At minimum, track recommendation acceptance rate,
  human-override rate, and feasible-slot-found rate.
- **No authentication/authorization** around who can approve human-input
  escalations, and no durable audit trail beyond the session.
- **Human-input node UX is unspecified** — `RequestInput` yields a
  request that some client (web UI, Slack bot, ops dashboard) must
  surface and respond to; that client doesn't exist in this repo.
- **No load/concurrency testing** against a real, rate-limited capacity API.
- **Confidence calibration is unverified** — the model self-reports
  confidence with no evidence yet that it correlates with actual
  recommendation quality.
- **`deployment/deploy.py` has not been run against a real GCP project**
  as part of building this repo — verify project/region values and IAM
  permissions before running it in your environment.
