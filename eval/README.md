# Agent evaluation (`eval/`)

Regression tests for the **conversational agent** using ADK's `AgentEvaluator`.
Unlike the unit suite under `tests/` (deterministic, no model), these replay
scripted intake conversations against the real `root_agent`, so they **need a
live LLM backend** and are kept separate from the hermetic tests.

## What's here

| File | Purpose |
|---|---|
| `golden_cases.py` | The golden cases, built from `smart_assignment.mock_customers`: a natural-language intake message per fixture + the expected tool trajectory. |
| `build_evalset.py` | Deterministically renders the cases into an ADK `EvalSet` JSON. Run `python3 -m eval.build_evalset` to regenerate the dataset. |
| `data/slot_recommendation.test.json` | The generated `EvalSet` (do not hand-edit — regenerate). |
| `data/test_config.json` | The scoring criteria ADK auto-discovers from this folder. |
| `test_eval.py` | The pytest entry point that runs `AgentEvaluator`. |

## What is scored (and what isn't, yet)

**Phase 2a — trajectory only.** `test_config.json` sets `tool_trajectory_avg_score`
only. That checks the agent drives the pipeline correctly —
`intake_customer` → `find_candidate_routes` → `evaluate_and_score_routes` →
`recommend_or_escalate` — and catches structural regressions (a dropped or
reordered tool, or the address-resolution branch firing when it shouldn't).

That metric runs with `match_type: IN_ORDER`, not ADK's `EXACT` default: those
four calls must all appear, in that order, with exactly the expected args, but
**extra trailing calls are tolerated**. That matters because the two escalate
cases also hand off to a human — `escalation_triage` (when
`SMART_ASSIGNMENT_USE_ESCALATION_TRIAGE` is on, the default) and ADK's
`adk_request_input`. Their only arguments are model-authored prose that differs
every run, so they can't be pinned in the dataset without making the suite
permanently flaky. Under `EXACT` both escalate cases fail. See the comment on
`_PIPELINE_AFTER_INTAKE` in `golden_cases.py`.

`intake_customer`'s expected arguments are the **known ground-truth fields** of
each mock customer (derived from the fixture, not invented), so the trajectory
expectation is real. The agent's final natural-language **response is not scored
yet**: that text is the LLM's narration and can only be captured faithfully from
a real run — so `response_match_score` is intentionally absent here rather than
asserted against text we can't generate offline.

**Phase 2b — final-response quality.** A capture helper will run the real agent to
record the actual final responses into the dataset, and `response_match_score`
will be re-enabled. That step is deferred because it needs a live backend to
build and verify.

## Running locally

Needs a configured backend (see `.env.example`); the CI job uses
`sage-gemini-2.5-flash`.

```bash
pip install -e ".[dev,eval]"     # dev = test tooling; eval = google-adk[eval],
                                 # which ADK's AgentEvaluator needs at run time
python3 -m eval.build_evalset    # regenerate the dataset if cases changed
pytest eval/test_eval.py
```

The `eval` extra is separate from `dev` on purpose: it pulls ADK's evaluation
stack (scikit-learn, vertexai, rouge-score …), and the hermetic `tests/` suite
must never require it. Without it `AgentEvaluator` raises
`ModuleNotFoundError: Eval module is not installed`.

Trajectory scoring is identical on the mock demo routes, so this runs fine
without a data snapshot. If you have the prepared parquet cache under `data/dev/`
and want the agent to load it (the default `cache` data source) instead of
falling back to mock, also install the parquet engine: `pip install -e
".[cache]"` (adds `pyarrow`). Without it, the cache read fails and you'll see a
"using the mock demo routes instead" warning — expected, not an error.

### Running a subset locally while developing (cost control)

Every case replays the full agent pipeline against your live LLM backend, and
ADK's own default runs each case **twice** (`num_runs=2`) — so a plain
`pytest eval/test_eval.py` against all 4 committed cases is 8 live
conversations. Two env vars (unset by default, so normal behavior is
unchanged; **not used by CI**, which always evaluates the full committed
dataset) trim that while iterating:

```bash
# Just one case, one run each -- cheapest inner loop.
SMART_ASSIGNMENT_EVAL_IDS=bayou_city_bistro_recommend \
SMART_ASSIGNMENT_EVAL_NUM_RUNS=1 \
pytest eval/test_eval.py

# Multiple cases: comma-separate the eval_id (see golden_cases.py).
SMART_ASSIGNMENT_EVAL_IDS=bayou_city_bistro_recommend,galleria_grill_escalate_low_score \
pytest eval/test_eval.py
```

`SMART_ASSIGNMENT_EVAL_IDS` doesn't hand-edit the committed JSON (which can't
have comments and is checked by `tests/eval/test_build_evalset.py` for staying
in sync with `golden_cases.py`) — it renders a scratch subset from
`golden_cases.py` on the fly via the same `build_evalset` machinery that
produces the real file, so the subset can never drift from it, and nothing
under `eval/data/` is touched. See the docstring on `_eval_dataset_path` in
`test_eval.py` for exactly what it does.

## CI: advisory first

The `agent-eval` job in `.github/workflows/ci.yml` runs this on PRs but is
**advisory** (`continue-on-error: true`) — it reports, it does not block. It also
no-ops cleanly when the model credentials aren't configured as repo secrets, so
it never fails a PR for infrastructure reasons.

The plan is to keep it advisory until the trajectory thresholds prove stable over
a few real PRs, then flip it to a required check. Enabling Phase 2b's
response-match scoring is a separate step on the same dataset.

## Adding or changing cases

Edit `golden_cases.py`, then `python3 -m eval.build_evalset` to regenerate the
JSON, and commit both. `tests/eval/test_build_evalset.py` asserts the committed
JSON stays in sync with the builder and is schema-valid, so a stale hand-edit is
caught by the hermetic suite.

## Human feedback on top of eval runs

Trajectory scoring above is automated and deterministic; it can't judge things
like "was this escalation brief actually useful" or "is the final message
clear." For that, point a run of this suite at a self-hosted trace backend
(`SMART_ASSIGNMENT_USE_TRACING=true` + an OTLP endpoint, see `.env.example`) so
each replayed case's agent turns and decisions land as a trace, then have a
human score those traces. Two backend options, same workflow either way:

- `deployment/langfuse/README.md` — self-hosted Langfuse via Podman, with
  structured Annotation Queues for batch review.
- `deployment/phoenix/README.md` — self-hosted Arize Phoenix, no container
  runtime required; good when Podman/Docker isn't available.
