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
pip install -e ".[dev]"          # google-adk[extensions] for the litellm path
python3 -m eval.build_evalset    # regenerate the dataset if cases changed
pytest eval/test_eval.py
```

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
