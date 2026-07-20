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
| `data/captured_responses.json` | Committed `{eval_id: {final_response, escalated}}` map written by `capture.py` (Phase 2b). |
| `test_eval.py` | The pytest entry point that runs `AgentEvaluator` (trajectory, full dataset). |
| `capture.py` | Runs the live agent once per case to record its real final response + whether it escalated (Phase 2b). |
| `test_response_match.py` | Separate pytest entry point: `response_match_score`, scoped to captured cases known NOT to have escalated. See its module docstring for why escalate cases can't be scored this way at all. |
| `case_selection.py` | Shared `SMART_ASSIGNMENT_EVAL_IDS` subset filter used by `test_eval.py`, `capture.py`, and `test_quality.py` — one place owning that env var's name and validation. |
| `deepeval_llm.py` | `SmartAssignmentDeepEvalLLM` — adapts this repo's own `generate_text` (any `SMART_ASSIGNMENT_LLM_BACKEND`) to DeepEval's judge-model interface. |
| `test_quality.py` | Separate pytest entry point (Phase 3a): DeepEval G-Eval `brief_quality`/`response_clarity`, scored directly against captured `{final_response, escalated}` data — no ADK dataset involved. |

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

**Phase 2b — final-response quality.** `eval/capture.py` runs the real agent to
record the actual final responses, so the dataset's `final_response` fields can be
populated and `response_match_score` re-enabled. It needs a live backend, so it's
a deliberate, separate step from the deterministic `build_evalset.py`:

```bash
# Needs a configured backend (Sage creds, or a standard Gemini key) — see .env.example.
python3 -m eval.capture --check   # dry run: print what would be captured, write nothing
python3 -m eval.capture           # capture, then regenerate the dataset from it
```

This writes **`eval/data/captured_responses.json`** (a committed, reviewable
`{eval_id: {final_response, escalated}}` map — `escalated` records whether the
case handed off via ADK's `request_input` long-running tool rather than ending on
plain text; see `eval/test_response_match.py` below for why that matters) and
regenerates `eval/data/slot_recommendation.test.json` with `final_response`
populated from it. **Commit both.** Because `build_evalset.py` reads the
committed capture file (extracting just the text), the dataset stays byte-stable
and the hermetic sync test still holds; with the file absent (a fresh Phase-2a
checkout) `final_response` stays `null` and structural output is reproduced
exactly.

### Scoring `response_match_score` — `eval/test_response_match.py`, not `test_config.json`

Unlike trajectory scoring, `response_match_score` is **not** added to the shared
`eval/data/test_config.json`. Two reasons, both load-bearing:

1. It's dataset-wide, not per-case — enabling it there would apply to the full
   committed dataset (and CI, which always runs that), including any case whose
   `final_response` is still `null` (not yet captured). ADK doesn't error on a
   `null` reference; it silently scores that case `0.0` and `FAILED`, dragging the
   overall score down for reasons that have nothing to do with response quality.
2. **`response_match_score` cannot meaningfully score an ESCALATE-outcome case at
   all**, regardless of threshold. [VERIFIED against installed google-adk 2.5.0
   source]: an escalation ends the turn on ADK's `request_input` long-running tool
   call; `Event.is_final_response()` treats that tool-call event as the turn's
   final response, but its content holds a `function_call` part, not `.text` —
   so ADK's own live-eval extraction of the "actual" response is always `""` for
   an escalated case, forcing a `0.0` no matter how good the real handoff message
   was. `capture.py` works around this on the *reference* side only (it manually
   pulls the handoff `message` out of the tool call for the case it captures);
   there's no equivalent on the live/actual side during evaluation, and that's
   ADK-internal, not something this repo controls. See
   `eval/test_response_match.py`'s module docstring for the full trace through
   ADK's source.

So `eval/test_response_match.py` is a **separate** pytest entry point that scores
`response_match_score` against a scratch dataset containing only the captured
cases known to be `escalated: False` — real signal on recommend cases, no false
failures on escalate ones, and the committed `test_config.json` (and therefore
the default full-dataset run and CI) stay completely unaffected:

```bash
# Needs a configured backend, same as capture.py.
pytest eval/test_response_match.py
```

It **skips cleanly** (not a failure) when no case is yet known to be a clean
recommend — capture one (see the data-source caveat above: try a case still
documented as `recommend` in `golden_cases.py`, but be aware the real outcome may
differ) to unskip it. Its threshold (`0.5` as of writing, in the file itself, not
`test_config.json`) is a starting point, not calibrated — tune it once there's a
real distribution of scores to look at.

#### `final_response_match_v2` — the LLM-as-judge alternative, scored alongside v1

The same file also scores ADK's `final_response_match_v2`: instead of literal
ROUGE-1 word overlap, a judge LLM rates whether the response is valid given the
reference, tolerating paraphrasing/format/order differences — a materially
better signal for prose, at a materially higher cost (an extra LLM call per
sample; ADK's own default is 5 samples, majority-voted). It has the **exact same
escalate-case blind spot as v1** — [verified against installed google-adk 2.5.0
source] `llm_as_judge_utils.get_text_from_content` still only reads `.text` parts
of `Content`, same as v1, so it never sees the escalation handoff message either
(that lives in a `function_call`'s args) — so it's scoped by the same
recommend-only filter, not added anywhere broader.

`eval/test_response_match.py` pins `judge_model_options.num_samples = 1` (not
ADK's default of 5) to keep this cheap while there's only a handful of captured
cases; bump it once there's a reason to trust majority-vote stability over a
single judge call. It's also marked `@experimental` in ADK's own source — its
shape or behavior may move under future ADK versions.

> **Data source matters here.** Capture runs the real agent, which by default
> loads route capacity from whatever's under `data/dev/*.parquet` (the "cache"
> data source — see `integrations/route_capacity_client.py`), not the built-in
> mock routes `golden_cases.py`'s `expected_outcome`/`note` fields describe. If
> you capture with a real cache snapshot present, the captured `final_response`
> (and the recommend-vs-escalate outcome) reflects **today's real capacity**,
> which is the intended behavior for this project but will drift as real
> capacity data changes over time — a case that recommends today may escalate
> after a recapture months later, with no code change involved. That's expected,
> not a regression; re-run `eval.capture` to refresh when you want the golden
> dataset to reflect current capacity. If you instead want a stable,
> never-drifting reference (matching the mock-data design intent in
> `golden_cases.py`'s comments), run capture with
> `SMART_ASSIGNMENT_DATA_SOURCE=mock` set.

### `eval/test_quality.py` — Phase 3a: DeepEval G-Eval quality metrics

`response_match_score`/`final_response_match_v2` are similarity-to-reference
metrics — and structurally cannot score an ESCALATE case at all (see above),
leaving the highest-stakes prose (the escalation/handoff brief a human
specialist acts on) with zero automated signal. `test_quality.py` closes that
gap with two **reference-free** DeepEval G-Eval rubrics (no `expected_output`
set — the judge rates the response on its own merits, not fidelity to a
captured reference), scored **directly against captured text** — no ADK
`EvalSet`/`AgentEvaluator` involved at all, so there's no scratch dataset file
to render; this file only *reads* `eval/data/captured_responses.json`.

Both rubrics are drawn from the human-annotation dimensions in
[`deployment/phoenix/README.md`](../deployment/phoenix/README.md)'s feedback
table, so the automated score and the human-annotation vocabulary stay
aligned:

| Metric | Scored on | Rubric source |
|---|---|---|
| `brief_quality` | captures with `escalated: true` | `deployment/phoenix/README.md`'s `brief_quality` row — "is the escalation/handoff brief useful?" |
| `response_clarity` | captures with `escalated: false` | same table's `response_clarity` row — "is the final customer message clear?" |

That table's other two rows (`decision_correct` — already covered
deterministically by trajectory scoring's `recommend_or_escalate` call —
and `slot_reasonable`) and grounded-layer rationale-faithfulness (a different
granularity: the decision layer's own reasoning, not the agent's final
customer-facing prose) are **out of scope here**, deferred to a later phase.

```bash
pip install -e ".[dev,eval-quality]"
pytest eval/test_quality.py
```

The `eval-quality` extra is pinned to `deepeval==2.6.6` specifically (not a
floor) — newer DeepEval releases require `python-dotenv>=1.1.1`, which
conflicts with this project's own `litellm==1.83.7` (hard-pins
`python-dotenv==1.0.1`, for Sage SDK compatibility). See the pin's comment in
`pyproject.toml` if that litellm constraint is ever loosened — DeepEval's
public API drifted from 2.6.6 (e.g. `LLMTestCaseParams` here vs. the newer
`SingleTurnParams` alias, no built-in `GeminiModel` at 2.6.6), so re-verify
against whatever version actually resolves before bumping the pin.

Each test **skips cleanly** (not a failure) when no captured case is yet known
to have the matching outcome — same `SMART_ASSIGNMENT_EVAL_IDS` subset-cost
knob as `test_eval.py`/`capture.py` applies here too.
`SMART_ASSIGNMENT_EVAL_NUM_RUNS` does **not** apply: nothing here re-runs the
live agent, only the judge call scores already-captured text.

The judge model is `deepeval_llm.py`'s `SmartAssignmentDeepEvalLLM`, backed by
`shared/llm.py`'s own `generate_text` — the same function every other grounded
call in this repo already uses, so it works under **any**
`SMART_ASSIGNMENT_LLM_BACKEND` (including Sage-only) with no per-backend branch
needed here, unlike `test_response_match.py`'s ADK-judge path (which needs
`sage_judge_llm.py`'s registry adapter because ADK's own `LlmAsAJudgeCriterion`
resolves its model through ADK-core's generic registry — a constraint this
file doesn't have). Its model is set via the same per-role convention as every
other LLM call in this repo (`Config.for_role`) — override it independently of
the app's main model with `SMART_ASSIGNMENT_MODEL_QUALITY_JUDGE` (standard
backend) if you want a stronger/different judge than the agent's own
operational model.

> **DeepEval makes an outbound network call at import time, independent of
> telemetry opt-out — and it can't be fully suppressed from within this repo's
> own code when run via `pytest`.** [VERIFIED against installed deepeval 2.6.6
> source]: `DEEPEVAL_TELEMETRY_OPT_OUT` only covers usage-analytics events; a
> SEPARATE switch, `DEEPEVAL_UPDATE_WARNING_OPT_OUT`, gates an HTTPS GET to
> `pypi.org` (a "newer version available" check, 5s timeout, silently
> swallowed on failure). Both `deepeval_llm.py` and `test_quality.py` set both
> via `os.environ.setdefault` before their own `deepeval` imports — but
> **deepeval registers itself as a pytest plugin** (a `pytest11` entry point),
> which `pytest` auto-imports during its own startup, before ANY of this
> repo's Python code runs. That in-code `setdefault` only helps a bare, non-
> `pytest` import; the actual `pytest eval/test_quality.py` invocation is
> unaffected by it. To silence it locally, export both as real shell env vars
> **before** invoking `pytest`:
> ```bash
> export DEEPEVAL_TELEMETRY_OPT_OUT=YES DEEPEVAL_UPDATE_WARNING_OPT_OUT=YES
> pytest eval/test_quality.py
> ```
> CI's `quality-eval` job sets both in its step's `env:` block for the same
> reason. Purely cosmetic/latency, not a functional issue — the check fails
> silently (no error) if `pypi.org` is unreachable, relevant in a Sage-only
> environment where such egress may be blocked or audited.

Need one confirmed `escalated: true` capture to exercise `brief_quality` for
the first time — none exists yet as of writing (only
`woodlands_fresh_cafe_recommend` is a confirmed `escalated: false`). Recapture
a case already known to escalate under real cache data (see the callout above)
to get one cheaply:

```bash
SMART_ASSIGNMENT_EVAL_IDS=bayou_city_bistro_recommend python3 -m eval.capture
```

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
SMART_ASSIGNMENT_EVAL_IDS=woodlands_fresh_cafe_recommend \
SMART_ASSIGNMENT_EVAL_NUM_RUNS=1 \
pytest eval/test_eval.py

# Multiple cases: comma-separate the eval_id (see golden_cases.py).
SMART_ASSIGNMENT_EVAL_IDS=woodlands_fresh_cafe_recommend,galleria_grill_escalate_low_score \
pytest eval/test_eval.py
```

`SMART_ASSIGNMENT_EVAL_IDS` doesn't hand-edit the committed JSON (which can't
have comments and is checked by `tests/eval/test_build_evalset.py` for staying
in sync with `golden_cases.py`) — it renders a scratch subset from
`golden_cases.py` on the fly via the same `build_evalset` machinery that
produces the real file, so the subset can never drift from it, and nothing
under `eval/data/` is touched. Parsing/validation of the env var lives in one
shared place, `case_selection.py`, so `test_eval.py` and `capture.py` can't
drift on what a comma-separated subset means. See the docstring on
`_eval_dataset_path` in `test_eval.py` for exactly what it does there.

`capture.py` (Phase 2b) honors the same `SMART_ASSIGNMENT_EVAL_IDS` to limit
which cases get a live run — useful since it's a separate cost center from
`test_eval.py`. `SMART_ASSIGNMENT_EVAL_NUM_RUNS` does **not** apply to it:
`capture.py` already only runs each case once (there's no multi-run/consensus
step to control, unlike `AgentEvaluator`'s `num_runs`). Unlike `test_eval.py`,
a filtered (non-`--check`) capture run **merges** into any existing
`captured_responses.json` rather than replacing it, so recapturing one case
never regresses the other committed cases' `final_response` back to `null`:

```bash
# Recapture just one case's response, cheaply.
SMART_ASSIGNMENT_EVAL_IDS=woodlands_fresh_cafe_recommend python3 -m eval.capture --check
```

## CI: advisory first

The `agent-eval` job in `.github/workflows/ci.yml` runs `test_eval.py`, and the
sibling `quality-eval` job runs `test_quality.py` — both **advisory**
(`continue-on-error: true`): they report, they don't block. Both no-op cleanly
when `SAGE_*` credentials aren't configured as repo secrets, so neither fails a
PR for infrastructure reasons.

**CI currently only triggers on `main`** (see the `on:` block at the top of
`ci.yml`) — a PR targeting `dev` won't run either job yet. Widening those
triggers to include `dev` is a deliberate, separate follow-up, not bundled into
the change that added `quality-eval`.

The plan is to keep both advisory until their thresholds prove stable over a
few real PRs, then flip them to required checks.

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
