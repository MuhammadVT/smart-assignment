# Human feedback on eval runs — self-hosted Arize Phoenix (no containers)

This is a **second option** for the same loop as `deployment/langfuse/`: stand up
a local, self-hosted trace backend, point the agent's eval runs at it, and have
a human **score the resulting traces**. Use this one if Podman (or any
container runtime) is unavailable or locked down on your machine — Phoenix runs
as a plain Python process, no container runtime required.

Nothing here changes the product. It reuses the same opt-in tracing seam
(`smart_assignment/shared/tracing.py`), which is already vendor-neutral: it
exports over standard OTLP/HTTP, and Phoenix ingests OTLP directly — so this
directory only swaps the **backend** (Phoenix instead of Langfuse); the
**workflow** (run eval, review traces, attach human scores) is the same shape.

> **Why this is additive & safe.** Tracing is gated by `Config.use_tracing`
> (`SMART_ASSIGNMENT_USE_TRACING`, default **off**) and degrades to a silent
> no-op on any failure — SDK missing, backend unreachable, bad endpoint. A
> broken or absent Phoenix instance can never change a decision or fail an eval
> for infrastructure reasons. See `docs/architecture/README.md` → *Tracing &
> observability*.

---

## The loop at a glance

```
 eval/test_eval.py (real agent, real backend)
        │  builds root_agent → configure_tracing() installs the OTLP exporter
        ▼
 OpenTelemetry spans  ──►  Phoenix (self-hosted, plain process)  ──►  human reviewer
   agent turns              traces / spans / annotations              opens a trace,
   tool calls                                                         clicks Annotate,
   grounded decisions                                                 scores it
        ▲                                                                  │
        └──────────────────  feedback informs the next eval / thresholds ◄─┘
```

- **Automated eval** (`eval/`) scores *trajectory* deterministically (did the
  agent drive `intake → find → evaluate → recommend/escalate` correctly).
- **Human feedback** (Phoenix) scores what automation can't judge offline —
  *was the escalation brief good? was the chosen slot reasonable? was the final
  message clear?* — as **annotations attached to the very traces those eval
  runs produced.**

---

## Langfuse vs. Phoenix — which to use

| | `deployment/langfuse/` | `deployment/phoenix/` (this one) |
|---|---|---|
| Runtime | Podman/Docker, 6 containers (Postgres, ClickHouse, Redis, MinIO, web, worker) | `pip install`, one local process |
| Setup step | Create org/project, generate API keys | None — local instance needs no signup |
| Human review UX | Structured **Annotation Queues** (score configs, batch assignment, multiple reviewers) | Per-trace **Annotate** button + SDK; queue-like review is filter-and-step-through |
| Best for | A team's shared review workstation | A single reviewer, or environments where containers aren't available |

Both plug into the same `OTEL_EXPORTER_OTLP_ENDPOINT` seam — you can point the
same eval run at either, or swap later, with zero code changes.

---

## Prerequisites

- Python 3.9+ (same interpreter the repo already uses).
- The repo installed with the observability extra so the exporter is present:
  ```bash
  pip install -e ".[observability]"
  ```
- Phoenix itself, in a **separate** virtual environment (it is an external
  observability backend, not a dependency of `smart_assignment` — same
  discipline as Langfuse being an external service rather than a library):
  ```bash
  python3 -m venv ~/.venvs/phoenix
  source ~/.venvs/phoenix/bin/activate
  pip install arize-phoenix
  deactivate
  ```

---

## 1. Start Phoenix

```bash
cd deployment/phoenix
./phoenix.sh up        # starts `phoenix serve` in the background
./phoenix.sh status    # check it's running + show the data dir
./phoenix.sh logs      # follow server logs
./phoenix.sh down      # stop; add --purge to also delete local trace data
```

Prefer to do it by hand? That's all the script does:

```bash
source ~/.venvs/phoenix/bin/activate
PHOENIX_WORKING_DIR=./deployment/phoenix/.data phoenix serve
```

When it's up, the **UI is at http://localhost:6006**. No sign-up, org, or API
key needed for a local instance — that whole step from the Langfuse guide is
skipped here.

> **Persistence.** Phoenix stores traces in a local SQLite file under
> `PHOENIX_WORKING_DIR` (the script points this at `deployment/phoenix/.data/`
> so it survives restarts and is easy to find/clean up). For a shared,
> multi-reviewer instance instead of a single laptop, Phoenix also supports
> Postgres via `PHOENIX_SQL_DATABASE_URL` — see the self-hosting docs linked
> below if you outgrow the local file.

---

## 2. Point the agent's eval at Phoenix

Add these to the repo-root `.env` (see `.env.example` → *Tracing*). No code
change is needed — this is the same, vendor-neutral OTLP path the tracing
module already prefers:

```bash
SMART_ASSIGNMENT_USE_TRACING=true
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:6006/v1/traces

# Optional: the name traces appear under in Phoenix's project picker (defaults
# to smart-assignment). Use a distinct value to make eval-run traces easy to find:
OTEL_SERVICE_NAME=smart-assignment-eval
```

Do **not** also set the `LANGFUSE_*` trio — `OTEL_EXPORTER_OTLP_ENDPOINT` takes
precedence over it in `shared/tracing.py`, but leaving both around invites
confusion about which backend is actually receiving spans.

---

## 3. Run eval so traces land in Phoenix

The eval harness replays the golden intake conversations against the **real**
agent, which builds `root_agent` → `configure_tracing()` → installs the
exporter. So a normal eval run emits a full trace tree per case:

```bash
python3 -m eval.build_evalset      # regenerate the dataset if cases changed
pytest eval/test_eval.py           # needs a live LLM backend (see eval/README.md)
```

Refresh the Phoenix UI and open the `smart-assignment-eval` project — one
trace per replayed case, each with the agent's turns, the pipeline tool calls,
and the grounded decision spans nested underneath. (Spans are deliberately
PII-free: backend/model/role/sizes/latency, **not** prompt or response text —
see the tracing module.)

---

## 4. Add human feedback (the point)

Phoenix stores human judgments as **annotations** on spans/traces, tagged
`annotator_kind=HUMAN` (as opposed to `LLM` or `CODE` annotators), so they're
distinguishable from any automated scoring you add later.

### a) Ad-hoc annotation on a single trace
Open a trace → **Annotate** (top right) → fill in a label/score and an
optional explanation → save. Good for spot-checking one surprising escalation.
You can edit or delete your annotations later.

### b) Working through a batch of eval-run traces
Phoenix doesn't have a dedicated queue object like Langfuse's Annotation
Queues; the equivalent workflow is filter-then-step-through:

1. In the Traces view, filter to the project/service name you set
   (`smart-assignment-eval`) and to the time range of your eval run.
2. Open each trace and annotate it. Useful dimensions for this agent (create
   these as annotation names the first time you use them — Phoenix doesn't
   require pre-declaring a schema the way Langfuse's Score Configs do):
   | Name | Suggested shape | What it's judging |
   |---|---|---|
   | `decision_correct` | label: correct / incorrect | was recommend-vs-escalate right? |
   | `slot_reasonable` | label: good / acceptable / wrong | was the chosen delivery slot reasonable? |
   | `brief_quality` | score 1–5 | is the escalation/handoff brief useful? |
   | `response_clarity` | score 1–5 | is the final customer message clear? |
3. To split the batch across teammates, share the Phoenix URL (it's a single
   local instance, so this only works if reviewers can reach that host/port —
   e.g. over a shared network or via port-forwarding, not out of the box the
   way a deployed Langfuse instance would be).
4. Optionally, send the annotated traces to a **Dataset** so the human-scored
   set is easy to re-pull later (for comparing against trajectory scores, or
   as a seed for a future LLM-as-judge).

The annotations are now attached to the same traces the automated eval
produced — **human feedback layered on top of the deterministic eval, not
replacing it.**

---

## Teardown

```bash
cd deployment/phoenix
./phoenix.sh down            # stop the process, keep .data/ (trace history)
./phoenix.sh down --purge    # stop and delete .data/ (all local trace history)
```

---

## Caveats

- This is a **single local process** for one reviewer or a small team on a
  shared network — no auth, no HA, no backups by default. Fine for
  human-in-the-loop review; follow Phoenix's self-hosting docs (Postgres
  backend, Docker/Kubernetes deployment) before any shared production use.
- The helper script and instructions here have **not** been executed against a
  live host as part of building this repo (same convention as
  `deployment/deploy.py`) — verify the `phoenix serve` invocation and default
  ports against the `arize-phoenix` version you install. The exporter path
  they drive, however, is exercised by `tests/shared/test_tracing.py`.

### Sources
- Phoenix self-hosting: <https://arize.com/docs/phoenix/self-hosting>
- Phoenix persistence (SQLite/Postgres, `PHOENIX_WORKING_DIR`): <https://docs.arize.com/phoenix/deployment/persistence>
- Capturing human feedback via annotations: <https://docs.arize.com/phoenix/tracing/how-to-tracing/feedback-and-annotations/capture-feedback>
- Annotating in the UI: <https://docs.arize.com/phoenix/tracing/how-to-tracing/feedback-and-annotations/annotating-in-the-ui>
