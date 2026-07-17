# Human feedback on eval runs — self-hosted Langfuse via Podman

This guide shows how to stand up a **self-hosted Langfuse** instance with
**Podman**, point the agent's eval runs at it, and then have a human **score the
resulting traces** — closing an *automated eval → human feedback* loop.

Nothing here changes the product. It reuses the repo's existing, opt-in tracing
seam (`smart_assignment/shared/tracing.py`): with the flag on and the
`LANGFUSE_*` env set, every agent turn, tool call, and grounded decision is
already exported over OpenTelemetry to Langfuse. This directory only adds the
**backend** (Podman) and the **workflow** (annotation) around that seam.

> **Why this is additive & safe.** Tracing is gated by `Config.use_tracing`
> (`SMART_ASSIGNMENT_USE_TRACING`, default **off**) and degrades to a silent
> no-op on any failure — SDK missing, backend unreachable, bad credentials. A
> broken or absent Langfuse instance can never change a decision or fail an eval
> for infrastructure reasons. See `docs/architecture/README.md` → *Tracing &
> observability*.

---

## The loop at a glance

```
 eval/test_eval.py (real agent, real backend)
        │  builds root_agent → configure_tracing() installs the OTLP exporter
        ▼
 OpenTelemetry spans  ──►  Langfuse (self-hosted, Podman)  ──►  human reviewer
   agent turns              traces / observations / scores        opens an
   tool calls                                                     Annotation Queue,
   grounded decisions                                             scores each trace
        ▲                                                              │
        └──────────────  feedback informs the next eval / thresholds ◄─┘
```

- **Automated eval** (`eval/`) scores *trajectory* deterministically (did the
  agent drive `intake → find → evaluate → recommend/escalate` correctly).
- **Human feedback** (Langfuse) scores what automation can't judge offline —
  *was the escalation brief good? was the chosen slot reasonable? was the final
  message clear?* — as **scores attached to the very traces those eval runs
  produced.**

---

## Prerequisites

- **Podman 4.1+** with Compose support. Either works:
  - `podman compose ...` (built-in, Podman ≥ 4.1), or
  - `podman-compose ...` (the standalone Python tool: `pip install podman-compose`).
- The Podman machine/socket running (`podman machine start` on macOS/Windows;
  the rootless service on Linux is fine).
- ~4 GB free RAM and a few GB of disk — the Langfuse v3 stack runs Postgres,
  ClickHouse, Redis, and MinIO alongside the web + worker containers.
- The repo installed with the observability extra so the exporter is present:
  ```bash
  pip install -e ".[observability]"
  ```

---

## 1. Bring up Langfuse with Podman

Langfuse publishes a canonical multi-service `docker-compose.yml`. The helper
script here fetches that upstream file and runs it **through Podman** (it never
hand-maintains a fork of their stack, which would silently drift):

```bash
cd deployment/langfuse
./podman-langfuse.sh up        # downloads the upstream compose (once) and starts it
./podman-langfuse.sh status    # show container state
./podman-langfuse.sh logs      # follow web + worker logs
./podman-langfuse.sh down      # stop; add --volumes to also wipe data
```

Prefer to do it by hand? That's all the script does:

```bash
curl -fsSL https://raw.githubusercontent.com/langfuse/langfuse/main/docker-compose.yml \
    -o docker-compose.yml
podman compose up -d          # or: podman-compose up -d
```

When it's healthy, the **UI is at http://localhost:3000**.

> **Dev secrets are dev-only.** The upstream compose ships placeholder secrets
> (`NEXTAUTH_SECRET=mysecret`, `SALT=mysalt`, a zeroed `ENCRYPTION_KEY`, default
> Postgres/ClickHouse/Redis/MinIO passwords) and binds the datastores to
> `127.0.0.1`. That is fine for a local feedback workstation. **Do not expose
> this to a network or treat it as production** without rotating every secret —
> see Langfuse's self-hosting docs.

### Rootless Podman notes

- Ports `3000` (web) and `9090` (MinIO API) are published to the host; the rest
  bind to loopback. All are > 1024, so rootless needs no extra privilege.
- If ClickHouse or MinIO won't start, it's almost always memory pressure — give
  the Podman machine more RAM (`podman machine stop && podman machine set
  --memory 6144 && podman machine start`).
- SELinux hosts: the upstream named volumes work as-is; you only need `:Z` if you
  switch to bind mounts.

---

## 2. Create a project and get keys

1. Open http://localhost:3000 and **sign up** (the first local account is the
   instance admin — no email server needed).
2. Create an **Organization** → **Project**.
3. **Project → Settings → API Keys → Create** to get a **Public key**
   (`pk-lf-…`) and **Secret key** (`sk-lf-…`).

---

## 3. Point the agent's eval at Langfuse

Add these to the repo-root `.env` (see `.env.example` → *Tracing*). No code
change is needed — the exporter derives its OTLP endpoint and Basic-auth header
from the `LANGFUSE_*` trio:

```bash
SMART_ASSIGNMENT_USE_TRACING=true
LANGFUSE_HOST=http://localhost:3000
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...

# Optional: the name traces appear under in Langfuse (defaults to smart-assignment).
# Use a distinct value to make eval-run traces easy to find, e.g.:
OTEL_SERVICE_NAME=smart-assignment-eval
```

Under the hood the exporter POSTs to
`${LANGFUSE_HOST}/api/public/otel/v1/traces` — standard OTLP, no
Langfuse-specific client. If you'd rather stay fully vendor-neutral, set
`OTEL_EXPORTER_OTLP_ENDPOINT`/`OTEL_EXPORTER_OTLP_HEADERS` instead; those take
precedence.

---

## 4. Run eval so traces land in Langfuse

The eval harness replays the golden intake conversations against the **real**
agent, which builds `root_agent` → `configure_tracing()` → installs the
exporter. So a normal eval run emits a full trace tree per case:

```bash
python3 -m eval.build_evalset      # regenerate the dataset if cases changed
pytest eval/test_eval.py           # needs a live LLM backend (see eval/README.md)
```

Refresh the Langfuse **Traces** view — one trace per replayed case, each with
the agent's turns, the pipeline tool calls, and the grounded decision spans
nested underneath. (Spans are deliberately PII-free: backend/model/role/sizes/
latency, **not** prompt or response text — see the tracing module.)

---

## 5. Add human feedback (the point)

Langfuse stores human judgments as **scores** on traces/observations. Two ways,
smallest first:

### a) Ad-hoc score on a single trace
Open a trace → **Annotate** → pick or create a score dimension → save. Good for
spot-checking one surprising escalation.

### b) Annotation Queue (for reviewing a batch of eval runs)
This is the workflow that scales across an eval run:

1. **Create a Score Config** (*Settings → Scores*, or inline when building the
   queue) — one per dimension you want humans to judge. Useful starters for this
   agent:
   | Dimension | Type | Values |
   |---|---|---|
   | `decision_correct` | BOOLEAN | true / false — was recommend-vs-escalate right? |
   | `slot_reasonable` | CATEGORICAL | good / acceptable / wrong |
   | `brief_quality` | NUMERIC | 1–5 — is the escalation/handoff brief useful? |
   | `response_clarity` | NUMERIC | 1–5 — is the final customer message clear? |
2. **Create an Annotation Queue** (*Evaluation → Annotation Queues*), attach
   those score configs, and **add the eval traces** to it (filter Traces by the
   `OTEL_SERVICE_NAME` you set, select them, *Add to queue*).
3. **Review**: open the queue and step through each trace, assigning scores.
   Invite teammates to split the batch — Langfuse supports multiple annotators
   per queue.

The scores are now attached to the same traces the automated eval produced. You
can filter and export them (UI or the public API) to compare human judgment
against the trajectory scores, track quality across changes, or curate a
ground-truth set for a future LLM-as-judge — **human feedback layered on top of
the deterministic eval, not replacing it.**

---

## Teardown

```bash
cd deployment/langfuse
./podman-langfuse.sh down            # stop containers, keep data
./podman-langfuse.sh down --volumes  # stop and delete all Langfuse data
```

---

## Caveats

- This stack is for **local human-in-the-loop review and testing**. It has no
  HA, scaling, or backups; secrets are placeholders. Follow Langfuse's
  self-hosting guide before any shared or production use.
- The helper script and instructions here have **not** been executed against a
  live Podman host as part of building this repo (same convention as
  `deployment/deploy.py`) — verify image pulls and resource limits in your
  environment. The exporter path they drive, however, is exercised by
  `tests/shared/test_tracing.py`.

### Sources
- Langfuse self-hosting (Docker Compose): <https://langfuse.com/self-hosting/docker-compose>
- Annotation Queues: <https://langfuse.com/docs/evaluation/evaluation-methods/annotation-queues>
- Human annotation / scores via UI: <https://langfuse.com/docs/evaluation/evaluation-methods/annotation>
