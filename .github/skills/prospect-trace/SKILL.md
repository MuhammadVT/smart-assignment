---
name: prospect-trace
description: "Capture a live end-to-end execution trace of the Smart Assignment root_agent for one prospect and publish it as a visual walkthrough page: every deterministic calculation, every LLM prompt and reply, and which branches fired. Use when someone wants to see, explain, or review how the agent reached a decision for a given address/order. Triggers: 'trace this prospect', 'show me what the agent does with', 'visualize the decision for', 'end-to-end walkthrough', 'what goes into the LLM', 'explain this recommendation', 'why did it pick that route'."
argument-hint: "Prospect details: address, order size in cases, and optionally a preferred day + time window (e.g. '1201 Lake Woodlands Dr, The Woodlands, TX 77380, 150 cases, THU 09:00-12:00')"
---

# Prospect trace → visual walkthrough

Runs the **real** agent against a given prospect, captures everything that
happened, and turns it into a page that explains the decision end to end.

The point is auditability made legible: a reader should finish the page knowing
exactly which numbers were computed by Python, which were reasoned by a model,
what text each model call received, what it returned, and what checked it
afterwards.

## When to use

- Someone asks how the agent reached a particular recommendation or escalation.
- You need to review or explain the decision pipeline for a specific address.
- You are demonstrating the architecture to a product owner or an ops reviewer.
- You changed the pipeline and want a before/after on a real prospect.

Do **not** hand-write any figure in the page. Every number comes from the capture.

## Prerequisites

Working credentials for the configured backend (`SMART_ASSIGNMENT_LLM_BACKEND`
in `.env`) — the capture drives real model calls. With none, the pipeline still
runs but the grounded layers fall back deterministically, and the page should say
so rather than pretend a model was consulted.

## Procedure

### 1. Capture the trace

```bash
python .github/skills/prospect-trace/capture_trace.py \
  --address "1201 Lake Woodlands Dr, The Woodlands, TX 77380" \
  --cases 150 --day THU --window 09:00-12:00 \
  --out .trace/<slug> --quiet
```

`--day` and `--window` are optional but must be given together. `--name`,
`--customer-number` and `--message` (to override the exact sentence sent to the
agent) are also available. The script is read-only with respect to the repo and
writes two files:

- `.trace/<slug>/trace.json` — everything, ~50–140 KB
- `.trace/<slug>/digest.md` — a compact orientation pass

If it exits non-zero the agent turn failed; read `agent_error` in the trace and
report that rather than publishing a page built on a broken run.

### 2. Read the digest first, then pull details

Read `digest.md` in full — it gives you the shape of the run: which flags were
on, the candidate ranking, the constraint verdicts, every scored (route, slot)
with its arithmetic, the gate, the nested LLM calls, the verifier verdict, the
tool-call sequence, and the final reply.

Then go into `trace.json` for the specifics the page needs. Useful keys:

| Key | What it holds |
|---|---|
| `config`, `resolved_models` | every knob the decision used; the model per role |
| `geo.ranking` | the full proximity ranking, each row tagged `top_n` / `preferred_day_keep` / `cut` |
| `geo.preferred_day_keep` | the extra candidate, its global rank and the distance limit it cleared |
| `routes[].neighbors`, `.clusters`, `.menu` | slot construction: weights, anchors, windows, contention, `openness_calc` |
| `route_slot_totals[].calc` | the weighted total written out as a substitution |
| `packet`, `judgment_prompt` | exactly what the grounded call received |
| `nested_llm_calls[]` | prompt, reply, whether it answered via tool call or text, latency |
| `verification` | the deterministic verdict on the model's reply |
| `agent_llm_traffic`, `triage_llm_traffic` | every request/response on the conversation loop and in the triage sub-agent |
| `events`, `session_state`, `final_reply` | tool results, state written, the reply the user sees |

### 3. Author the page

Write a **body-only HTML fragment** (no `<!doctype>`, `<html>`, `<head>` or
`<body>` tags) to `.trace/<slug>/page.fragment.html`, with
`.github/skills/prospect-trace/artifact.css` inlined verbatim in a `<style>`
block and a `<title>`.

**Structure** — follow this spine; drop sections that did not happen and add ones
that did:

1. **Masthead** — the input sentence verbatim, a config fingerprint (backend,
   model, geocoder, data source, the `use_*` flags, the bar, `top_n`), and a
   verdict strip (outcome, route, slot, score vs bar, whether human review was
   required, wall clock).
2. **Flow strip** — one node per stage, each tagged with its lane, so the
   sequence of model turns and code steps is visible before the detail.
3. **Per stage, in execution order** — a card per step, `data-lane="code"` or
   `data-lane="model"`:
   - *model* stages get an IN pane (system instruction excerpt, declared tools,
     conversation so far) and an OUT pane (the reply verbatim);
   - *code* stages get `.calc` rows: the formula, the substitution with real
     numbers, and the result.
4. **The grounded judgment call** — its prompt broken into its three parts
   (framing, evidence packet, output contract), the model's reply **verbatim and
   complete**, then the verifier's checks one by one with the real verdict.
5. **Branches that did not fire** — a table naming each dormant path and why
   (address resolution, triage, `request_input`, each escalation reason, the
   k-sample consensus, the grounded fallback). This is as informative as what did
   run.
6. **Ledger** — per model call: prompt tokens, output tokens, latency; plus the
   deterministic time. Note whether the timings were warm.

**Rules:**

- Every figure traces to the capture. If you want to state something the trace
  does not contain, either compute it from trace values and say you did, or
  leave it out.
- Reproduce model output verbatim, including any awkward phrasing. It is
  evidence, not copy.
- Lead each stage with what it *decides*, not what it is called.
- Call out what is genuinely surprising in this run — a candidate eliminated
  before scoring, a factor that saturated, a tool channel that silently degraded,
  a verifier tolerance that mattered. A page that only narrates the happy path
  wastes the capture.
- Wide content (tables, `<pre>`) must scroll inside its own container; the body
  must never scroll sideways.

### 4. Publish

**Claude Code** — publish `page.fragment.html` with the Artifact tool (set a
`favicon`, a one-line `description`, and a stable `<title>`). To update an
existing page, pass its URL so the link is preserved rather than minting a new
one.

**GitHub Copilot Chat, or any harness without artifact publishing** — wrap the
fragment into a standalone page and tell the user the path:

```bash
python .github/skills/prospect-trace/capture_trace.py \
  --wrap-html .trace/<slug>/page.fragment.html \
  --out-html  .trace/<slug>/page.html
```

The page is fully self-contained (inlined CSS, no external requests), so it opens
straight from disk.

### 5. Report

Summarise in chat: the outcome, the decisive factor, and anything the run
revealed that the reader would not have predicted. Link or name the page.

## Notes

- `.trace/` is git-ignored. Captures contain a real customer address — treat them
  as you would any other PII and do not commit them.
- The judgment call is sampled, so its prose differs run to run; the chosen index
  and the verified figures do not. Say so if you publish two captures of the same
  prospect.
- Route labels are written `<route id> - <route name>` everywhere, matching the
  convention the pipeline and prompts enforce.
