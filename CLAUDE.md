# Working in this repository

Guidance for any agent (or human) changing this codebase. Read it before adding
or modifying LLM-backed behavior. For *how* the existing layers are built, see
`docs/architecture/README.md`; this file is the *why* and the spirit.

## How to read this document

These are **principles and proven defaults, not a cage.** They exist to protect a
few real guarantees (below) — not to stop you from thinking. If you see a cleaner,
simpler, or genuinely better solution than what's described here, **take it** —
just make sure it still upholds the guarantees, and say *why* it's better in your
PR. The worst outcome is not "deviated from the recipe"; it's "silently lost a
guarantee" or "added complexity nobody needed." Favor the better idea over the
familiar one, and explain your reasoning.

## The guiding principle

**Leverage the reasoning advantage of an LLM, but keep that reasoning grounded on
evidence and constrained to a valid set of options** — so a decision comes from
reasoning over real facts, not from artificial weights that are hard to justify
and inflexible in dynamic situations.

Concretely, an LLM here **reasons and selects; it does not invent.** It picks an
option from a deterministically enumerated set and cites the facts it used. It
never free-generates a value (a route, a window, a score) that a downstream system
will act on.

## The guarantees that must hold (the *what*)

However you build a decision layer, these must survive. They're the point:

- **Never worse than the deterministic baseline.** The deterministic pipeline
  stays intact and always runs; an LLM layer is *additive* and downstream of it.
  On any failure it falls back to the deterministic result. Never make an LLM the
  only thing between input and an irreversible action.
- **No fabricated actionable values.** An LLM's actionable output is checked
  deterministically before anything acts on it; a value a user sees is either
  verified against real facts or visibly caveated.
- **Auditability & observability.** A human can reconstruct *why* a decision was
  made. Every fallback logs its reason.
- **Opt-in, no regressions.** New LLM behavior is gated by a `Config.use_*` flag,
  default off; with the flag off, prior behavior is reproduced exactly.
- **A layer changes only what it owns.** `slotpick` re-orders the slot for the
  chosen route but never touches the route, score, or decision; `triage` composes
  a handoff brief but never changes the decision. Keep that discipline.

## The proven recipe (the *how* — a strong default)

Every LLM decision layer in this repo follows the same shape, and it's the
recommended starting point. Three reference implementations: `judgment/`
(recommend vs. escalate), `triage/` (escalation brief), `slotpick/` (final
delivery-slot pick). Match it unless you have a better way to meet the guarantees:

1. **Enumerate the valid options deterministically** (see any `evidence.py`); the
   LLM chooses *from that set*, by index/id.
2. **Build an evidence packet** of the raw facts the decision needs — numbers/labels
   the model can cite, not prose.
3. **Return a structured choice with citations** (see any `schema.py`): the chosen
   option plus `{index, field, value}` citations to packet facts.
4. **Verify deterministically, no LLM** (see any `verifier.py`): the choice is in
   the set and every cited value matches the packet within tolerance. Retry once
   with the failure as feedback.
5. **Fall back to the deterministic result** on *any* failure (parse, failed
   verification, backend/credentials error), logging why.
6. **Gate behind a flag, default off.**

### Demote deterministic heuristics — don't remove them

When a deterministic scorer/blend already exists (e.g. the slot `SLOT_WEIGHT_*`
blend), **keep it, but demote it:** it stays the **fallback**, and its verdict (its
score, and the option it would pick on its own) goes **into the evidence packet as
reference** — a *strong default the model may agree with, or diverge from with
justification in its rationale.* The goal is to move hand-tuned weights from being
the sole, opaque decider to being one grounded input among several, while the
auditable heuristic remains the floor. A **demotion, not a deletion** — and don't
let the weights silently override the LLM either.

### Match rigor to stakes; stay lightweight

Spend verification effort in proportion to the cost of being wrong. High-stakes
(auto-assign vs. escalate a customer): resample `k` times and require consensus
before overriding the conservative path. Low-stakes (which of ~3 feasible slots):
a single grounded call + verify + fallback, no resampling. Don't over-engineer a
low-stakes layer; don't under-verify a high-stakes one.

## Core engineering principles

Independent of the LLM specifics, aim for code that stays easy to change:

- **Clean, robust, well-documented.** Readable names, small functions, clear
  error handling and fallbacks. Document the *why*, not the obvious *what*; keep
  `docs/architecture/README.md` in step with the code.
- **Modular, low coupling, high cohesion.** Each package/module owns one concern
  (evidence / schema / verify / prompt / fallback are deliberately separate);
  depend on small interfaces, not concrete internals.
- **Easy to plug in and extend.** New decision layers should slot in behind a flag
  and a config knob without touching unrelated call sites — as `judgment/`,
  `triage/`, and `slotpick/` do. Prefer adding a strategy over branching an
  existing one.
- **Favor simplicity; avoid over-engineering.** Build for the requirement in front
  of you, not an imagined one. The simplest design that meets the guarantees wins.
- **Keep imports credential-free.** Construct agents/backends lazily so importing a
  package never requires credentials (see the lazy construction in `triage/agent.py`
  and the PEP 562 module `__getattr__`s).
- **Model choice is per-role.** Resolve the model via `Config.for_role(<role>)`; add
  a role constant + `SMART_ASSIGNMENT_MODEL_<ROLE>` env override rather than
  hardcoding a model at a call site.

## A sanity check before you commit

Not a gate on thinking — just a checklist so a guarantee doesn't slip:

- [ ] Flag-gated and off by default; flag-off reproduces prior output exactly.
- [ ] The LLM selects from an enumerated set and cites facts; it doesn't
      free-generate an actionable value (or, if you diverged, the guarantees still
      hold and the PR explains how).
- [ ] Something deterministic rejects out-of-set choices and fabricated numbers.
- [ ] There's a deterministic fallback on every failure path, and it logs why.
- [ ] Any pre-existing heuristic is demoted (fallback + reference), not deleted.
- [ ] `python3 -m pytest -q` and `python3 -m flake8` are clean.
- [ ] `docs/architecture/README.md` reflects the change.
