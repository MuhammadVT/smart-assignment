"""
Grounded LLM judgment for the recommend-or-escalate decision (opt-in).

This package is an **alternative** to the default weighted-sum + fixed-threshold
decision made in `pipeline.decide`. Instead of collapsing the soft factors into
one number and gating on `Config.total_score_threshold`, it:

  1. builds a structured *evidence packet* of the raw per-candidate facts
     (`evidence.py`),
  2. asks an LLM to reason freely over the trade-offs and return a
     schema-constrained judgment (`schema.py`, `prompts.py`),
  3. deterministically *verifies* that every claim in that judgment is grounded
     in the packet and that the picked route is actually feasible
     (`verifier.py`), and
  4. spends extra independent samples only on "escalation-side" cases and
     combines them by a configurable consensus rule (`judge.py`).

What it does NOT change: hard constraints (`shared/constraints.py`) still run
first in the pipeline and remain the only thing that can eliminate a candidate.
The LLM only ever chooses among the already-feasible set (enforced twice: the
output schema and the verifier), so it can never place a customer on an
over-capacity or out-of-area route. On any mechanical failure (malformed
output, ungrounded claim that survives one corrective retry) it falls back to
the existing deterministic weighted pick + `DeterministicReasoner` text, so the
worst case is never worse than today's behavior.
"""

from __future__ import annotations

from smart_assignment.judgment.judge import (
    GroundedJudge,
    Judge,
    WeightedScoreJudge,
    default_judge,
)

__all__ = ["GroundedJudge", "Judge", "WeightedScoreJudge", "default_judge"]
