"""
Turn human feedback into candidate eval cases -- the offline back-half of the
flywheel (Production traces -> human labels -> **dataset curation** -> calibrate
evals -> tune).

This is deliberately a *curation* step, not an auto-promotion. It reads the
durable feedback log and produces candidate eval cases in a review-ready JSON
artifact; a human then inspects them and promotes the good ones into the
code-defined golden set (``eval/golden_cases.py``). That boundary is the point:
per CLAUDE.md, human feedback must feed an offline, human-driven loop -- never a
live one that silently mutates what the system does. Nothing here runs in the
request path, and nothing it writes changes a decision.

The mapping is honest about what it can and can't infer. A thumbs-up *confirms*
the observed outcome as ground truth. A thumbs-down is recorded verbatim with its
note but leaves ``suggested_expected_outcome`` **unset** -- a negative says the
decision was wrong, not *how* it was wrong (the reviewer may have wanted an
escalation, or just a different feasible route/slot), so a human sets the target
during promotion rather than the tool guessing one. The observed outcome and the
intake facts come from the decision *context* the app captured at feedback time.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from smart_assignment.feedback.schema import ANNOTATOR_HUMAN, FeedbackRecord
from smart_assignment.feedback.store import read_records

logger = logging.getLogger(__name__)

# Labels we read as a clear negative / positive signal. Anything else is carried
# through as a neutral verdict (no suggested outcome inferred).
_NEGATIVE_LABELS = frozenset({"thumbs_down", "down", "bad", "incorrect", "wrong"})
_POSITIVE_LABELS = frozenset({"thumbs_up", "up", "good", "correct"})


@dataclass(frozen=True)
class CuratedCase:
    """A review-ready candidate eval case distilled from one annotation.

    Aligns to the concepts in ``eval.golden_cases.GoldenCase`` (a proposed
    ``eval_id``, the intake ``query``/facts, an ``expected_outcome``) while
    staying explicit that it is a *candidate*: ``suggested_expected_outcome`` is
    only set when the human verdict cleanly implies one, and ``provenance`` ties
    the case back to the exact feedback and trace it came from."""

    eval_id: str
    query: Optional[str]
    observed_outcome: Optional[str]
    human_verdict: str  # "negative" | "positive" | "neutral"
    human_label: str
    human_score: Optional[float]
    note: Optional[str]
    suggested_expected_outcome: Optional[str]
    context: Dict[str, Any]
    provenance: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _verdict(record: FeedbackRecord) -> str:
    """Classify a record as a negative/positive/neutral quality signal from its
    label first, then its score (score < 0, or 0 on a 0/1 thumb encoding, reads
    negative; > 0 positive)."""
    label = (record.label or "").strip().lower()
    if label in _NEGATIVE_LABELS:
        return "negative"
    if label in _POSITIVE_LABELS:
        return "positive"
    if record.score is not None:
        if record.score <= 0:
            return "negative"
        return "positive"
    return "neutral"


def _suggested_outcome(verdict: str, observed_outcome: Optional[str]) -> Optional[str]:
    """The eval target the human verdict implies, or ``None`` when it isn't
    cleanly invertible.

    A **positive** confirms the observed outcome as ground truth, so that becomes
    the suggested target. A **negative** is deliberately left ``None`` for a human
    to decide: a thumbs-down means "this decision was wrong," but not *how* -- the
    reviewer may have wanted an escalation, or simply a *different feasible route
    or slot* than the one recommended. Guessing ``escalate`` would encode a target
    the human never actually chose, so we surface the verdict + note and let the
    reviewer set the expected outcome during promotion."""
    if verdict == "positive":
        return (observed_outcome or "").strip().lower() or None
    return None


def _query_from_context(context: Dict[str, Any]) -> Optional[str]:
    """Best-effort reconstruction of an intake-style query from captured context,
    or ``None`` when there isn't enough to bother (a human fills it in on
    promotion). Kept lossy on purpose -- this is a candidate, not a fixture."""
    parts: List[str] = []
    name = str(context.get("name") or "").strip()
    address = str(context.get("address") or "").strip()
    if name:
        parts.append(name)
    if address:
        parts.append(address)
    cases = context.get("order_quantity_cases")
    if cases:
        parts.append(f"{cases} cases")
    day = context.get("preferred_day")
    window = context.get("preferred_window")
    if day and window:
        parts.append(f"prefers {day} {window}")
    return ", ".join(parts) if parts else None


def curate_feedback(
    path: str,
    *,
    only_negative: bool = False,
) -> List[CuratedCase]:
    """Read the feedback log at ``path`` and distill HUMAN annotations into
    candidate eval cases (most-recent annotation wins per decision).

    ``only_negative`` keeps just the failure signals (the highest-value
    regression candidates). Non-human annotations (LLM/CODE) are skipped -- this
    curates *ground truth*, and those are the thing ground truth calibrates."""
    latest: "Dict[str, FeedbackRecord]" = {}
    for record in read_records(path):
        if record.annotator_kind != ANNOTATOR_HUMAN:
            continue
        key = record.target.decision_id or ""
        if not key:
            continue
        # created_at is an ISO-8601 string; lexical max == chronological max.
        prior = latest.get(key)
        if prior is None or (record.created_at or "") >= (prior.created_at or ""):
            latest[key] = record

    cases: List[CuratedCase] = []
    for decision_id, record in latest.items():
        verdict = _verdict(record)
        if only_negative and verdict != "negative":
            continue
        context = dict(record.context or {})
        observed_outcome = context.get("outcome")
        short = decision_id[:8] if decision_id else "unknown"
        cases.append(
            CuratedCase(
                eval_id=f"feedback_{short}_{verdict}",
                query=_query_from_context(context),
                observed_outcome=observed_outcome,
                human_verdict=verdict,
                human_label=record.label,
                human_score=record.score,
                note=record.note,
                suggested_expected_outcome=_suggested_outcome(verdict, observed_outcome),
                context=context,
                provenance={
                    "decision_id": decision_id,
                    "trace_id": record.target.trace_id,
                    "span_id": record.target.span_id,
                    "session_id": record.target.session_id,
                    "annotator_id": record.annotator_id,
                    "created_at": record.created_at,
                },
            )
        )
    # Stable order: negatives first (highest value), then by eval_id.
    cases.sort(key=lambda c: (c.human_verdict != "negative", c.eval_id))
    return cases


def write_curation(cases: List[CuratedCase], out_path: str) -> None:
    """Write curated cases to ``out_path`` as a JSON array for human review.
    Creates parent dirs; overwrites (curation is a regenerated snapshot)."""
    directory = os.path.dirname(os.path.abspath(out_path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(
            [c.to_dict() for c in cases],
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
