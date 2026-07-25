"""
Phase 0 / Tier 3 -- read *explicit per-dimension* human annotations from any
backend and normalize them to the one ``HumanLabel`` shape ``judge_calibration``
consumes. This is the sharp calibration signal (a human directly rating
``response_clarity`` good/bad), as opposed to the holistic thumb (Tier 1.5) or a
note tag (Tier 2).

Three sources, one shape -- vendor-free first, with Phoenix and Langfuse behind
the same normalizer:

* **vendor-free** (default) -- the durable JSONL feedback log. A dimensional
  annotation is just a ``FeedbackRecord`` whose ``label`` is ``"<dimension>:<verdict>"``
  (e.g. ``"response_clarity:bad"``, ``"brief_quality:4"``), reusing the existing
  schema with no change. Holistic thumbs on the same decision are merged in.
* **phoenix** (today) -- Phoenix stores human judgments as *annotations* on the
  decision trace (annotation name = dimension, value = a label or a 1-5 score;
  see ``deployment/phoenix/README.md``). Read via the Phoenix client, correlated
  by trace id.
* **langfuse** (later) -- the same idea via Langfuse *scores*; adapter provided,
  read lazily.

Only the ``_rows_to_labels`` / label-parsing core is pure and unit-tested; the
live client calls are lazily imported and defensive, so importing this module
needs neither backend nor credentials.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

from eval.judge_calibration import DIMENSIONS, HumanLabel

logger = logging.getLogger(__name__)

_POSITIVE_WORDS = {
    "good", "correct", "acceptable", "ok", "okay", "pass", "passed", "yes",
    "clear", "useful", "reasonable", "up", "thumbs_up",
}
_NEGATIVE_WORDS = {
    "bad", "incorrect", "wrong", "fail", "failed", "no", "unclear", "confusing",
    "poor", "down", "thumbs_down",
}
_THUMB = {"thumbs_up": "up", "thumbs_down": "down", "up": "up", "down": "down"}


def verdict_to_bool(value: Any) -> Optional[bool]:
    """Normalize a dimension verdict -- a word (good/bad, correct/incorrect) or a
    score (1-5, or a 0-1 fraction) -- to a boolean, or ``None`` when unrecognized.
    Scores: >= 4 on a 1-5 scale, or >= 0.6 on a 0-1 scale, count as positive."""
    text = str(value).strip().lower()
    if not text:
        return None
    try:
        num = float(text)
    except ValueError:
        if text in _POSITIVE_WORDS:
            return True
        if text in _NEGATIVE_WORDS:
            return False
        return None
    if num <= 1.0:
        return num >= 0.6
    return num >= 4.0


def parse_dimension_label(label: str) -> Optional[Tuple[str, bool]]:
    """Parse a vendor-free ``"<dimension>:<verdict>"`` label into
    ``(dimension, positive)``, or ``None`` if it isn't a known dimension verdict."""
    if not label or ":" not in label:
        return None
    name, _, value = label.partition(":")
    dim = name.strip().lower()
    if dim not in DIMENSIONS:
        return None
    positive = verdict_to_bool(value)
    return (dim, positive) if positive is not None else None


def vendorfree_labels(path: str) -> List[HumanLabel]:
    """Read HUMAN feedback records into merged ``HumanLabel``s -- both dimensional
    (``dim:verdict`` labels) and holistic thumbs, one label per decision."""
    from smart_assignment.feedback.store import read_records

    merged: Dict[str, Dict[str, Any]] = {}
    for record in read_records(path):
        if record.annotator_kind != "HUMAN":
            continue
        key = record.target.decision_id
        if not key:
            continue
        entry = merged.setdefault(
            key, {"outcome": None, "thumb": None, "note": None, "dimensions": {}}
        )
        context = record.context or {}
        if context.get("outcome") and not entry["outcome"]:
            entry["outcome"] = context.get("outcome")
        if record.note and not entry["note"]:
            entry["note"] = record.note

        dimensional = parse_dimension_label(record.label or "")
        if dimensional is not None:
            entry["dimensions"][dimensional[0]] = dimensional[1]
            continue
        thumb = _THUMB.get((record.label or "").strip().lower())
        if thumb and not entry["thumb"]:
            entry["thumb"] = thumb

    return [
        HumanLabel(
            decision_id=key,
            outcome=entry["outcome"],
            thumb=entry["thumb"],
            note=entry["note"],
            dimensions=entry["dimensions"],
        )
        for key, entry in merged.items()
    ]


def _rows_to_labels(rows: Iterable[Dict[str, Any]]) -> List[HumanLabel]:
    """Normalize backend annotation rows (Phoenix/Langfuse) into merged
    ``HumanLabel``s. Each row: ``{key, dimension, value}`` (an explicit dimension
    annotation) and/or ``{key, thumb, outcome, note}``. ``key`` is the correlation
    id (a trace id) shared with the judge verdicts."""
    merged: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        key = row.get("key")
        if not key:
            continue
        entry = merged.setdefault(
            key, {"outcome": None, "thumb": None, "note": None, "dimensions": {}}
        )
        if row.get("outcome") and not entry["outcome"]:
            entry["outcome"] = row["outcome"]
        if row.get("note") and not entry["note"]:
            entry["note"] = row["note"]
        thumb = _THUMB.get(str(row.get("thumb") or "").strip().lower())
        if thumb and not entry["thumb"]:
            entry["thumb"] = thumb
        dim = str(row.get("dimension") or "").strip().lower()
        if dim in DIMENSIONS and row.get("value") is not None:
            positive = verdict_to_bool(row["value"])
            if positive is not None:
                entry["dimensions"][dim] = positive
    return [
        HumanLabel(
            decision_id=key,
            outcome=entry["outcome"],
            thumb=entry["thumb"],
            note=entry["note"],
            dimensions=entry["dimensions"],
        )
        for key, entry in merged.items()
    ]


def phoenix_labels(
    project: Optional[str] = None, endpoint: Optional[str] = None
) -> List[HumanLabel]:
    """Read dimension annotations from Phoenix (today), correlated by trace id.
    Lazily imports the Phoenix client; env supplies defaults (``OTEL_SERVICE_NAME``,
    ``PHOENIX_COLLECTOR_ENDPOINT`` / ``OTEL_EXPORTER_OTLP_ENDPOINT``)."""
    import phoenix as px  # lazy

    project = project or os.environ.get("OTEL_SERVICE_NAME", "smart-assignment")
    endpoint = endpoint or os.environ.get("PHOENIX_COLLECTOR_ENDPOINT") or os.environ.get(
        "OTEL_EXPORTER_OTLP_ENDPOINT"
    )
    client = px.Client(endpoint=endpoint) if endpoint else px.Client()

    rows: List[Dict[str, Any]] = []
    try:
        annotations = client.get_span_annotations_dataframe(project_name=project)
    except Exception:  # noqa: BLE001 - API name/shape varies by Phoenix version
        logger.warning(
            "Could not read Phoenix span annotations (client/API version?). "
            "See phoenix_curate.py for the version-current query pattern."
        )
        return []
    for _, row in annotations.reset_index().iterrows():
        rows.append(
            {
                "key": row.get("context.trace_id") or row.get("trace_id"),
                "dimension": row.get("annotation_name") or row.get("name"),
                "value": row.get("result.label") or row.get("result.score")
                or row.get("label") or row.get("score"),
            }
        )
    return _rows_to_labels(rows)


def langfuse_labels() -> List[HumanLabel]:
    """Read dimension scores from Langfuse (later), correlated by trace id. Lazily
    imports the Langfuse client; credentials come from the ``LANGFUSE_*`` env, the
    same trio tracing uses. Structured to share the ``_rows_to_labels`` normalizer."""
    from langfuse import Langfuse  # lazy

    client = Langfuse()
    rows: List[Dict[str, Any]] = []
    try:
        scores = client.api.score.get().data  # Langfuse scores API
    except Exception:  # noqa: BLE001 - API shape varies by Langfuse version
        logger.warning("Could not read Langfuse scores (client/API version?).")
        return []
    for score in scores:
        rows.append(
            {
                "key": getattr(score, "trace_id", None),
                "dimension": getattr(score, "name", None),
                "value": getattr(score, "value", None) or getattr(score, "string_value", None),
            }
        )
    return _rows_to_labels(rows)


def load_labels(source: str, *, log: Optional[str] = None) -> List[HumanLabel]:
    """Dispatch to a label source: ``vendorfree`` (default, needs ``log``),
    ``phoenix``, or ``langfuse``. Backend sources read connection details from env."""
    source = (source or "vendorfree").strip().lower()
    if source == "vendorfree":
        if not log:
            raise ValueError("vendorfree label source needs a --log path")
        return vendorfree_labels(log)
    if source == "phoenix":
        return phoenix_labels()
    if source == "langfuse":
        return langfuse_labels()
    raise ValueError(f"unknown label source {source!r}; use vendorfree|phoenix|langfuse")
