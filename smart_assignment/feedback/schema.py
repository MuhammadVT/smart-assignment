"""
The structured shape of one piece of human (or automated) feedback.

A ``FeedbackRecord`` is a quality judgment *about a decision the pipeline already
made* -- it never carries an actionable value the system will act on. It is the
unit written to the durable log (``store.py``), emitted as an OTLP span
(``emit.py``), and later curated into eval cases (``curate.py``).

The shape is deliberately vendor-neutral. It borrows the *concepts* that every
annotation system shares -- a target span, who evaluated it, a categorical label,
an optional numeric score, a note -- without importing any vendor's SDK or
committing to any vendor's payload. Mapping a record onto a specific backend is a
thin, swappable step at the edge (see ``emit.py``), not baked into this type.

Nothing here needs credentials or heavy dependencies, so importing this module is
free (the same discipline the rest of the repo's LLM layers hold).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

# Who produced the judgment. Mirrors the industry-standard "annotator kind" so a
# human thumbs-down, an LLM-as-judge score, and a deterministic code check all
# flow through the SAME record and the same neutral pipe.
ANNOTATOR_HUMAN = "HUMAN"
ANNOTATOR_LLM = "LLM"
ANNOTATOR_CODE = "CODE"
_ANNOTATOR_KINDS = frozenset({ANNOTATOR_HUMAN, ANNOTATOR_LLM, ANNOTATOR_CODE})

# What the feedback is *about*. The first production increment annotates the
# final recommendation as a whole; the enum leaves room to target an individual
# grounded step later without a schema change.
DECISION_FINAL_RESPONSE = "final_response"
DECISION_RECOMMEND_OR_ESCALATE = "recommend_or_escalate"
DECISION_SLOTPICK = "slotpick"
DECISION_ROUTE_SLOT = "route_slot"
DECISION_ADDRESS_RESOLVE = "address_resolve"
_DECISION_KINDS = frozenset(
    {
        DECISION_FINAL_RESPONSE,
        DECISION_RECOMMEND_OR_ESCALATE,
        DECISION_SLOTPICK,
        DECISION_ROUTE_SLOT,
        DECISION_ADDRESS_RESOLVE,
    }
)


class FeedbackValidationError(ValueError):
    """Raised when a ``FeedbackRecord`` is structurally invalid. A distinct type
    so the capture layer can reject a bad annotation deterministically (and a web
    endpoint can turn it into a 400) without catching unrelated ``ValueError``s."""


@dataclass(frozen=True)
class FeedbackTarget:
    """Which decision an annotation is about.

    ``trace_id``/``span_id`` are the OpenTelemetry coordinates of the decision,
    present only when tracing was on for that run (best-effort -- see
    ``shared.tracing.current_trace_context``). ``decision_id`` is always present:
    a stable id the app minted for the decision, so feedback can be joined back to
    it even with tracing off. ``session_id`` groups a browser conversation."""

    decision_id: str
    decision_kind: str = DECISION_FINAL_RESPONSE
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    session_id: Optional[str] = None


@dataclass(frozen=True)
class FeedbackRecord:
    """One quality judgment about a decision.

    ``label`` is the categorical verdict (e.g. ``"thumbs_up"``/``"thumbs_down"``,
    or a finer ``"wrong_route"``); ``score`` an optional normalized number;
    ``note`` a freeform explanation (the one field that may carry PII -- scrubbed
    by ``feedback.scrub`` before it is persisted when ``feedback_scrub_pii`` is
    on). ``context`` is an optional snapshot of the decision's facts (name,
    address, outcome, ...) captured for later curation -- also scrub-eligible.
    ``created_at`` is an ISO-8601 timestamp the *caller* supplies (kept out of
    this module so the record stays a pure, side-effect-free value)."""

    target: FeedbackTarget
    label: str
    annotator_kind: str = ANNOTATOR_HUMAN
    score: Optional[float] = None
    note: Optional[str] = None
    annotator_id: Optional[str] = None
    created_at: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """A JSON-serializable dict for the durable log / transport."""
        return asdict(self)


def validate_feedback(record: FeedbackRecord) -> None:
    """Reject a structurally invalid record, raising ``FeedbackValidationError``.

    Deterministic and dependency-free -- this is the "something rejects bad input
    before anything acts on it" guarantee applied to feedback capture. Checks:
    a non-empty ``decision_id`` and ``label``, a known ``annotator_kind`` and
    ``decision_kind``, and a ``score`` that is finite and within a sane range
    ([-1, 1] covers both 0/1 thumb encodings and signed scores)."""
    if not isinstance(record.target, FeedbackTarget):
        raise FeedbackValidationError("feedback target must be a FeedbackTarget")
    if not (record.target.decision_id or "").strip():
        raise FeedbackValidationError("feedback target.decision_id is required")
    if record.target.decision_kind not in _DECISION_KINDS:
        raise FeedbackValidationError(
            f"unknown decision_kind {record.target.decision_kind!r}; "
            f"valid: {sorted(_DECISION_KINDS)}"
        )
    if not (record.label or "").strip():
        raise FeedbackValidationError("feedback label is required")
    if record.annotator_kind not in _ANNOTATOR_KINDS:
        raise FeedbackValidationError(
            f"unknown annotator_kind {record.annotator_kind!r}; "
            f"valid: {sorted(_ANNOTATOR_KINDS)}"
        )
    if record.score is not None:
        score = record.score
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            raise FeedbackValidationError("feedback score must be a number")
        # NaN/inf compare falsy against bounds; catch them explicitly.
        if score != score or score in (float("inf"), float("-inf")):
            raise FeedbackValidationError("feedback score must be finite")
        if not (-1.0 <= float(score) <= 1.0):
            raise FeedbackValidationError("feedback score must be within [-1, 1]")
