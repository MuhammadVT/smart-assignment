"""
The durable record of every automated LLM-judge verdict -- what a judge scored,
why, and with which judge model.

``eval/test_quality.py`` and ``eval/test_rationale_faithfulness.py`` ask an LLM
judge to score an output, compare that score to a threshold, and fail the test
when it falls short. Without this module the score itself is *ephemeral*: it
lives on the metric object for one loop iteration and is gone when the process
exits. A failing verdict survives only as pytest output; a PASSING verdict
leaves no trace at all -- so "brief_quality passed" tells you nothing about
whether it scored 0.55 or 0.95, and a judge that quietly drifts (because the
judge MODEL changed, not the agent) is invisible.

It is also the missing producer in the flywheel: ``scripts/calibrate_judges.py``
already consumes ``{decision_id: {dimension: {passed, score}}}`` verdicts
"produced by running the judges", and nothing produced them. Recording here is
what lets a later phase measure the judges against human labels rather than
trusting them.

**A deliberate sibling of the human-feedback log.** This mirrors
``smart_assignment/feedback/store.py`` on purpose -- same append-only JSON Lines
format, same one-self-describing-record-per-line, same defensive writes. The two
logs are counterparts (human labels vs. machine verdicts on the same quality
dimensions) and are meant to be read together. They stay separate files because
``feedback_data/annotations.jsonl`` is the *production* audit trail of judgments
on real customer decisions, while this is *eval-run* output; merging them would
mix provenance domains and let judge rows reach human-label curation.

**Robustness, asymmetric on purpose.** A failure to *persist* a verdict is
logged and swallowed: recording is additive, and a bad path or a full disk must
never turn an advisory eval red. A failure of the *judge call itself* is NOT
swallowed -- it propagates, because a judge that cannot score is a real eval
failure, and recording it as "no score" would bury it.

**Vocabulary.** ``dimension`` is the judge's name.
``brief_quality``/``response_clarity`` are the canonical human-annotation
dimensions (``eval/judge_calibration.DIMENSIONS``, mirroring
``deployment/phoenix/README.md``), so a machine verdict and a human label speak
one language and join on ``(decision_id, dimension)``. No validation is imposed
here: a judge may legitimately score a dimension with no human counterpart yet
(``rationale_faithfulness``), and refusing such a record would drop exactly the
data this log exists to keep.

Imports stay light and credential-free. The metric and test-case objects are
duck-typed -- only ``a_measure``/``score``/``reason``/``threshold`` and
``actual_output`` are touched -- so importing this module never pulls DeepEval,
and ``Config``/``eval.dataset`` are imported lazily inside the functions that
need them.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)

# How much of the judged text to keep inline. Enough to recognize the output at a
# glance in a log tail; the full text is identified by ``output_ref`` instead of
# duplicated here (see JudgeVerdictRecord).
_EXCERPT_CHARS = 300

# Serialize appends within a process so two judged cases can't interleave partial
# lines. (JSONL across processes is still append-safe line by line; this lock
# just protects the same-process fast path -- same reasoning as feedback/store.py.)
_WRITE_LOCK = threading.Lock()


@dataclass(frozen=True)
class JudgeVerdictRecord:
    """One judge's verdict on one judged output.

    ``eval_id`` identifies the eval case; ``decision_id`` is the id the verdict
    JOINS on -- for a golden fixture it is just the ``eval_id``, but for a case
    curated from production feedback it is the real decision the human also
    labeled, which is what makes calibration possible. It is always populated
    (never ``None`` on a recorded row) so a reader never has to guess.

    ``output_ref`` is a content hash of the full judged text and
    ``output_excerpt`` its first few hundred characters. Storing a ref rather
    than the whole output keeps the log lean and avoids duplicating text that
    already lives in ``eval/data/golden_responses.json`` -- while still
    answering "did the judged text change between runs?" for
    ``test_rationale_faithfulness``, whose prose is regenerated every run and
    stored nowhere else.

    ``judge`` records the JUDGE's model/backend, kept separate from ``run`` (the
    product model + dataset the judged output came from, as produced by
    ``eval.dataset.run_provenance``). Conflating them would hide the exact
    question calibration asks: did the score move because the agent changed, or
    because the judge did?

    ``judged_at`` is supplied by the caller, so this stays a pure value with no
    hidden clock -- the same discipline ``feedback.schema.FeedbackRecord`` holds
    for ``created_at``.
    """

    eval_id: str
    dimension: str
    threshold: float
    passed: bool
    decision_id: str
    score: Optional[float] = None
    reason: Optional[str] = None
    output_ref: Optional[str] = None
    output_excerpt: Optional[str] = None
    judged_at: Optional[str] = None
    judge: Dict[str, Any] = field(default_factory=dict)
    run: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """A JSON-serializable dict for the durable log."""
        return asdict(self)

    def failure_line(self) -> str:
        """The one-line "below threshold" message a test collects into its
        assertion. Kept here so every judge test reports failures identically."""
        score = "n/a" if self.score is None else f"{self.score:.2f}"
        return f"{self.eval_id}: {score} < {self.threshold} -- {self.reason}"


def utc_now_iso() -> str:
    """Now, as an ISO-8601 UTC timestamp -- the ``judged_at`` callers stamp."""
    return datetime.now(timezone.utc).isoformat()


def content_ref(text: str) -> str:
    """A stable short content hash of the judged text, in the same
    ``sha256:<16 hex>`` form ``eval.dataset.dataset_content_ref`` uses, so the
    two provenance refs read alike."""
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def excerpt(text: str) -> str:
    """The head of the judged text, ellipsized, for at-a-glance readability."""
    collapsed = " ".join(text.split())
    if len(collapsed) <= _EXCERPT_CHARS:
        return collapsed
    return collapsed[:_EXCERPT_CHARS] + "..."


def _opt_float(value: Any) -> Optional[float]:
    """``value`` as a float, or ``None`` when it isn't a usable number. Booleans
    are rejected explicitly (``True`` is an ``int`` in Python, and a judge that
    "scored" ``True`` has not produced a score)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def default_log_path() -> str:
    """Where verdicts are recorded (``Config.judge_log_path``). An empty value
    disables recording -- the path is the switch."""
    from smart_assignment.shared.config import DEFAULT_CONFIG

    return DEFAULT_CONFIG.judge_log_path


def judge_provenance(metric: Any = None) -> Dict[str, Any]:
    """Who did the judging: the resolved quality-judge model and backend, plus
    the metric class. Deliberately resolved via ``Config.for_role`` rather than
    read off the metric, so it reports the same model
    ``eval/deepeval_llm.py`` actually calls."""
    from smart_assignment.shared.config import DEFAULT_CONFIG, ROLE_QUALITY_JUDGE

    provenance: Dict[str, Any] = {
        "backend": DEFAULT_CONFIG.llm_backend,
        "model": DEFAULT_CONFIG.resolved_model(ROLE_QUALITY_JUDGE),
    }
    if metric is not None:
        provenance["metric"] = type(metric).__name__
    return provenance


@lru_cache(maxsize=1)
def current_run_provenance() -> Dict[str, Any]:
    """The dataset/model provenance of the run being judged -- the SAME block
    ``eval.capture`` records next to each captured response, so a verdict and the
    response it scored are attributable to the same world.

    Computed ONCE per process, and that is not just an optimization: replaying a
    case mutates the in-memory mock fixtures, so
    ``eval.dataset.dataset_content_ref`` returns a *different* hash after every
    pipeline run. Recomputing per verdict would stamp each record with a
    different "dataset" -- provenance that identifies the run's mutation state
    instead of the dataset. ``eval.capture`` avoids the same trap by snapshotting
    before it runs anything; a caller that replays cases should likewise take
    this value BEFORE the first replay and pass it as ``run=``."""
    from eval.dataset import resolve_eval_dataset, run_provenance

    return run_provenance(resolve_eval_dataset())


def append_verdict(record: JudgeVerdictRecord, path: Optional[str] = None) -> bool:
    """Append one verdict to the JSONL log, creating parent dirs as needed.

    Returns ``True`` when written, ``False`` when recording is disabled (empty
    path) or the write failed -- never raises. A verdict that cannot be persisted
    must not fail the eval that produced it (see the module docstring)."""
    target = default_log_path() if path is None else path
    if not (target or "").strip():
        logger.debug("Judge log path is empty; not recording %s.", record.dimension)
        return False
    try:
        directory = os.path.dirname(os.path.abspath(target))
        if directory:
            os.makedirs(directory, exist_ok=True)
        line = json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True)
        with _WRITE_LOCK:
            with open(target, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        return True
    except Exception:  # noqa: BLE001 - persistence is best-effort; never raise
        logger.warning("Could not append judge verdict to %s; not recorded.", target, exc_info=True)
        return False


def _record_from_dict(raw: dict) -> JudgeVerdictRecord:
    """Rebuild a record from a parsed JSONL line, tolerating extra keys and an
    older/hand-edited row, so the log stays readable as the shape evolves."""
    eval_id = str(raw.get("eval_id", ""))
    return JudgeVerdictRecord(
        eval_id=eval_id,
        dimension=str(raw.get("dimension", "")),
        threshold=_opt_float(raw.get("threshold")) or 0.0,
        passed=bool(raw.get("passed")),
        decision_id=str(raw.get("decision_id") or eval_id),
        score=_opt_float(raw.get("score")),
        reason=raw.get("reason"),
        output_ref=raw.get("output_ref"),
        output_excerpt=raw.get("output_excerpt"),
        judged_at=raw.get("judged_at"),
        judge=raw.get("judge") or {},
        run=raw.get("run") or {},
    )


def iter_verdicts(path: Optional[str] = None) -> Iterator[JudgeVerdictRecord]:
    """Yield every recorded verdict, skipping blank or malformed lines (logged at
    debug). A missing log yields nothing -- reading an empty history is an empty
    result, not an error."""
    target = default_log_path() if path is None else path
    if not (target or "").strip() or not os.path.exists(target):
        return
    with open(target, "r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield _record_from_dict(json.loads(line))
            except Exception:  # noqa: BLE001 - one bad line must not abort the read
                logger.debug(
                    "Skipping malformed judge-verdict line %d in %s.", lineno, target, exc_info=True
                )


def read_verdicts(path: Optional[str] = None) -> List[JudgeVerdictRecord]:
    """All recorded verdicts as a list (convenience over ``iter_verdicts``)."""
    return list(iter_verdicts(path))


async def measure_and_record(
    metric: Any,
    test_case: Any,
    *,
    eval_id: str,
    dimension: str,
    decision_id: Optional[str] = None,
    run: Optional[Dict[str, Any]] = None,
    path: Optional[str] = None,
) -> JudgeVerdictRecord:
    """Score ``test_case`` with ``metric``, record the verdict, and return it.

    This is the single seam every judge test goes through, because they all did
    the same four things by hand: measure, read the score/reason back off the
    metric, compare to the threshold, and format a failure line. Centralizing it
    means the recording can't be forgotten at a new call site -- and, critically,
    that the score is SNAPSHOT the instant it is produced: the metrics are
    module-level singletons reused across cases, so ``metric.score`` belongs to
    whichever case was measured last.

    A judge-call failure propagates (see the module docstring); only the write is
    best-effort."""
    await metric.a_measure(test_case)

    score = _opt_float(getattr(metric, "score", None))
    threshold = _opt_float(getattr(metric, "threshold", None)) or 0.0
    output = str(getattr(test_case, "actual_output", "") or "")

    record = JudgeVerdictRecord(
        eval_id=eval_id,
        dimension=dimension,
        threshold=threshold,
        # A judge that returned no usable number has NOT passed; treating it as a
        # pass would let a broken judge wave everything through.
        passed=score is not None and score >= threshold,
        decision_id=decision_id or eval_id,
        score=score,
        reason=getattr(metric, "reason", None),
        output_ref=content_ref(output) if output else None,
        output_excerpt=excerpt(output) if output else None,
        judged_at=utc_now_iso(),
        judge=judge_provenance(metric),
        run=current_run_provenance() if run is None else run,
    )
    append_verdict(record, path)
    return record
