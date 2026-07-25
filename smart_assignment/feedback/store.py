"""
The durable, backend-independent record of every annotation -- the audit source
of truth and the input to curation.

Feedback is written here *first*, before any observability emit, precisely
because the trace backend is best-effort and swappable while auditability is a
hard guarantee (see CLAUDE.md): a human must be able to reconstruct what feedback
was given even if Phoenix/Langfuse/whatever was down at the time. The format is
append-only JSON Lines -- one self-describing record per line -- so it is trivial
to tail, grep, ship, and re-read for curation, with no database and no vendor.

Writes are defensive: a failure to persist (bad path, full disk, permissions)
is logged and swallowed, never raised into the caller. Feedback is additive; a
lost annotation must not break the request that produced it.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Iterator, List

from smart_assignment.feedback.schema import (
    FeedbackRecord,
    FeedbackTarget,
)

logger = logging.getLogger(__name__)

# Serialize appends within a process so concurrent web requests can't interleave
# partial lines. (JSONL across processes is still append-safe line-by-line; this
# lock just protects the same-process fast path.)
_WRITE_LOCK = threading.Lock()


def append_record(record: FeedbackRecord, path: str) -> bool:
    """Append one record to the JSONL log at ``path``, creating parent dirs as
    needed. Returns ``True`` on success, ``False`` (logged) on any failure --
    never raises, so a storage problem can't break the feedback request."""
    try:
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        line = json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True)
        with _WRITE_LOCK:
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        return True
    except Exception:  # noqa: BLE001 - persistence is best-effort; never raise
        logger.warning("Could not append feedback to %s; annotation not persisted.", path,
                       exc_info=True)
        return False


def _record_from_dict(raw: dict) -> FeedbackRecord:
    """Rebuild a ``FeedbackRecord`` from a parsed JSONL line, tolerating extra
    keys and a missing/legacy target so a hand-edited log still reads."""
    target_raw = raw.get("target") or {}
    target = FeedbackTarget(
        decision_id=str(target_raw.get("decision_id", "")),
        decision_kind=target_raw.get("decision_kind", "final_response"),
        trace_id=target_raw.get("trace_id"),
        span_id=target_raw.get("span_id"),
        session_id=target_raw.get("session_id"),
    )
    return FeedbackRecord(
        target=target,
        label=raw.get("label", ""),
        annotator_kind=raw.get("annotator_kind", "HUMAN"),
        score=raw.get("score"),
        note=raw.get("note"),
        annotator_id=raw.get("annotator_id"),
        created_at=raw.get("created_at"),
        context=raw.get("context") or {},
    )


def iter_records(path: str) -> Iterator[FeedbackRecord]:
    """Yield every record in the JSONL log, skipping blank or malformed lines
    (logged at debug). A missing file yields nothing -- curation over an empty
    history is just an empty result, not an error."""
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield _record_from_dict(json.loads(line))
            except Exception:  # noqa: BLE001 - one bad line must not abort the read
                logger.debug("Skipping malformed feedback line %d in %s.", lineno, path,
                             exc_info=True)


def read_records(path: str) -> List[FeedbackRecord]:
    """All records in the log as a list (convenience over ``iter_records``)."""
    return list(iter_records(path))
