"""
Batch result records and sinks.

A :class:`BatchRecord` is the batch's OWNED output contract -- the union of what
the Customer View renders (``payload``, the exact ``build_workflow_payload`` dict,
so a batch result renders with **no** frontend change) and the batch envelope
(id, timestamp, outcome, escalation brief, or an error). A :class:`ResultSink`
persists records; :class:`JsonlResultSink` is the durable default and a real
API/DB sink implements the same one-method protocol with no runner change.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, Union

# The three terminal outcomes a batch run assigns per prospect.
OUTCOME_RECOMMEND = "recommend"
OUTCOME_ESCALATE = "escalate"
OUTCOME_NEEDS_ATTENTION = "needs_attention"


@dataclass
class BatchRecord:
    """One prospect's batch result.

    ``payload`` is the exact ``build_workflow_payload`` dict (``frontendHtml``,
    ``resultHtml``, ``map``, ...), present on ``recommend`` and ``escalate`` so the
    Customer View renders it unchanged; it is ``None`` on ``needs_attention`` (no
    decision was produced -- e.g. the address wouldn't geocode). ``triage_brief`` and
    ``review_reason`` are set on ``escalate``; ``error`` on ``needs_attention``."""

    prospect_id: str
    generated_at: str
    outcome: str
    payload: Optional[dict] = None
    review_reason: Optional[str] = None
    triage_brief: Optional[str] = None
    error: Optional[str] = None

    def to_json(self) -> dict:
        """A tidy JSON-safe dict: the envelope always, optional fields only when set,
        and ``payload`` last (it is by far the largest)."""
        out: dict = {
            "prospect_id": self.prospect_id,
            "generated_at": self.generated_at,
            "outcome": self.outcome,
        }
        for key in ("review_reason", "triage_brief", "error"):
            value = getattr(self, key)
            if value:
                out[key] = value
        if self.payload is not None:
            out["payload"] = self.payload
        return out


class ResultSink(Protocol):
    """Where batch results go. The real Customer-View sink (an API POST or a DB
    write) implements this same ``emit`` and drops in with no runner change."""

    def emit(self, record: BatchRecord) -> None:
        ...


class ListResultSink:
    """Collects records in memory -- for tests and small in-process runs."""

    def __init__(self) -> None:
        self.records: list[BatchRecord] = []

    def emit(self, record: BatchRecord) -> None:
        self.records.append(record)


class JsonlResultSink:
    """Append-only JSONL sink -- one JSON record per line, the durable default (the
    same persist-then-forget discipline as the feedback store). Used as a context
    manager so the file is flushed and closed even if the run raises::

        with JsonlResultSink("out.jsonl") as sink:
            BatchRunner(source, sink).run()
    """

    def __init__(self, path: Union[str, Path]) -> None:
        self._path = Path(path)
        self._fh = None

    def __enter__(self) -> "JsonlResultSink":
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self._path.open("w", encoding="utf-8")
        return self

    def emit(self, record: BatchRecord) -> None:
        if self._fh is None:
            raise RuntimeError("JsonlResultSink must be used as a context manager")
        self._fh.write(json.dumps(record.to_json(), ensure_ascii=False) + "\n")
        self._fh.flush()

    def __exit__(self, *exc) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
