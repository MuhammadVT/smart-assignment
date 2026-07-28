"""
Batch (non-conversational) mode: run the deterministic slot-recommendation
pipeline unattended over many Salesforce-sourced prospects, one pass each, and
write one result per prospect for the sales-consultant Customer View.

This is a new orchestration seam AROUND the existing pipeline, not a fork of it:
the decision logic (hard constraints, scoring, grounded route-slot decision,
triage brief) is unchanged. Only the two human-in-the-loop steps are replaced by
deterministic policies -- see ``runner.py``. The mode is an entry point
(``scripts/run_batch.py``), not a global config switch, so the conversational
paths are untouched.
"""

from __future__ import annotations

from smart_assignment.batch.runner import BatchRunner, BatchSummary, run_one
from smart_assignment.batch.sink import (
    OUTCOME_ESCALATE,
    OUTCOME_NEEDS_ATTENTION,
    OUTCOME_RECOMMEND,
    BatchRecord,
    JsonlResultSink,
    ListResultSink,
    ResultSink,
)
from smart_assignment.batch.source import (
    MockProspectSource,
    Prospect,
    ProspectSource,
)

__all__ = [
    "BatchRunner",
    "BatchSummary",
    "run_one",
    "BatchRecord",
    "ResultSink",
    "ListResultSink",
    "JsonlResultSink",
    "OUTCOME_RECOMMEND",
    "OUTCOME_ESCALATE",
    "OUTCOME_NEEDS_ATTENTION",
    "Prospect",
    "ProspectSource",
    "MockProspectSource",
]
