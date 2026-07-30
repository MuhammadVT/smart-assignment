"""
Batch (non-interactive) mode: run the workflow unattended over many
Salesforce-sourced prospects, one per turn, and write one result per prospect for
the sales-consultant Customer View.

It drives the REAL agent architecture (``agent.build_batch_agent``) via
:class:`~smart_assignment.batch.agent_runner.AgentBatchRunner`, so the batch
inherits the agent's natural-language reasoning and the escalation-triage brief --
non-interactively (intake is seeded from the CRM, an escalation is recorded rather
than paused on). The deterministic pipeline (``runner.run_one``) remains the
**floor** the runner falls back to whenever the agent is unavailable (no
credentials) or a turn fails, so batch is never worse than the deterministic
baseline and still runs fully offline.

This is an orchestration seam AROUND the existing pipeline, not a fork of it: the
decision logic (hard constraints, scoring, grounded route-slot decision, triage
brief) is unchanged. The mode is an entry point (``scripts/run_batch.py``), not a
global config switch, so the conversational paths are untouched.
"""

from __future__ import annotations

from smart_assignment.batch.agent_runner import AgentBatchRunner
from smart_assignment.batch.runner import BatchSummary, run_one
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
    "AgentBatchRunner",
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
