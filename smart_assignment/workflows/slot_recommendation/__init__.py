"""
Delivery slot recommendation workflow.

Two front ends share the same deterministic pipeline (pipeline.py):

  - `graph.py`'s `root_agent` -- the original deterministic ADK `Workflow`
    graph, a one-shot batch path. Exposed here so it's directly importable
    (`adk web smart_assignment/workflows/slot_recommendation`).
  - `conversational_agent.py`'s `root_agent` -- the conversational
    `LlmAgent` that collects a prospect's details over multiple turns and
    calls the pipeline as tools. This is what `smart_assignment/agent.py`
    points at as the top-level `root_agent` for `adk run`/`adk web`.
"""

from smart_assignment.workflows.slot_recommendation.graph import root_agent

__all__ = ["root_agent"]
