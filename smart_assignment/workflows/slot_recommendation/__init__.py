"""
Delivery slot recommendation workflow (graph-based architecture).

Exposes `root_agent`, the ADK Workflow object for this workflow, so it
can be imported either directly (`adk web smart_assignment/workflows/
slot_recommendation`) or composed by smart_assignment/agent.py.
"""

from smart_assignment.workflows.slot_recommendation.graph import root_agent

__all__ = ["root_agent"]
