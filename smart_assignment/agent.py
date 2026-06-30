"""
ADK entry point. ADK's CLI (`adk run`, `adk web`) and deployment tooling
look for a `root_agent` here.

Today this repo has one workflow (slot_recommendation), so root_agent is
that workflow directly. If/when a second, architecturally different
workflow is added under smart_assignment/workflows/, this file is the
place to decide how they coexist -- e.g. point root_agent at a small
dispatch/router agent that delegates by intent, or expose each workflow
separately for `adk web smart_assignment/workflows/<name>` and leave this
file pointing at the default. That decision belongs here, not inside any
individual workflow.
"""

from smart_assignment.workflows.slot_recommendation import root_agent

__all__ = ["root_agent"]
