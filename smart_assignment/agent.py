"""
ADK entry point. ADK's CLI (`adk run`, `adk web`) and deployment tooling
look for a `root_agent` here.

`root_agent` is the conversational `LlmAgent`
(workflows/slot_recommendation/conversational_agent.py): it collects a
prospect's details over multiple turns and calls the same deterministic
pipeline (pipeline.py) as tools, rather than computing anything itself.

The original deterministic ADK `Workflow` graph
(workflows/slot_recommendation/graph.py) is unchanged and still directly
importable for a one-shot, non-interactive path -- e.g.
`adk run smart_assignment.workflows.slot_recommendation.graph` -- or for a
future backend trigger that doesn't need a conversation.

If/when a second, architecturally different workflow is added under
smart_assignment/workflows/, this file is the place to decide how they
coexist -- e.g. point root_agent at a small dispatch/router agent that
delegates by intent. That decision belongs here, not inside any individual
workflow.
"""

from smart_assignment.workflows.slot_recommendation.conversational_agent import (
    root_agent,
)

__all__ = ["root_agent"]
