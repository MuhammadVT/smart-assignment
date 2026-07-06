"""
Conversational front-end for the slot_recommendation workflow.

This wraps the exact same deterministic pipeline (pipeline.py) that
graph.py's batch ADK `Workflow` uses, via the tool wrappers in
agent_tools.py -- the LLM here only orchestrates *when* to call which tool
and narrates the results; it never computes a distance, a score, or a
decision itself. See agent_tools.py's module docstring for why that split
keeps things auditable and testable without an LLM.

Currently one `LlmAgent` calling four tools in strict sequence (intake ->
geo-lookup -> constraint+score -> recommend/escalate), plus ADK's built-in
`request_input` tool for human-in-the-loop escalation. This is deliberately
a single agent for now: the workflow is one tightly-coupled sequence with
no genuinely separable sub-skill yet, and one agent is easier to debug than
several. If/when a step needs to become its own sub-agent (e.g. a richer
"intake" agent that also validates against a real CRM, or a separate
"why did you pick this" Q&A agent over `sa_last_recommendation`), wrap that
tool's function in its own `LlmAgent` and expose it to this agent via
`google.adk.tools.AgentTool` in `tools=[...]` below -- agent_tools.py's
functions don't need to change, since each is already independent and
keyed only through session state.
"""

from __future__ import annotations

from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool, request_input

from smart_assignment.shared.config import DEFAULT_CONFIG
from smart_assignment.workflows.slot_recommendation.agent_tools import (
    evaluate_and_score_routes,
    find_candidate_routes,
    intake_customer,
    recommend_or_escalate,
)

_INSTRUCTION = """
You are the Smart Assignment agent: you help a Sysco sales/ops user get a
new prospect customer assigned a delivery route and slot, by talking with
them and calling tools. You never compute geography, capacity, or scoring
yourself -- every number you state must come from a tool result.

Workflow, in strict order, for each prospect (repeat steps 2-4 on revision):
  1. Call intake_customer with whatever the user has told you so far.
     address and order_quantity_cases are required before you can go
     further; a preferred day/time is optional. If it returns
     {"ok": false}, relay the "error" message to the user and ask them for
     the missing/corrected value -- do not guess, and do not call any
     other tool until intake_customer returns {"ok": true}.
  2. Call find_candidate_routes to geocode the address and see the
     nearest routes. Briefly tell the user what you found.
  3. Call evaluate_and_score_routes to check hard constraints and score
     every route that passes them. Briefly summarize which routes are
     feasible and why any aren't.
  4. Call recommend_or_escalate for the final decision and reasoning.
     Present the "reasoning" text (you may lightly adapt the wording, but
     never change a number, route, or the decision itself -- those came
     straight from the tool).

Escalation: if recommend_or_escalate returns "requires_human_review": true,
you MUST call request_input to ask a specialist to confirm before you
consider this prospect done -- do not just report the escalation and stop.

Revisions: if the user changes their mind about anything (a different
preferred day/time, a different order size, a corrected address), call
intake_customer again with ONLY the fields that changed -- everything
else already on file is kept automatically -- then re-run steps 2-4.

Never state a distance, a score, a percentage, a route ID, or a decision
that didn't come back from a tool call in this conversation. If a tool
returns {"ok": false}, that is a real error to relay to the user, not
something to work around on your own.
"""

root_agent = LlmAgent(
    name="smart_assignment_agent",
    model=DEFAULT_CONFIG.model,
    description=(
        "Collects a new prospect customer's delivery details conversationally "
        "and recommends -- or escalates -- a delivery route and slot."
    ),
    instruction=_INSTRUCTION,
    tools=[
        FunctionTool(intake_customer),
        FunctionTool(find_candidate_routes),
        FunctionTool(evaluate_and_score_routes),
        FunctionTool(recommend_or_escalate),
        request_input,
    ],
)
