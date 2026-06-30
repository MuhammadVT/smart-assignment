"""
Instruction text and structured-output/input schemas for this workflow's
LLM node(s). Kept separate from nodes.py so prompt iteration doesn't
require touching orchestration code, and so eval/ can reference the
schemas directly when building golden datasets.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SlotPromptInput(BaseModel):
    prompt_text: str


class SlotRecommendationOutput(BaseModel):
    recommended_route_id: str = Field(description="route_id of the recommended slot")
    recommended_day: str = Field(description="Day of week of the recommended slot")
    recommended_window_start: str = Field(description="HH:MM start of proposed arrival window")
    recommended_window_end: str = Field(description="HH:MM end of proposed arrival window")
    confidence: float = Field(description="0.0-1.0 confidence in this recommendation")
    reasoning: str = Field(
        description=(
            "2-4 sentence explanation referencing the SPECIFIC tradeoffs "
            "considered: geographic fit, capacity utilization, customer "
            "preference match, and reliability implications. Must be "
            "concrete enough for an ops reviewer to audit the decision."
        )
    )
    rejected_alternatives: list[str] = Field(
        description=(
            "For each other feasible option NOT chosen, one short sentence "
            "explaining why it was passed over."
        )
    )


RECOMMEND_SLOT_INSTRUCTION = """You are a delivery slot recommendation specialist for a
foodservice distributor. You will be given a customer's profile and a list
of delivery slots that have ALREADY been verified to satisfy all hard
operational constraints (vehicle capacity, temperature compatibility,
driver hours). Do not re-evaluate those constraints -- they're guaranteed.

Your job is to choose the SINGLE BEST option by optimizing for delivery
RELIABILITY and CUSTOMER SATISFACTION, not just route efficiency in
isolation. Concretely weigh:

1. Geographic fit (geographic_fit_score): tighter clustering with other
   stops on that route generally means more predictable arrival times
   and lower risk of cascading delays from upstream stops.
2. Capacity utilization after assignment: avoid pushing a route to its
   capacity ceiling if a similarly-good option leaves more buffer --
   tightly-packed routes are more fragile to day-of disruption (traffic,
   short-staffing, large unexpected orders from existing customers).
3. Customer preference match: a stated day/time preference should be
   honored when it doesn't meaningfully trade off against (1) or (2).
4. When options are close, prefer the option with more buffer/slack,
   since reliability for a NEW customer relationship matters more than
   marginal efficiency gains.

Output your confidence honestly. If two options are nearly tied or the
tradeoffs are genuinely ambiguous, reflect that with a LOWER confidence
score (below 0.7) rather than overstating certainty -- a human reviewer
will check anything under 0.7.

Your reasoning must name the specific factors that drove the decision,
in language an operations manager could audit and explain to the
customer if asked.

Input data:
{prompt_text}"""
