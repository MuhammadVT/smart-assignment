"""
Prompt text for the workflow's optional LLM reasoning layer, kept separate
from logic so prompt iteration doesn't require touching orchestration code.

Note the division of labor: the LLM does NOT choose the slot or compute the
score — that is deterministic (constraints.py + scoring.py). The model only
turns the already-decided, fully-quantified result into a fluent, auditable
explanation. This keeps the decision reproducible and testable while still
allowing a natural-language narrative.
"""

from __future__ import annotations

REASONING_SYSTEM_PREAMBLE = (
    "You are a Sysco delivery-slot assignment specialist. Rewrite the following "
    "machine-generated decision as a concise, auditable explanation (2-4 sentences) "
    "that an operations manager could act on and explain to the customer. Preserve "
    "the decision, the route/day/window, and every number exactly — do not re-rank, "
    "second-guess, or invent facts."
)


def build_reasoning_prompt(machine_trace: str) -> str:
    """Wrap a deterministic decision trace into the LLM rewrite prompt."""
    return f"{REASONING_SYSTEM_PREAMBLE}\n\nDecision to rewrite:\n{machine_trace}"
