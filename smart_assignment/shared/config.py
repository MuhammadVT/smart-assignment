"""
Cross-workflow configuration. Centralizing tunable thresholds here means
ops can adjust business rules (e.g. via environment variables) without
touching workflow or tools code.

[ASSUMPTION] Default values below are starting points, not validated
Sysco policy — see README.md "Assumptions requiring correction".
"""

from __future__ import annotations

import os


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


# [ASSUMPTION] Max fraction of rated vehicle capacity a route may be filled
# to after assigning a new customer. Confirm against Sysco's real capacity
# buffer policy (for late-add stops, traffic, short-staffing, etc).
MAX_UTILIZATION_AFTER_ASSIGNMENT: float = _float_env("SMART_ASSIGNMENT_MAX_UTILIZATION", 0.90)

# Confidence threshold below which the slot_recommendation workflow
# escalates to a human reviewer instead of auto-committing. Arbitrary
# starting point — tune against real human-override rates once available.
SLOT_RECOMMENDATION_CONFIDENCE_THRESHOLD: float = _float_env(
    "SMART_ASSIGNMENT_CONFIDENCE_THRESHOLD", 0.70
)

# Default Gemini model used across workflows unless a workflow overrides it.
DEFAULT_MODEL: str = os.environ.get("SMART_ASSIGNMENT_MODEL", "gemini-flash-latest")
