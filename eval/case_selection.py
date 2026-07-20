"""
Shared SMART_ASSIGNMENT_EVAL_IDS handling for eval/test_eval.py and
eval/capture.py. Both let a developer run a comma-separated subset of
eval/golden_cases.py's GOLDEN_CASES to cut live-LLM cost while iterating (see
"Running a subset locally while developing" in eval/README.md). Neither CI nor
the hermetic suite sets this env var, so unset returns every case unchanged --
the knob only exists to be opted into, never to silently narrow a real run.
"""

from __future__ import annotations

import os
from typing import List

from eval.golden_cases import GoldenCase

EVAL_IDS_ENV = "SMART_ASSIGNMENT_EVAL_IDS"


def select_cases(cases: List[GoldenCase]) -> List[GoldenCase]:
    """``cases``, or -- when EVAL_IDS_ENV names a comma-separated subset of
    eval_ids -- just those, in the order named.

    Raises ``ValueError`` naming any unknown id (plus the valid set) rather
    than silently running nothing or an unintended subset; callers decide how
    to surface that (test_eval.py lets pytest report it, capture.py's CLI
    turns it into a clean SystemExit)."""
    raw = os.environ.get(EVAL_IDS_ENV)
    if not raw or not raw.strip():
        return cases

    wanted = [eval_id.strip() for eval_id in raw.split(",") if eval_id.strip()]
    by_id = {case.eval_id: case for case in cases}
    missing = [eval_id for eval_id in wanted if eval_id not in by_id]
    if missing:
        raise ValueError(
            f"{EVAL_IDS_ENV} named unknown eval_id(s): {missing}. "
            f"Valid ids: {sorted(by_id)}"
        )
    return [by_id[eval_id] for eval_id in wanted]
