"""
Shared ``SMART_ASSIGNMENT_EVAL_IDS`` handling for the eval TEST runners
(``eval/test_eval.py``, ``eval/test_quality.py``,
``eval/test_rationale_faithfulness.py``).

``SMART_ASSIGNMENT_EVAL_IDS`` is a LOCAL, shell-only cost-control knob: it lets a
developer run a comma-separated subset of ``eval/golden_cases.py``'s
``GOLDEN_CASES`` to cut live-LLM cost while iterating. Two guardrails keep it from
silently doing the wrong thing where a subset is dangerous:

* **It must never narrow a CI run.** CI scores the full golden set, so
  ``select_cases`` raises if the var is set while CI is active (GitHub Actions
  sets ``CI=true``) -- a stray ``.env`` or repo/org var can't quietly shrink
  coverage.
* **It is NOT how ``eval/capture.py`` picks cases.** Capture *writes* the
  committed dataset, so it takes an explicit ``--ids`` flag via
  ``filter_cases_by_ids`` and does not read this env var -- a value left in
  ``.env`` (which ``load_dotenv`` pulls into the environment) can never silently
  capture a partial dataset.

A narrowing run is always announced loudly (a warning naming running vs. full),
so a subset is never invisible.
"""

from __future__ import annotations

import logging
import os
from typing import List, Sequence

from eval.golden_cases import GoldenCase

logger = logging.getLogger(__name__)

EVAL_IDS_ENV = "SMART_ASSIGNMENT_EVAL_IDS"
# GitHub Actions (and most CI systems) set CI=true. Read ONLY to reject a subset
# filter in CI -- never to change what runs otherwise.
_CI_ENV = "CI"


def parse_eval_ids(raw: str) -> List[str]:
    """Split a comma-separated eval_id string into a clean list (blanks dropped)."""
    return [eval_id.strip() for eval_id in raw.split(",") if eval_id.strip()]


def filter_cases_by_ids(
    cases: Sequence[GoldenCase], ids: Sequence[str], *, source: str = "eval_id subset"
) -> List[GoldenCase]:
    """``cases`` restricted to ``ids``, in the order named. Raises ``ValueError``
    -- naming ``source`` and the unknown id(s) plus the valid set -- rather than
    silently running an unintended subset.

    The EXPLICIT-subset primitive: it reads no environment, so a *write*
    operation (``eval/capture.py``) takes its subset only from an argument it was
    handed, never from an ambient ``SMART_ASSIGNMENT_EVAL_IDS``."""
    by_id = {case.eval_id: case for case in cases}
    missing = [eval_id for eval_id in ids if eval_id not in by_id]
    if missing:
        raise ValueError(
            f"{source} names unknown eval_id(s): {missing}. Valid ids: {sorted(by_id)}"
        )
    return [by_id[eval_id] for eval_id in ids]


def _ci_active() -> bool:
    return os.environ.get(_CI_ENV, "").strip().lower() in ("1", "true", "yes", "on")


def select_cases(cases: Sequence[GoldenCase]) -> List[GoldenCase]:
    """``cases``, or -- when ``SMART_ASSIGNMENT_EVAL_IDS`` names a comma-separated
    subset -- just those, in the order named. For the eval TEST runners only.

    - Unset/blank: every case, unchanged (the knob only ever opts IN).
    - Set under CI (``CI`` truthy): ``ValueError`` -- CI must score the full
      golden set; this local-only knob can't silently shrink it.
    - Set locally: the named subset, with a loud warning naming running vs. full.

    Unknown ids raise ``ValueError`` (see ``filter_cases_by_ids``)."""
    raw = os.environ.get(EVAL_IDS_ENV)
    if not raw or not raw.strip():
        return list(cases)

    if _ci_active():
        raise ValueError(
            f"{EVAL_IDS_ENV} is set ({raw!r}) but this is a CI run ({_CI_ENV} is set). "
            f"{EVAL_IDS_ENV} is a local cost-control knob and must not narrow CI, which "
            "scores the full golden set -- unset it in the CI environment."
        )

    selected = filter_cases_by_ids(cases, parse_eval_ids(raw), source=EVAL_IDS_ENV)
    logger.warning(
        "%s is set: running %d/%d golden case(s) [%s]. Local cost-control subset; "
        "CI scores all cases.",
        EVAL_IDS_ENV,
        len(selected),
        len(list(cases)),
        ", ".join(case.eval_id for case in selected),
    )
    return selected
