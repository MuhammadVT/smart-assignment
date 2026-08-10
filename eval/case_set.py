"""Which eval CASES a run scores -- declared, not hardcoded.

``eval/dataset.py`` made the *world* a declared input: eval picks its route
capacity and geocoding via ``SMART_ASSIGNMENT_EVAL_DATASET`` rather than
inheriting whatever a developer had set. The cases themselves stayed hardcoded
-- six modules imported ``GOLDEN_CASES`` directly -- so the other half of "what
did this run actually score?" was answerable only by reading the source.

That mattered beyond tidiness. ``eval/case_source.py`` and
``python3 -m eval.build_evalset --cases <file>`` already turn curated production
feedback into runnable ``GoldenCase`` objects, and those are the cases that carry
a real ``decision_id`` -- the only ones a human label can join to (see
``eval/judge_calibration.py``). But no test runner could be pointed at them, so
curated cases could be *built* and never *run*.

This module closes that, deliberately mirroring ``eval/dataset.py`` rather than
inventing a second convention:

* **Declared, not defaulted.** ``SMART_ASSIGNMENT_EVAL_CASES`` selects the set;
  unset means ``golden``, the built-in fixtures, which reproduces prior behavior
  exactly.
* **A value, not a code change.** Point it at a curated candidates JSON
  (``scripts/curate_feedback.py`` / ``scripts/phoenix_curate.py`` output) and
  every eval entry point follows. No call site edits, no new branch.
* **Loud, never a silent guess.** An unreadable path, an unparseable file, or a
  set that resolves to zero cases raises -- an empty case set would score nothing
  and still report success, the same silent green ``eval/inference_guard.py`` and
  ``eval/run_config.py`` exist to prevent.

Two things this deliberately does NOT change:

* ``eval/build_evalset.py``'s default stays ``GOLDEN_CASES``. The *committed*
  dataset must always be the golden one, whatever a shell variable says.
* ``tests/eval/test_dataset_lock.py`` stays golden-pinned. It guards a committed
  artifact, so it must not follow an ambient selection.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from eval.golden_cases import GOLDEN_CASES, GoldenCase

logger = logging.getLogger(__name__)

CASE_SET_ENV = "SMART_ASSIGNMENT_EVAL_CASES"
DEFAULT_CASE_SET = "golden"


@dataclass(frozen=True)
class CaseSet:
    """A resolved set of eval cases plus where it came from.

    ``kind`` distinguishes the *code-defined* built-in fixtures (``"code"``) from
    a *file-backed* curated set (``"file"``), the same code/snapshot split
    ``eval.dataset.EvalDataset`` draws. ``skipped`` carries the candidates a
    file-backed set could not replay (a PII-redacted address, say) so a caller
    can report them rather than silently scoring fewer cases than the file holds.
    """

    name: str
    kind: str
    cases: Tuple[GoldenCase, ...]
    path: Optional[str] = None
    skipped: Tuple[Dict[str, str], ...] = ()

    @property
    def is_default(self) -> bool:
        """True for the built-in golden fixtures -- the one set the committed
        artifacts under ``eval/data/`` correspond to."""
        return self.kind == "code" and self.name == DEFAULT_CASE_SET


# The code-defined case sets. File-backed sets are addressed by path, not
# registered here, because there is no fixed inventory of curated files.
_KNOWN_CASE_SETS: Dict[str, CaseSet] = {
    DEFAULT_CASE_SET: CaseSet(name=DEFAULT_CASE_SET, kind="code", cases=tuple(GOLDEN_CASES)),
}


def _load_file_case_set(raw: str) -> CaseSet:
    """A curated candidates JSON as a :class:`CaseSet`. Raises ``ValueError``
    naming the valid forms when the value is neither a known name nor a readable
    file -- a typo must not be mistaken for "some other case set"."""
    if not os.path.isfile(raw):
        raise ValueError(
            f"{CASE_SET_ENV}={raw!r} is neither a known case set nor a readable file. "
            f"Valid: {sorted(_KNOWN_CASE_SETS)}, or a path to a curated candidates JSON "
            "(the output of scripts/curate_feedback.py or scripts/phoenix_curate.py)."
        )
    # Lazy: only a file-backed set needs the loader, and this module is imported
    # by the hermetic suite.
    from eval.case_source import load_curated_cases

    try:
        cases, skipped = load_curated_cases(raw)
    except Exception as exc:  # noqa: BLE001 - re-raised with the variable named
        raise ValueError(f"{CASE_SET_ENV}={raw!r} could not be loaded: {exc}") from exc

    if not cases:
        detail = f" ({len(skipped)} candidate(s) skipped: see the report)" if skipped else ""
        raise ValueError(
            f"{CASE_SET_ENV}={raw!r} produced no replayable eval cases{detail}. "
            "Scoring zero cases would report success over nothing; fix or re-curate the file."
        )
    return CaseSet(
        name=f"file:{os.path.basename(raw)}",
        kind="file",
        cases=tuple(cases),
        path=raw,
        skipped=tuple(skipped),
    )


def resolve_case_set() -> CaseSet:
    """The declared case set (``SMART_ASSIGNMENT_EVAL_CASES``, default
    ``golden``).

    A non-default selection is logged at warning level, naming the set and its
    size, for the same reason ``case_selection.select_cases`` warns when it
    narrows: what a run scored must never be invisible.
    """
    raw = (os.environ.get(CASE_SET_ENV) or "").strip()
    if not raw:
        return _KNOWN_CASE_SETS[DEFAULT_CASE_SET]

    known = _KNOWN_CASE_SETS.get(raw)
    case_set = known if known is not None else _load_file_case_set(raw)

    if not case_set.is_default:
        logger.warning(
            "%s is set: scoring case set %r (%d case(s)) instead of the %d built-in "
            "golden fixtures.",
            CASE_SET_ENV,
            case_set.name,
            len(case_set.cases),
            len(GOLDEN_CASES),
        )
        for entry in case_set.skipped:
            logger.warning("  skipped candidate %s: %s", entry.get("eval_id"), entry.get("reason"))
    return case_set


def resolve_cases() -> List[GoldenCase]:
    """Just the cases from :func:`resolve_case_set`, as a list -- the common call
    at a runner that does not care where they came from."""
    return list(resolve_case_set().cases)
