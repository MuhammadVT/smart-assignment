"""
Score the current model against a self-contained snapshot dataset.

This is the "run the current model against the curated dataset" step of the
flywheel: for every case in a snapshot bundle, re-run the pipeline against the
bundle's own frozen world and check the decision against the golden target --
the recommend/escalate **outcome** and, on a recommend, the **route-slot**
(recommended route id + window). Because a snapshot dataset carries its own world
(see integrations/snapshot_data.py), this runs fully offline and deterministically
with no live data source -- exactly like scoring against ``mock``.

**Two model paths, one easy toggle.** ``path`` selects how the decision is made:

* ``"deterministic"`` -- the weighted-sum decision (grounded judgment off). Fully
  offline, no credentials; this is the self-contained CI gate.
* ``"llm"`` -- grounded LLM judgment on. Needs an LLM backend + credentials;
  without them the pipeline transparently falls back to the deterministic result,
  so it still runs (just without the LLM in the loop).

Pick it with the ``path`` argument or ``SMART_ASSIGNMENT_EVAL_MODEL_PATH``.

The scorer is side-effect-free: it saves and restores the data-source/geocoder
env it pins, so running it never leaks a pinned snapshot into the rest of a
process (a test suite, a REPL).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional

from eval.dataset import EvalDataset, all_datasets, pin_dataset

logger = logging.getLogger(__name__)

EVAL_MODEL_PATH_ENV = "SMART_ASSIGNMENT_EVAL_MODEL_PATH"
PATH_DETERMINISTIC = "deterministic"
PATH_LLM = "llm"

# The env vars pin_dataset mutates -- saved/restored so scoring is side-effect-free.
_PINNED_ENV = (
    "SMART_ASSIGNMENT_DATA_SOURCE",
    "SMART_ASSIGNMENT_GEOCODER",
    "SMART_ASSIGNMENT_DATA_SOURCE_STRICT",
    "SMART_ASSIGNMENT_SNAPSHOT_DIR",
)


@dataclass
class CaseScore:
    """One case's decision vs. its golden target. A ``*_match`` is ``None`` when
    the dimension doesn't apply (no route/window is expected on an escalate)."""

    eval_id: str
    expected_outcome: str
    got_outcome: str
    outcome_match: bool
    expected_route_id: Optional[str] = None
    got_route_id: Optional[str] = None
    route_match: Optional[bool] = None
    expected_window: Optional[str] = None
    got_window: Optional[str] = None
    window_match: Optional[bool] = None
    route_slot_match: Optional[bool] = None
    error: Optional[str] = None


@dataclass
class DatasetScore:
    dataset: str
    path: str
    cases: List[CaseScore] = field(default_factory=list)

    @property
    def outcome_total(self) -> int:
        return len(self.cases)

    @property
    def outcome_pass(self) -> int:
        return sum(1 for c in self.cases if c.outcome_match)

    @property
    def route_slot_total(self) -> int:
        return sum(1 for c in self.cases if c.route_slot_match is not None)

    @property
    def route_slot_pass(self) -> int:
        return sum(1 for c in self.cases if c.route_slot_match)

    def failures(self) -> List[CaseScore]:
        return [
            c
            for c in self.cases
            if not c.outcome_match or c.route_slot_match is False or c.error
        ]

    def all_pass(self) -> bool:
        return not self.failures()

    def summary(self) -> str:
        return (
            f"[{self.dataset} / {self.path}] outcome {self.outcome_pass}/{self.outcome_total}, "
            f"route-slot {self.route_slot_pass}/{self.route_slot_total}"
        )


def config_for_path(path: str, base=None):
    """A ``Config`` for the requested model path (a copy of ``base`` / env config
    with the decision-strategy flags set), so the same scorer runs either path."""
    from dataclasses import replace

    from smart_assignment.shared.config import Config

    base = base or Config.from_env()
    if path == PATH_LLM:
        return replace(base, use_grounded_judgment=True)
    # Deterministic: the weighted-sum decision, grounded off -- fully offline.
    return replace(base, use_grounded_judgment=False)


def _score_case(case_dict, config) -> CaseScore:
    from eval.case_source import SkippedCase, candidate_to_case
    from smart_assignment.pipeline import run_slot_recommendation
    from smart_assignment.reasoning import DeterministicReasoner
    from smart_assignment.shared.models import Decision

    eval_id = str(case_dict.get("eval_id", "?"))
    expected_outcome = str(case_dict.get("expected_outcome", "")).lower()
    expected_route = case_dict.get("expected_route_id")
    expected_window = case_dict.get("expected_window")

    try:
        case = candidate_to_case(case_dict)
    except SkippedCase as exc:
        return CaseScore(eval_id, expected_outcome, "", False, error=str(exc))

    # Deterministic reasoner keeps the run offline; the decision strategy (weighted
    # vs grounded) is what `config` selects.
    result = run_slot_recommendation(case.customer, config=config, reasoner=DeterministicReasoner())
    rec = result.recommendation
    got_outcome = "recommend" if rec.decision == Decision.RECOMMENDED else "escalate"
    got_route = rec.recommended_route_id
    got_window = rec.recommended_window

    score = CaseScore(
        eval_id=eval_id,
        expected_outcome=expected_outcome,
        got_outcome=got_outcome,
        outcome_match=(got_outcome == expected_outcome),
        expected_route_id=expected_route,
        got_route_id=got_route,
        expected_window=expected_window,
        got_window=got_window,
    )
    # Route / window only apply when the golden target specifies them (a recommend).
    if expected_route is not None:
        score.route_match = got_route == expected_route
    if expected_window is not None:
        score.window_match = got_window == expected_window
    if expected_route is not None:
        # A route-slot match requires the route AND (when specified) the window.
        score.route_slot_match = bool(score.route_match) and (
            score.window_match is not False
        )
    return score


def score_dataset(
    dataset: EvalDataset, *, path: str = PATH_DETERMINISTIC, config=None
) -> DatasetScore:
    """Run the current model over every case in ``dataset`` and score the outcome
    and route-slot against the golden targets. Side-effect-free: restores the
    pinned env afterwards."""
    from smart_assignment.integrations import snapshot_data
    from smart_assignment.integrations.route_capacity_client import clear_route_cache

    if not dataset.path:
        raise ValueError(f"Dataset {dataset.name!r} is not a snapshot (no bundle path).")

    saved = {key: os.environ.get(key) for key in _PINNED_ENV}
    run_config = config_for_path(path, config)
    result = DatasetScore(dataset=dataset.name, path=path)
    try:
        pin_dataset(dataset)
        for case_dict in snapshot_data.load_cases(dataset.path):
            result.cases.append(_score_case(case_dict, run_config))
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        clear_route_cache()
    return result


def score_all_snapshots(*, path: str = PATH_DETERMINISTIC) -> List[DatasetScore]:
    """Score every discovered snapshot dataset (skips the code-defined ``mock``)."""
    return [
        score_dataset(dataset, path=path)
        for dataset in all_datasets().values()
        if dataset.kind == "snapshot"
    ]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=None, help="Snapshot dataset name (default: all).")
    parser.add_argument(
        "--path",
        default=os.environ.get(EVAL_MODEL_PATH_ENV, PATH_DETERMINISTIC),
        choices=[PATH_DETERMINISTIC, PATH_LLM],
        help="Decision path (default: deterministic, or SMART_ASSIGNMENT_EVAL_MODEL_PATH).",
    )
    args = parser.parse_args()

    if args.dataset:
        datasets = [all_datasets()[args.dataset]]
    else:
        datasets = [d for d in all_datasets().values() if d.kind == "snapshot"]

    ok = True
    for dataset in datasets:
        score = score_dataset(dataset, path=args.path)
        print(score.summary())
        for failure in score.failures():
            print(
                f"  FAIL {failure.eval_id}: expected {failure.expected_outcome}"
                f"/{failure.expected_route_id}/{failure.expected_window} got "
                f"{failure.got_outcome}/{failure.got_route_id}/{failure.got_window}"
                + (f" ({failure.error})" if failure.error else "")
            )
        ok = ok and score.all_pass()

    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
