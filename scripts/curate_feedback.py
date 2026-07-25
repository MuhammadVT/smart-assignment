"""
Curate the human-feedback log into candidate eval cases (offline, human-driven).

Reads the append-only feedback log written by the ``feedback`` package and
distills HUMAN annotations into review-ready candidate eval cases -- the
"dataset curation" step of the feedback flywheel. It prints a summary and, with
``--out``, writes a JSON array a human can inspect and promote into
``eval/golden_cases.py``. It NEVER changes a decision or auto-edits the golden
set; promotion is a deliberate human step.

Run:
    python3 scripts/curate_feedback.py                          # summarize the default log
    python3 scripts/curate_feedback.py --only-negative          # just the failure signals
    python3 scripts/curate_feedback.py --out eval/data/feedback_candidates.json
    python3 scripts/curate_feedback.py --log path/to/annotations.jsonl
"""

from __future__ import annotations

import argparse

from smart_assignment.feedback.curate import curate_feedback, write_curation
from smart_assignment.shared.config import DEFAULT_CONFIG


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log",
        default=DEFAULT_CONFIG.feedback_log_path,
        help="Path to the feedback JSONL log (default: Config.feedback_log_path).",
    )
    parser.add_argument(
        "--only-negative",
        action="store_true",
        help="Keep only negative (failure) annotations -- the highest-value regression candidates.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Write curated candidate cases to this JSON path (for human review/promotion).",
    )
    args = parser.parse_args()

    cases = curate_feedback(args.log, only_negative=args.only_negative)
    negatives = sum(1 for c in cases if c.human_verdict == "negative")
    print(f"Read feedback log: {args.log}")
    print(f"Curated {len(cases)} candidate case(s) ({negatives} negative).")
    for case in cases:
        target = case.suggested_expected_outcome or "(human to decide)"
        print(
            f"  - {case.eval_id}: verdict={case.human_verdict} "
            f"observed={case.observed_outcome or '?'} -> suggested={target}"
        )
    if args.out:
        write_curation(cases, args.out)
        print(f"Wrote {len(cases)} candidate case(s) to {args.out} for review.")


if __name__ == "__main__":
    main()
