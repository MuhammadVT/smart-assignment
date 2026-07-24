"""
Judge calibration (advisory) -- report how well the automated LLM judges agree
with the human labels, so you know which judges to trust before gating on them.

Purely advisory and gated by ``Config.use_judge_calibration`` (default off): with
the flag off this is a no-op, so calibration never runs unless explicitly turned
on. It reads the vendor-free feedback log for human labels and a precomputed
judge-verdicts JSON, and prints (and optionally writes) a per-dimension +
composite agreement report -- Cohen's kappa, a dangerous-cell rate, and a trust
band. No decision is changed; nothing is gated.

Verdicts file shape (produced by running the judges over the same decisions):
    {"<decision_id>": {"response_clarity": {"passed": true, "score": 0.9}, ...}}

Run:
    SMART_ASSIGNMENT_USE_JUDGE_CALIBRATION=true python3 scripts/calibrate_judges.py \\
        --verdicts eval/data/judge_verdicts.json
    ... --log feedback_data/annotations.jsonl --out eval/data/calibration.json --min-n 20
"""

from __future__ import annotations

import argparse
import json

from eval.judge_calibration import (
    calibrate,
    human_labels_from_feedback,
    verdicts_from_mapping,
)
from smart_assignment.shared.config import DEFAULT_CONFIG


def _print_report(report: dict) -> None:
    totals = report["totals"]
    print(
        f"human labels: {totals['human_labels']}  judge verdicts: {totals['judge_verdicts']}  "
        f"aligned pairs: {totals['aligned_pairs']}"
    )
    rows = list(report["dimensions"].items()) + [("composite", report["composite"])]
    print(f"{'dimension':<22}{'n':>4}  {'kappa':>7}  {'danger':>7}  trust")
    for name, r in rows:
        kappa = "n/a" if r["cohen_kappa"] is None else f"{r['cohen_kappa']:.2f}"
        danger = "n/a" if r["dangerous_cell_rate"] is None else f"{r['dangerous_cell_rate']:.0%}"
        print(f"{name:<22}{r['n']:>4}  {kappa:>7}  {danger:>7}  {r['trust']}")
    # Surface the dangerous disagreements for the most-used judges.
    for name, r in rows:
        if r["top_disagreements"]:
            print(f"\n{name} — top disagreements (judge vs human):")
            for d in r["top_disagreements"][:5]:
                note = f" — “{d['note']}”" if d.get("note") else ""
                print(f"  [{d['source']}] {d['kind']} {d['decision_id']}{note}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log",
        default=DEFAULT_CONFIG.feedback_log_path,
        help="Feedback JSONL log with the human labels (default: Config.feedback_log_path).",
    )
    parser.add_argument(
        "--verdicts",
        required=True,
        help="Precomputed judge-verdicts JSON: {decision_id: {dimension: {passed, score}}}.",
    )
    parser.add_argument("--out", default=None, help="Write the full report JSON here.")
    parser.add_argument(
        "--min-n",
        type=int,
        default=20,
        help="Minimum aligned pairs before a trust band is reported (else 'insufficient').",
    )
    args = parser.parse_args()

    if not DEFAULT_CONFIG.use_judge_calibration:
        print(
            "Judge calibration is off (advisory). Set "
            "SMART_ASSIGNMENT_USE_JUDGE_CALIBRATION=true to run it."
        )
        return

    labels = human_labels_from_feedback(args.log)
    with open(args.verdicts, "r", encoding="utf-8") as handle:
        verdicts = verdicts_from_mapping(json.load(handle))

    report = calibrate(labels, verdicts, min_n=args.min_n)
    _print_report(report)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        print(f"\nWrote full report to {args.out}")


if __name__ == "__main__":
    main()
