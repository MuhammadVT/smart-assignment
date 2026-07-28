"""
Batch (non-conversational) runner for the Smart Assignment workflow.

Runs the deterministic pipeline once per prospect over a whole batch and writes
one JSONL result per prospect -- the production shape where prospects flow
Salesforce -> Smart Assignment -> the sales-consultant Customer View, with no
human in the loop. The conversational chat mode (`adk web`, the web app) is
unaffected; this is a separate entry point, exactly like `scripts/run_local.py`
and `scripts/run_web.py`.

Run:
    python3 scripts/run_batch.py --mock-geocoder            # built-in demo prospects, offline
    python3 scripts/run_batch.py --source prospects.json --out results.jsonl

`--mock-geocoder` forces the offline MockGeocoder (deterministic, no network) for
a demo; without it the same geocoder every other surface uses is resolved from
SMART_ASSIGNMENT_GEOCODER. Grounded reasoning and triage honor the same config
flags as the chat mode; with no credentials they fall back to the deterministic
result, so the demo run works fully offline.
"""

from __future__ import annotations

import argparse

from smart_assignment.batch import BatchRunner, JsonlResultSink, MockProspectSource
from smart_assignment.integrations.geocoding_client import MockGeocoder


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Smart Assignment workflow in batch over a set of prospects."
    )
    parser.add_argument(
        "--source",
        help=(
            "JSON file of prospect intake records (a list of objects with "
            "address / order_quantity_cases / optional preferred_day+window / "
            "optional prospect_id). Default: the built-in SAMPLE_CUSTOMERS."
        ),
    )
    parser.add_argument(
        "--out",
        default="batch_results.jsonl",
        help="Output JSONL path (one result record per prospect). Default: batch_results.jsonl",
    )
    parser.add_argument(
        "--mock-geocoder",
        action="store_true",
        help="Force the offline MockGeocoder (deterministic, no network) for a demo run.",
    )
    args = parser.parse_args()

    source = (
        MockProspectSource.from_json(args.source)
        if args.source
        else MockProspectSource.from_samples()
    )
    geocoder = MockGeocoder() if args.mock_geocoder else None

    with JsonlResultSink(args.out) as sink:
        summary = BatchRunner(source, sink, geocoder=geocoder).run()

    print(f"Batch complete: {summary.total} prospects -> {args.out}")
    print(
        f"  recommend={summary.recommend}  "
        f"escalate={summary.escalate}  "
        f"needs_attention={summary.needs_attention}"
    )


if __name__ == "__main__":
    main()
