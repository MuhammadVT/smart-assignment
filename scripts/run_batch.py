"""
Batch runner for the Smart Assignment workflow.

Runs the REAL agent architecture (``agent.build_batch_agent``) unattended over a
whole batch of prospects -- one non-interactive turn each -- and writes one JSONL
result per prospect, the production shape where prospects flow Salesforce ->
Smart Assignment -> the sales-consultant Customer View with no human in the loop.
The per-prospect engine is the agent (its NL reasoning + the escalation-triage
brief); the deterministic pipeline is the floor it falls back to. A separate entry
point, like ``scripts/run_local.py`` and ``scripts/run_web.py``.

Run:
    python3 scripts/run_batch.py --mock-geocoder            # demo prospects, offline
    python3 scripts/run_batch.py --source prospects.json --out results.jsonl

``--mock-geocoder`` forces the offline MockGeocoder for a demo. The agent needs
LLM credentials for the configured backend; without them (or on any agent
failure) the run degrades to the deterministic pipeline automatically, so it never
dead-ends and still runs fully offline. Grounded reasoning and triage honor the
same config flags as the chat mode.
"""

from __future__ import annotations

import argparse
import asyncio

from smart_assignment.batch import AgentBatchRunner, JsonlResultSink, MockProspectSource
from smart_assignment.integrations.geocoding_client import MockGeocoder


async def _run(source, sink, geocoder):
    return await AgentBatchRunner(source, sink, geocoder=geocoder).run()


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
        summary = asyncio.run(_run(source, sink, geocoder))

    print(f"Batch complete: {summary.total} prospects -> {args.out}")
    print(
        f"  recommend={summary.recommend}  "
        f"escalate={summary.escalate}  "
        f"needs_attention={summary.needs_attention}"
    )


if __name__ == "__main__":
    main()
