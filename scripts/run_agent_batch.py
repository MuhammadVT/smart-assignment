"""
Agent-driven batch runner for the Smart Assignment workflow.

Runs the REAL conversational agent architecture (``agent.build_batch_agent``)
unattended over a whole batch of prospects -- one non-interactive turn each --
and writes one JSONL result per prospect. It is the agent-based sibling of
``scripts/run_batch.py``: same source/sink/output contract, but the per-prospect
engine is the agent (its NL reasoning + the escalation-triage brief) instead of
the bare deterministic pipeline. A separate entry point, like every other script.

Run:
    python3 scripts/run_agent_batch.py --mock-geocoder            # demo prospects
    python3 scripts/run_agent_batch.py --source prospects.json --out results.jsonl

``--mock-geocoder`` forces the offline MockGeocoder for a demo. The agent needs
LLM credentials for the configured backend; without them (or on any agent
failure) the run degrades to the deterministic pipeline automatically, so it
never dead-ends. Grounded reasoning and triage honor the same config flags as the
chat mode.
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
        description="Run the Smart Assignment AGENT in batch over a set of prospects."
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
        default="agent_batch_results.jsonl",
        help="Output JSONL path (one result record per prospect). "
        "Default: agent_batch_results.jsonl",
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

    print(f"Agent batch complete: {summary.total} prospects -> {args.out}")
    print(
        f"  recommend={summary.recommend}  "
        f"escalate={summary.escalate}  "
        f"needs_attention={summary.needs_attention}"
    )


if __name__ == "__main__":
    main()
