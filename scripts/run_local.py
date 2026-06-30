"""
Manual local smoke-test entry point. Picks a workflow by name and runs it
against a sample customer profile. Useful for quick end-to-end checks
without the ADK Web UI or CLI.

Requires a Gemini API key (or Vertex AI credentials) to execute the LLM
node -- set GOOGLE_API_KEY in your environment, or see
https://google.github.io/adk-docs/get-started/google-cloud/ for Vertex
AI auth.

Run:
    python3 scripts/run_local.py --workflow slot_recommendation
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import time

from google.adk.runners import InMemoryRunner
from google.adk.sessions import InMemorySessionService

from smart_assignment.shared.models import CustomerProfile, DayOfWeek

APP_NAME = "smart_assignment"
USER_ID = "local-dev-user"


def _sample_customer() -> CustomerProfile:
    return CustomerProfile(
        customer_id="CUST-NEW-9001",
        name="Riverside Diner",
        address="123 Example St, Sample City, ST",
        latitude=37.77,
        longitude=-122.41,
        weekly_order_volume_cases=150,
        product_temp_zone="mixed",
        requested_days=[DayOfWeek.TUE, DayOfWeek.WED],
        requested_time_window=(time(7, 0), time(11, 0)),
        delivery_priority="standard",
    )


async def run_slot_recommendation():
    from smart_assignment.workflows.slot_recommendation import root_agent

    customer = _sample_customer()
    session_service = InMemorySessionService()
    session = await session_service.create_session(app_name=APP_NAME, user_id=USER_ID)
    runner = InMemoryRunner(agent=root_agent, app_name=APP_NAME)

    print(f"Running slot_recommendation workflow for {customer.name}...\n")

    events = runner.run_async(
        user_id=USER_ID,
        session_id=session.id,
        new_message=customer,
    )

    async for event in events:
        if event.is_final_response():
            print("FINAL OUTPUT:")
            print(event.content)


WORKFLOWS = {
    "slot_recommendation": run_slot_recommendation,
    # Add new entries here as additional workflows are added under
    # smart_assignment/workflows/.
}


def main():
    parser = argparse.ArgumentParser(description="Run a smart_assignment workflow locally")
    parser.add_argument(
        "--workflow",
        choices=list(WORKFLOWS.keys()),
        default="slot_recommendation",
        help="Which workflow to run",
    )
    args = parser.parse_args()
    asyncio.run(WORKFLOWS[args.workflow]())


if __name__ == "__main__":
    main()
