"""
Prospect sources for batch mode.

A ``ProspectSource`` yields the prospects a batch run should process. In
production the addresses come from Salesforce/CRM (a ``SalesforceProspectSource``
implementing this same one-method contract); for local runs and CI a
``MockProspectSource`` serves the code-defined ``SAMPLE_CUSTOMERS`` or a JSON file
-- the batch analogue of ``MockGeocoder``. Because the runner depends only on the
``ProspectSource`` protocol, swapping in the real Salesforce adapter changes
nothing downstream.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Protocol, Union

from smart_assignment.mock_customers import SAMPLE_CUSTOMERS
from smart_assignment.shared.models import (
    PROSPECT_PLACEHOLDER_NAME,
    CustomerProfile,
    DayOfWeek,
    PreferredSlot,
)
from smart_assignment.shared.timeutils import parse_time


@dataclass(frozen=True)
class Prospect:
    """One unit of batch work: a CRM-sourced identifier paired with the intake
    profile the pipeline consumes.

    The ``prospect_id`` is kept SEPARATE from the profile because a prospect has a
    Salesforce id but usually no Sysco ``customer_number`` yet -- the id is how the
    result is later keyed for the Customer View, not a value the pipeline reasons
    over."""

    prospect_id: str
    profile: CustomerProfile


class ProspectSource(Protocol):
    """A source of prospects to run. The real Salesforce adapter implements this
    exact contract, so nothing downstream of it changes."""

    def prospects(self) -> Iterable[Prospect]:
        ...


class MockProspectSource:
    """An offline, deterministic source for local runs and CI. Build it from the
    code-defined ``SAMPLE_CUSTOMERS`` or from a JSON file of intake records."""

    def __init__(self, prospects: list[Prospect]) -> None:
        self._prospects = list(prospects)

    def prospects(self) -> Iterator[Prospect]:
        return iter(self._prospects)

    @classmethod
    def from_samples(cls) -> "MockProspectSource":
        """The built-in demo prospects, each assigned a stable synthetic id."""
        return cls(
            [
                Prospect(prospect_id=f"MOCK-{i:03d}", profile=profile)
                for i, profile in enumerate(SAMPLE_CUSTOMERS, start=1)
            ]
        )

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "MockProspectSource":
        """Load prospects from a JSON list of intake records (see
        :func:`_profile_from_dict` for the accepted fields)."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls([_prospect_from_dict(i, record) for i, record in enumerate(data, start=1)])


def _prospect_from_dict(index: int, record: dict) -> Prospect:
    return Prospect(
        prospect_id=str(record.get("prospect_id") or f"ROW-{index:03d}"),
        profile=_profile_from_dict(record),
    )


def _profile_from_dict(record: dict) -> CustomerProfile:
    """Build a ``CustomerProfile`` from a plain intake dict. A preferred slot needs
    all three of ``preferred_day`` / ``preferred_window_start`` /
    ``preferred_window_end`` -- a partial one is treated as no preference (the
    deterministic pipeline's own rule), never guessed at."""
    slot = None
    day = record.get("preferred_day")
    start = record.get("preferred_window_start")
    end = record.get("preferred_window_end")
    if day and start and end:
        slot = PreferredSlot(
            DayOfWeek(str(day).strip().upper()),
            (parse_time(start), parse_time(end)),
        )
    return CustomerProfile(
        name=record.get("name") or PROSPECT_PLACEHOLDER_NAME,
        address=record.get("address", ""),
        order_quantity_cases=int(record.get("order_quantity_cases") or 0),
        customer_number=record.get("customer_number"),
        preferred_slot=slot,
    )
