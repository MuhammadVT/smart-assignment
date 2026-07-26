"""
The headless decision service (`smart_assignment.service`).

Two things these tests exist to protect:

1. **It adds no decision logic.** `assign` must return exactly what
   `pipeline.run_slot_recommendation` produced for the same prospect, and must
   hand the caller's `Config` to it untouched. If that ever diverges, Mode 3 has
   become a second pipeline -- the thing this design is meant to avoid.
2. **A bad record is a bad row, not a stopped run.** A production batch has to
   survive an unusable prospect, an unresolvable address, and an outright bug,
   and still decide everything else.

**Why almost every call here pins `_DETERMINISTIC`.** Step 5 *samples* when
grounded reasoning is on, so with real credentials present two runs of the same
prospect may legitimately reach different decisions. A test that asserted a
specific outcome, or compared two runs, would then be asserting something about
the model rather than about this module -- passing or failing depending on whether
an API key happens to be configured. Pinning the deterministic path keeps these
tests about the service, and keeps them offline and fast. The grounded flags are
covered where they belong: by asserting they reach the pipeline unchanged.
"""

from __future__ import annotations

import asyncio
import json
from copy import deepcopy

import pytest

from smart_assignment import service
from smart_assignment.mock_customers import SAMPLE_CUSTOMERS
from smart_assignment.pipeline import run_slot_recommendation
from smart_assignment.shared.config import Config
from smart_assignment.shared.geo import (
    AddressNotFoundError,
    GeocodingServiceError,
)
from smart_assignment.shared.llm import _HOST_EVENT_LOOP
from smart_assignment.shared.models import CustomerProfile, DayOfWeek, GeoPoint

_ADDRESS = "1200 McKinney St, Houston, TX 77010"

# Step 5 with no model in the loop: reproducible, offline, and independent of
# whether the machine running the suite has credentials configured.
_DETERMINISTIC = Config(
    use_grounded_route_slot_pick=False,
    use_grounded_route_slot_escalation=False,
)


def _prospect(**overrides) -> CustomerProfile:
    fields = {
        "name": "Bayou City Bistro",
        "address": _ADDRESS,
        "order_quantity_cases": 90,
    }
    fields.update(overrides)
    return CustomerProfile(**fields)


class _BrokenGeocoder:
    """A geocoder that always fails, with whichever error the test needs."""

    def __init__(self, exc: Exception):
        self._exc = exc

    def geocode(self, address: str) -> GeoPoint:
        raise self._exc


# --- 1. the service adds nothing to the decision -----------------------------


@pytest.mark.parametrize(
    "name, expected_kind",
    [
        ("Bayou City Bistro", "RECOMMENDED"),
        ("Galleria Grill & Catering", "ESCALATED_LOW_SCORE"),
        ("Katy Prairie Steakhouse", "ESCALATED_NO_FEASIBLE_SLOT"),
    ],
)
def test_assign_returns_exactly_what_the_pipeline_decided(name, expected_kind):
    """Same input, same decision -- across all three decision kinds.

    Uses the bundled prospects, which were designed against the mock world to
    exercise the full outcome range (see mock_customers.py), rather than invented
    case counts whose outcome would drift with the scoring weights."""
    customer = next(c for c in SAMPLE_CUSTOMERS if c.name == name)

    direct = run_slot_recommendation(deepcopy(customer), config=_DETERMINISTIC)
    outcome = service.assign(deepcopy(customer), config=_DETERMINISTIC)

    assert outcome.ok
    assert direct.recommendation.decision.value == expected_kind
    assert outcome.decision == direct.recommendation.to_state_dict()


@pytest.mark.parametrize("pick", [False, True])
@pytest.mark.parametrize("escalation", [False, True])
def test_every_step5_config_reaches_the_pipeline_untouched(pick, escalation, monkeypatch):
    """Mode 3 is config-identical to Mode 2 at step 5, which only holds if the
    caller's Config arrives intact. Asserted over all four grounded combinations;
    the decision itself is run deterministically so this stays offline."""
    seen = {}
    config = Config(
        use_grounded_route_slot_pick=pick,
        use_grounded_route_slot_escalation=escalation,
    )

    def _spy(customer, routes=None, config=None, geocoder=None, recommendation=None):
        seen["config"] = config
        return run_slot_recommendation(customer, routes=routes, config=_DETERMINISTIC)

    monkeypatch.setattr(service, "run_slot_recommendation", _spy)
    service.assign(_prospect(), config=config)

    assert seen["config"] is config
    assert seen["config"].use_grounded_route_slot_pick is pick
    assert seen["config"].use_grounded_route_slot_escalation is escalation


def test_infeasible_candidates_carry_no_merit_score():
    """A rejected route must never show a score: hard constraints are absolute,
    and a number beside a rejection invites "it scored well, why wasn't it
    used?" (see pipeline._apply_route_slot_scores)."""
    outcome = service.assign(_prospect(order_quantity_cases=260), config=_DETERMINISTIC)
    assert outcome.ok
    rejected = [c for c in outcome.candidates if not c["feasible"]]
    assert rejected, "expected at least one infeasible candidate for this prospect"
    for candidate in rejected:
        assert "total_score" not in candidate
        assert "factor_scores" not in candidate


# --- 2. failures are rows, not exceptions ------------------------------------


def test_invalid_intake_returns_a_failure_rather_than_raising():
    outcome = service.assign(_prospect(address="   "), config=_DETERMINISTIC)
    assert outcome.ok is False
    assert outcome.error_kind == service.ERROR_INTAKE
    assert outcome.decision is None
    assert outcome.requires_human_review is True


def test_unresolvable_address_is_reported_not_repaired():
    """There is no user here to confirm a correction with, so an address that
    doesn't resolve is reported for a human to fix upstream."""
    geocoder = _BrokenGeocoder(AddressNotFoundError(_ADDRESS, "no match"))
    outcome = service.assign(_prospect(), config=_DETERMINISTIC, geocoder=geocoder)
    assert outcome.ok is False
    assert outcome.error_kind == service.ERROR_ADDRESS_NOT_FOUND


def test_geocoder_outage_is_distinguishable_from_a_bad_address():
    """Different kinds: a caller may retry a transport failure, but retrying an
    address that simply doesn't exist will never help."""
    geocoder = _BrokenGeocoder(GeocodingServiceError(_ADDRESS, "connection refused"))
    outcome = service.assign(_prospect(), config=_DETERMINISTIC, geocoder=geocoder)
    assert outcome.ok is False
    assert outcome.error_kind == service.ERROR_GEOCODER_UNAVAILABLE


def test_unexpected_error_is_contained_and_logged(caplog):
    geocoder = _BrokenGeocoder(RuntimeError("kaboom"))
    outcome = service.assign(_prospect(), config=_DETERMINISTIC, geocoder=geocoder)
    assert outcome.ok is False
    assert outcome.error_kind == service.ERROR_INTERNAL
    assert "kaboom" in (outcome.error or "")
    # A real bug must still be loud in the logs even though the row survived.
    assert any(r.levelname == "ERROR" for r in caplog.records)


# --- 3. batches --------------------------------------------------------------


def test_batch_continues_past_a_bad_record_and_keeps_input_order():
    batch = [
        _prospect(name="Good One"),
        _prospect(name="No Address", address=""),
        _prospect(name="Zero Cases", order_quantity_cases=0),
        _prospect(name="Good Two", order_quantity_cases=120),
    ]
    outcomes = service.assign_many(batch, config=_DETERMINISTIC)

    assert len(outcomes) == 4
    assert [o.ok for o in outcomes] == [True, False, False, True]
    assert [o.customer["name"] for o in outcomes] == [
        "Good One",
        "No Address",
        "Zero Cases",
        "Good Two",
    ]


def test_batch_fetches_the_route_world_once(monkeypatch):
    calls = {"n": 0}
    real = service.fetch_candidate_routes

    def _counting():
        calls["n"] += 1
        return real()

    monkeypatch.setattr(service, "fetch_candidate_routes", _counting)
    service.assign_many([_prospect(), _prospect(), _prospect()], config=_DETERMINISTIC)
    assert calls["n"] == 1


def test_batching_does_not_change_any_decision():
    """Sharing the route world and one event loop across a batch is an
    efficiency, not a behaviour change."""
    batch = [_prospect(), _prospect(order_quantity_cases=400)]
    batched = service.assign_many(deepcopy(batch), config=_DETERMINISTIC)
    singly = [service.assign(deepcopy(p), config=_DETERMINISTIC) for p in batch]
    assert [o.decision for o in batched] == [o.decision for o in singly]


# --- 4. the input adapter ----------------------------------------------------


def test_record_maps_the_expected_fields_and_ignores_unknown_ones():
    profile = service.from_salesforce_record(
        {
            "address": _ADDRESS,
            "order_quantity_cases": "90",  # CRMs often hand over strings
            "name": "Bayou City Bistro",
            "customer_number": "067-100001",
            "preferred_day": "tue",  # case-insensitive
            "preferred_window_start": "07:00",
            "preferred_window_end": "10:00",
            "sf_opportunity_id": "006XX000004TmiQ",  # unknown -> ignored
        }
    )
    assert profile.address == _ADDRESS
    assert profile.order_quantity_cases == 90
    assert profile.preferred_slot is not None
    assert profile.preferred_slot.day is DayOfWeek.TUE


def test_record_without_a_preference_is_fine():
    profile = service.from_salesforce_record(
        {"address": _ADDRESS, "order_quantity_cases": 90}
    )
    assert profile.preferred_slot is None
    assert profile.name == "New prospect"


@pytest.mark.parametrize(
    "record, expected",
    [
        ({"order_quantity_cases": 90}, "address"),
        ({"address": _ADDRESS}, "order_quantity_cases"),
        ({"address": _ADDRESS, "order_quantity_cases": "lots"}, "whole number"),
        ({"address": _ADDRESS, "order_quantity_cases": 90, "preferred_day": "TUE"}, "all three"),
        (
            {
                "address": _ADDRESS,
                "order_quantity_cases": 90,
                "preferred_day": "FUNDAY",
                "preferred_window_start": "07:00",
                "preferred_window_end": "10:00",
            },
            "preferred_day must be one of",
        ),
        (
            {
                "address": _ADDRESS,
                "order_quantity_cases": 90,
                "preferred_day": "TUE",
                "preferred_window_start": "breakfast",
                "preferred_window_end": "10:00",
            },
            "HH:MM",
        ),
    ],
)
def test_malformed_records_name_the_offending_field(record, expected):
    """The message has to say what to fix -- it is what an operator sees on a
    rejected row."""
    with pytest.raises(ValueError, match=expected):
        service.from_salesforce_record(record)


# --- 5. the wire form --------------------------------------------------------


def test_to_dict_is_json_safe_and_omits_the_in_process_trace():
    outcome = service.assign(_prospect(), config=_DETERMINISTIC)
    payload = outcome.to_dict()

    json.dumps(payload)  # raises if anything is unserializable
    assert "result" not in payload
    assert outcome.result is not None, "the trace is still available in-process"
    assert payload["decision"]["decision"] == outcome.result.recommendation.decision.value


def test_failure_wire_form_carries_the_reason_not_an_empty_decision():
    payload = service.assign(_prospect(address=""), config=_DETERMINISTIC).to_dict()
    assert payload["ok"] is False
    assert payload["requires_human_review"] is True
    assert payload["error_kind"] == service.ERROR_INTAKE
    assert "decision" not in payload and "candidates" not in payload


def test_requires_human_review_mirrors_the_decision():
    recommended = service.assign(_prospect(), config=_DETERMINISTIC)
    escalated = service.assign(_prospect(order_quantity_cases=400), config=_DETERMINISTIC)
    assert recommended.requires_human_review is False
    assert escalated.requires_human_review is True


# --- 6. the optional inline brief --------------------------------------------
#
# Deferred by default: composing the brief is the one open-ended, model-driven
# step in the system, and it belongs behind a specialist opening the escalation
# rather than on the critical path of a decision nobody may read.


@pytest.fixture
def brief_calls(monkeypatch):
    """Record calls to the headless brief composer without running an agent."""
    import smart_assignment.triage.headless as headless

    calls = []

    def _fake(customer, recommendation, config, **kwargs):
        calls.append((customer, recommendation))
        return "SITUATION\nstub brief."

    monkeypatch.setattr(headless, "compose_brief", _fake)
    return calls


def test_no_brief_is_composed_by_default(brief_calls):
    outcome = service.assign(_prospect(order_quantity_cases=400), config=_DETERMINISTIC)
    assert outcome.requires_human_review is True
    assert outcome.brief is None
    assert brief_calls == [], "the brief must be deferred unless asked for"
    assert "brief" not in outcome.to_dict()


def test_include_brief_attaches_one_on_an_escalation(brief_calls):
    outcome = service.assign(
        _prospect(order_quantity_cases=400), config=_DETERMINISTIC, include_brief=True
    )
    assert outcome.requires_human_review is True
    assert outcome.brief is not None and "SITUATION" in outcome.brief
    assert outcome.to_dict()["brief"] == outcome.brief
    assert len(brief_calls) == 1


def test_no_brief_when_the_prospect_was_auto_assigned(brief_calls):
    outcome = service.assign(_prospect(), config=_DETERMINISTIC, include_brief=True)
    assert outcome.requires_human_review is False
    assert outcome.brief is None
    assert brief_calls == [], "there is nothing to triage on a recommendation"


def test_include_brief_respects_the_escalation_triage_flag(brief_calls):
    from dataclasses import replace

    config = replace(_DETERMINISTIC, use_escalation_triage=False)
    outcome = service.assign(
        _prospect(order_quantity_cases=400), config=config, include_brief=True
    )
    assert outcome.brief is None
    assert brief_calls == []


def test_a_brief_that_cannot_be_composed_leaves_the_decision_intact(monkeypatch):
    """The brief is advisory. Losing it must cost the escalation nothing -- the
    structured facts a specialist needs are already on the outcome."""
    import smart_assignment.triage.headless as headless

    monkeypatch.setattr(headless, "compose_brief", lambda *a, **k: None)

    plain = service.assign(_prospect(order_quantity_cases=400), config=_DETERMINISTIC)
    with_brief = service.assign(
        _prospect(order_quantity_cases=400), config=_DETERMINISTIC, include_brief=True
    )

    assert with_brief.brief is None
    assert with_brief.decision == plain.decision
    assert with_brief.decision["review_reason"]
    assert with_brief.decision["rejected_alternatives"]


def test_batch_passes_the_brief_choice_through(brief_calls):
    outcomes = service.assign_many(
        [_prospect(), _prospect(order_quantity_cases=400)],
        config=_DETERMINISTIC,
        include_brief=True,
    )
    assert outcomes[0].brief is None  # recommended: nothing to triage
    assert outcomes[1].brief is not None  # escalated: brief attached
    assert len(brief_calls) == 1


# --- 7. event-loop ownership -------------------------------------------------


def test_one_loop_is_shared_across_calls_and_stays_open():
    """The reason this exists: `asyncio.run` per call would close the loop the
    backend's cached session is bound to, and the next call would fail with
    "Event loop is closed"."""
    with service._llm_host_loop():
        first = _HOST_EVENT_LOOP.get()
    with service._llm_host_loop():
        second = _HOST_EVENT_LOOP.get()

    assert first is second
    assert first is not None and first.is_running()


def test_an_existing_host_loop_is_never_overridden():
    """Under the web app, uvicorn's loop is already recorded and the backend's
    session is bound *there* -- replacing it would break the very thing the
    shared loop is protecting."""

    async def _outer():
        token = _HOST_EVENT_LOOP.set(asyncio.get_running_loop())
        try:
            server_loop = _HOST_EVENT_LOOP.get()
            with service._llm_host_loop():
                assert _HOST_EVENT_LOOP.get() is server_loop
        finally:
            _HOST_EVENT_LOOP.reset(token)

    asyncio.run(_outer())


def test_the_host_loop_setting_does_not_leak_to_the_caller():
    before = _HOST_EVENT_LOOP.get()
    with service._llm_host_loop():
        pass
    assert _HOST_EVENT_LOOP.get() is before
