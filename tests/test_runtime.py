"""
Tests for the agent-free one-shot entry point (``smart_assignment.runtime``).

Three properties matter here, in descending order of importance:

1. **It is a facade, not a second implementation.** ``assign`` must produce the
   same decision ``pipeline.run_slot_recommendation`` produces for the same
   customer and config -- otherwise the "fast path" could silently disagree with
   the conversational agent.
2. **``economy`` is the documented deterministic floor**, not a new behavior.
3. **The fast path stays free of the ADK import**, which is the whole reason a
   service can use it without paying the agent stack's cost.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from smart_assignment import runtime
from smart_assignment.integrations.geocoding_client import MockGeocoder
from smart_assignment.mock_customers import SAMPLE_CUSTOMERS
from smart_assignment.pipeline import run_slot_recommendation
from smart_assignment.reasoning import DeterministicReasoner
from smart_assignment.shared.config import Config
from smart_assignment.shared.geo import AddressNotFoundError, GeocodingServiceError

_ADDRESS = "1200 McKinney St, Houston, TX 77010"


@pytest.fixture
def geocoder() -> MockGeocoder:
    return MockGeocoder()


# --- the facade contract ---------------------------------------------------


def test_assign_matches_the_pipeline_directly(geocoder):
    """The whole point of a facade: same inputs, same decision as the pipeline."""
    config = runtime.economy_config()
    direct = run_slot_recommendation(
        SAMPLE_CUSTOMERS[0],
        config=config,
        geocoder=MockGeocoder(),
        reasoner=DeterministicReasoner(),
    )

    customer = SAMPLE_CUSTOMERS[0]
    slot = customer.preferred_slot
    viaduct = runtime.assign(
        customer.address,
        customer.order_quantity_cases,
        name=customer.name,
        customer_number=customer.customer_number,
        preferred_day=slot.day.value if slot else None,
        preferred_window_start=slot.window[0].strftime("%H:%M") if slot else None,
        preferred_window_end=slot.window[1].strftime("%H:%M") if slot else None,
        geocoder=geocoder,
    )

    assert viaduct["ok"] is True
    assert viaduct["decision"] == direct.recommendation.decision.value
    assert viaduct["recommended_route_id"] == direct.recommendation.recommended_route_id
    assert viaduct["recommended_window"] == direct.recommendation.recommended_window
    assert viaduct["total_score"] == direct.recommendation.total_score


def test_assign_serializes_the_full_audit_trail(geocoder):
    result = runtime.assign(_ADDRESS, 90, geocoder=geocoder)

    assert result["ok"] is True
    assert result["customer"]["address"] == _ADDRESS
    assert result["customer"]["latitude"] is not None
    # Every candidate considered rides along with its constraint outcomes, so a
    # caller can reconstruct WHY -- not just read the answer.
    assert result["candidates_considered"]
    for candidate in result["candidates_considered"]:
        assert candidate["route_id"]
        assert isinstance(candidate["feasible"], bool)
        assert candidate["constraints"]


def test_assign_result_is_json_safe(geocoder):
    import json

    json.dumps(runtime.assign(_ADDRESS, 90, geocoder=geocoder))


# --- cost profiles ---------------------------------------------------------


def test_economy_config_disables_every_llm_layer():
    economy = runtime.economy_config()
    assert economy.use_route_slot_scoring is False
    assert economy.use_grounded_route_slot_escalation is False
    assert economy.use_grounded_judgment is False
    assert economy.use_grounded_slot_selection is False
    assert economy.use_escalation_triage is False
    assert economy.use_address_resolution is False


def test_economy_reproduces_the_deterministic_baseline(geocoder):
    """``economy`` must equal the plain deterministic pipeline -- it is the floor
    every grounded layer already falls back to, invoked directly."""
    baseline = run_slot_recommendation(
        SAMPLE_CUSTOMERS[0],
        config=Config(
            use_route_slot_scoring=False,
            use_grounded_judgment=False,
            use_grounded_slot_selection=False,
        ),
        geocoder=MockGeocoder(),
        reasoner=DeterministicReasoner(),
    )
    customer = SAMPLE_CUSTOMERS[0]
    economy = runtime.assign(
        customer.address,
        customer.order_quantity_cases,
        name=customer.name,
        customer_number=customer.customer_number,
        profile=runtime.PROFILE_ECONOMY,
        geocoder=geocoder,
    )
    assert economy["decision"] == baseline.recommendation.decision.value
    assert economy["recommended_route_id"] == baseline.recommendation.recommended_route_id


def test_balanced_keeps_one_grounded_call_and_no_resampling():
    balanced = runtime.balanced_config()
    assert balanced.use_route_slot_scoring is True
    assert balanced.use_grounded_route_slot_escalation is True
    # k=1 is what disables the escalation-side resampling loop.
    assert balanced.judgment_sample_count == 1
    assert balanced.use_escalation_triage is False


def test_full_profile_returns_the_environment_config_unchanged():
    from smart_assignment.shared.config import DEFAULT_CONFIG

    assert runtime.resolve_config(runtime.PROFILE_FULL) is DEFAULT_CONFIG


def test_unknown_profile_raises():
    with pytest.raises(ValueError, match="unknown profile"):
        runtime.resolve_config("turbo")


def test_explicit_config_always_wins_over_profile(geocoder):
    """A caller with its own tuned Config is never overridden by a profile name."""
    explicit = Config(top_n_candidate_routes=1)
    result = runtime.assign(
        _ADDRESS, 90, profile=runtime.PROFILE_FULL, config=explicit, geocoder=geocoder
    )
    assert len(result["candidates_considered"]) == 1


# --- expected failures are values, not exceptions --------------------------


def test_missing_address_is_an_invalid_intake_value(geocoder):
    result = runtime.assign("", 90, geocoder=geocoder)
    assert result["ok"] is False
    assert result["error_kind"] == runtime.ERROR_INVALID_INTAKE


def test_non_positive_order_quantity_is_rejected(geocoder):
    result = runtime.assign(_ADDRESS, 0, geocoder=geocoder)
    assert result["ok"] is False
    assert result["error_kind"] == runtime.ERROR_INVALID_INTAKE


def test_partial_preferred_slot_is_rejected_with_a_useful_message(geocoder):
    result = runtime.assign(_ADDRESS, 90, preferred_day="TUE", geocoder=geocoder)
    assert result["ok"] is False
    assert "day AND both a start and end" in result["error"]


def test_unknown_day_is_rejected(geocoder):
    result = runtime.assign(
        _ADDRESS,
        90,
        preferred_day="FUNDAY",
        preferred_window_start="07:00",
        preferred_window_end="10:00",
        geocoder=geocoder,
    )
    assert result["ok"] is False
    assert "Unknown preferred_day" in result["error"]


class _NotFoundGeocoder:
    """A geocoder that can't resolve anything (MockGeocoder falls back to a
    default point rather than raising, so it can't exercise this path)."""

    def geocode(self, address: str):
        raise AddressNotFoundError(address, "no match")


class _BrokenGeocoder:
    def geocode(self, address: str):
        raise GeocodingServiceError(address, "upstream is down")


def test_unfindable_address_reports_address_not_found():
    result = runtime.assign("nowhere at all, XX", 90, geocoder=_NotFoundGeocoder())
    assert result["ok"] is False
    assert result["error_kind"] == runtime.ERROR_ADDRESS_NOT_FOUND
    assert "nowhere at all, XX" in result["error"]


def test_geocoder_outage_is_distinguishable_from_a_bad_address():
    """A caller must be able to tell 'retry later' from 'fix the address'."""
    result = runtime.assign(_ADDRESS, 90, geocoder=_BrokenGeocoder())
    assert result["ok"] is False
    assert result["error_kind"] == runtime.ERROR_GEOCODER_UNAVAILABLE


# --- batch -----------------------------------------------------------------


def test_assign_batch_preserves_order_and_isolates_failures(geocoder):
    results = runtime.assign_batch(
        [
            {"address": _ADDRESS, "order_quantity_cases": 90},
            {"address": "", "order_quantity_cases": 10},
            {"address": _ADDRESS, "order_quantity_cases": 20, "name": "Second"},
        ],
        geocoder=geocoder,
    )
    assert [r["ok"] for r in results] == [True, False, True]
    # A bad row never costs the others their result.
    assert results[2]["customer"]["name"] == "Second"


def test_assign_batch_fetches_the_route_world_once(geocoder):
    """The batch must not re-fetch routes per prospect (the reason it exists)."""
    calls = {"n": 0}
    real = runtime.fetch_candidate_routes

    def counting_fetch():
        calls["n"] += 1
        return real()

    runtime.fetch_candidate_routes = counting_fetch
    try:
        runtime.assign_batch(
            [{"address": _ADDRESS, "order_quantity_cases": n} for n in (10, 20, 30)],
            geocoder=geocoder,
        )
    finally:
        runtime.fetch_candidate_routes = real
    assert calls["n"] == 1


# --- import weight ---------------------------------------------------------


def test_importing_runtime_does_not_import_google_adk():
    """The fast path must not pay the agent stack's import cost. Run in a
    subprocess so an ADK import from another test can't mask a regression."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "import smart_assignment.runtime\n"
            "adk = [m for m in sys.modules if m.startswith('google.adk')]\n"
            "assert not adk, adk\n"
            "print('ok')\n",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
