"""Evidence packet + output-schema parsing for grounded address resolution.
Pure/offline: no geocoder, no LLM."""

from __future__ import annotations

import pytest

from smart_assignment.address_resolve.evidence import (
    NUMERIC_ADDRESS_FIELDS,
    build_address_packet,
    similarity,
)
from smart_assignment.address_resolve.schema import (
    AddressChoiceParseError,
    parse_address_choice,
)
from smart_assignment.shared.config import Config
from smart_assignment.shared.geo import AddressCandidate
from smart_assignment.shared.models import GeoPoint


def _cand(formatted, lat=29.0, lng=-95.0, components=None):
    return AddressCandidate(
        formatted=formatted, location=GeoPoint(lat, lng), components=components or {}
    )


def test_similarity_is_recall_over_query_tokens():
    # 3 of the 4 typed tokens (1200, st, houston) appear in the candidate.
    assert similarity("1200 McKiney St, Houston", "1200 McKinney St, Houston, TX 77010") == 0.75
    assert similarity("", "anything") == 0.0
    assert similarity("nomatch here", "totally different") == 0.0


def test_packet_enumerates_and_marks_best_similarity():
    cands = [
        _cand("5085 Westheimer Rd, Houston, TX 77056"),
        _cand("1200 McKinney St, Houston, TX 77010"),
    ]
    packet = build_address_packet("1200 McKinney St, Houston", cands, Config())
    assert packet.n == 2
    # index 1 shares more tokens -> the deterministic reference pick.
    assert packet.deterministic_choice_index == 1
    # facts carry the citable numeric field.
    assert set(packet.candidate_facts(1).keys()) == set(NUMERIC_ADDRESS_FIELDS)
    assert packet.candidate_at(1).formatted == "1200 McKinney St, Houston, TX 77010"
    # as_dict is JSON-safe and preserves order.
    d = packet.as_dict()
    assert [c["index"] for c in d["candidates"]] == [0, 1]


def test_packet_includes_components_only_when_present():
    cands = [_cand("A", components={"city": "Houston"}), _cand("B")]
    rows = build_address_packet("q", cands, Config()).as_dict()["candidates"]
    assert rows[0]["components"] == {"city": "Houston"}
    assert "components" not in rows[1]


def test_parse_valid_choice():
    choice = parse_address_choice(
        {
            "chosen_index": 1,
            "rationale": "closest",
            "citations": [{"index": 1, "field": "similarity", "value": 0.75}],
        }
    )
    assert choice.chosen_index == 1
    assert choice.citations[0].field == "similarity"


def test_parse_rejects_missing_index_and_blank_rationale():
    with pytest.raises(AddressChoiceParseError):
        parse_address_choice({"rationale": "x"})
    with pytest.raises(AddressChoiceParseError):
        parse_address_choice({"chosen_index": 0, "rationale": "   "})
