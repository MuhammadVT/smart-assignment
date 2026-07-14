"""Deterministic verification of an address choice against its packet."""

from __future__ import annotations

from smart_assignment.address_resolve.evidence import build_address_packet
from smart_assignment.address_resolve.schema import AddressChoice, AddressCitation
from smart_assignment.address_resolve.verifier import verify_choice
from smart_assignment.shared.config import Config
from smart_assignment.shared.geo import AddressCandidate
from smart_assignment.shared.models import GeoPoint


def _packet():
    cands = [
        AddressCandidate("1200 McKinney St, Houston, TX 77010", GeoPoint(29.7, -95.3)),
        AddressCandidate("5085 Westheimer Rd, Houston, TX 77056", GeoPoint(29.7, -95.4)),
    ]
    return build_address_packet("1200 McKinney St, Houston", cands, Config())


def test_valid_pick_and_citation_passes():
    packet = _packet()
    sim = packet.candidate_facts(0)["similarity"]
    choice = AddressChoice(0, "match", [AddressCitation(0, "similarity", sim)])
    assert verify_choice(choice, packet).ok


def test_out_of_range_index_fails():
    v = verify_choice(AddressChoice(9, "x", []), _packet())
    assert not v.ok
    assert "not a valid candidate" in v.as_feedback()


def test_uncitable_field_and_wrong_value_fail():
    packet = _packet()
    bad_field = verify_choice(AddressChoice(0, "x", [AddressCitation(0, "distance", 1.0)]), packet)
    assert not bad_field.ok
    wrong = verify_choice(AddressChoice(0, "x", [AddressCitation(0, "similarity", 0.01)]), packet)
    assert not wrong.ok


def test_near_tolerance_but_different_similarity_fails():
    # A citation 0.015 off the stored similarity is a different number, not a
    # rounding artifact -- the old 0.02 tolerance accepted it.
    packet = _packet()
    sim = float(packet.candidate_facts(0)["similarity"])
    off = verify_choice(
        AddressChoice(0, "x", [AddressCitation(0, "similarity", round(sim + 0.015, 4))]), packet
    )
    assert not off.ok
    exact = verify_choice(
        AddressChoice(0, "x", [AddressCitation(0, "similarity", round(sim + 0.004, 4))]), packet
    )
    assert exact.ok  # within 4dp-rounding slack
