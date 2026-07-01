"""
Unit tests for the Sysco customer-number format (shared/customer.py).
"""

from __future__ import annotations

import pytest

from smart_assignment.shared.customer import (
    is_valid_customer_number,
    split_customer_number,
    validate_customer_number,
)


@pytest.mark.parametrize("value", ["067-123456", "000-000000", "999-999999"])
def test_valid_numbers_accepted(value):
    assert is_valid_customer_number(value)
    assert validate_customer_number(value) == value


@pytest.mark.parametrize(
    "value",
    [
        "67-123456",  # site too short
        "067-12345",  # number too short
        "0671-23456",  # misplaced hyphen
        "067123456",  # missing hyphen
        "abc-123456",  # non-numeric site
        "067-12345a",  # non-numeric number
        "",  # empty
    ],
)
def test_invalid_numbers_rejected(value):
    assert not is_valid_customer_number(value)
    with pytest.raises(ValueError):
        validate_customer_number(value)


def test_surrounding_whitespace_is_tolerated():
    assert validate_customer_number("  067-123456 ") == "067-123456"


def test_split_returns_site_and_number():
    assert split_customer_number("067-123456") == ("067", "123456")
