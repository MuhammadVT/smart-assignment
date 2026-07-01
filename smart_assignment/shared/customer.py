"""
Sysco customer-number helpers.

At Sysco a customer is uniquely identified by a **customer number** in the
form ``NNN-NNNNNN`` — the first 3 digits are the site/OpCo, the last 6 are the
customer number within that site. The two together uniquely identify a
customer across the enterprise.

Centralizing the format here keeps validation consistent everywhere a customer
number is accepted (intake, the ADK entry node, tests).
"""

from __future__ import annotations

import re

# 3-digit site/OpCo, a hyphen, then a 6-digit per-site customer number.
CUSTOMER_NUMBER_PATTERN = r"^\d{3}-\d{6}$"
_CUSTOMER_NUMBER_RE = re.compile(CUSTOMER_NUMBER_PATTERN)

CUSTOMER_NUMBER_FORMAT_HINT = "expected Sysco customer number 'NNN-NNNNNN' (e.g. 067-123456)"


def is_valid_customer_number(value: str) -> bool:
    """True if ``value`` matches the Sysco customer-number format."""
    return bool(_CUSTOMER_NUMBER_RE.match(value or ""))


def normalize_customer_number(value: str) -> str:
    """Trim surrounding whitespace so lookups/validation are forgiving of it."""
    return (value or "").strip()


def validate_customer_number(value: str) -> str:
    """Return the normalized customer number, or raise ``ValueError`` if malformed."""
    normalized = normalize_customer_number(value)
    if not is_valid_customer_number(normalized):
        raise ValueError(f"Invalid customer number {value!r}: {CUSTOMER_NUMBER_FORMAT_HINT}")
    return normalized


def split_customer_number(value: str) -> tuple[str, str]:
    """Split a valid customer number into ``(site_id, site_customer_number)``."""
    normalized = validate_customer_number(value)
    site, number = normalized.split("-", 1)
    return site, number
