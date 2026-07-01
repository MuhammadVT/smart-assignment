"""
[MOCK] Sample new-customer intake records in a Sysco foodservice context.

Each account is chosen to exercise a different branch of the workflow so the
demo shows the full range of outcomes (clear recommend, low-confidence
escalation, no-feasible-slot escalation). Addresses resolve via
integrations/geocoding_client.py.

Customer numbers use the Sysco ``NNN-NNNNNN`` format (site/OpCo + per-site
number). All four sit on the same mock Houston site ``067``.
"""

from __future__ import annotations

from datetime import time

from smart_assignment.shared.models import CustomerProfile

SAMPLE_CUSTOMERS: list[CustomerProfile] = [
    # Downtown restaurant, modest order, clear morning preference.
    # -> sits right in the dense Central Houston route -> clean recommend.
    CustomerProfile(
        customer_number="067-100001",
        name="Bayou City Bistro",
        address="1200 McKinney St, Houston, TX 77010",
        order_quantity_cases=90,
        preferred_window=(time(7, 0), time(10, 0)),
    ),
    # Galleria caterer, larger order, no stated window.
    # -> two routes are plausible and close in score -> low-confidence review.
    CustomerProfile(
        customer_number="067-100002",
        name="Galleria Grill & Catering",
        address="5085 Westheimer Rd, Houston, TX 77056",
        order_quantity_cases=140,
        preferred_window=None,
    ),
    # Far-west Katy steakhouse, large order.
    # -> nearest routes are out of serviceable range or over capacity -> escalate.
    CustomerProfile(
        customer_number="067-100003",
        name="Katy Prairie Steakhouse",
        address="24600 Katy Fwy, Katy, TX 77494",
        order_quantity_cases=260,
        preferred_window=(time(6, 0), time(8, 0)),
    ),
    # The Woodlands cafe, mid-size order, late-morning preference.
    # -> the lightly-booked North route fits well -> clean recommend.
    CustomerProfile(
        customer_number="067-100004",
        name="Woodlands Fresh Cafe",
        address="1201 Lake Woodlands Dr, The Woodlands, TX 77380",
        order_quantity_cases=150,
        preferred_window=(time(9, 0), time(12, 0)),
    ),
]
