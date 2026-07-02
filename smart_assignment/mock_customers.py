"""
[MOCK] Sample new-customer intake records in a Sysco context.

Each account is chosen to exercise a different branch of the workflow so the
demo shows the full range of outcomes (clear recommend, low-total-score
escalation, no-feasible-slot escalation).

These are **prospects** — Salesforce/CRM has their address, but they don't
have a Sysco customer number yet, so ``customer_number`` is left unset
(``None``) and address is the identifier that drives everything, including
geocoding via integrations/geocoding_client.py. A preferred slot, when
stated, always includes a **day of week** plus a time-of-day window.
"""

from __future__ import annotations

from datetime import time

from smart_assignment.shared.models import CustomerProfile, DayOfWeek, PreferredSlot

SAMPLE_CUSTOMERS: list[CustomerProfile] = [
    # Downtown restaurant, modest order, prefers Tuesday mornings.
    # -> sits right in the dense Central Houston route -> clean recommend.
    CustomerProfile(
        name="Bayou City Bistro",
        address="1200 McKinney St, Houston, TX 77010",
        order_quantity_cases=90,
        preferred_slot=PreferredSlot(DayOfWeek.TUE, (time(7, 0), time(10, 0))),
    ),
    # Galleria caterer, a large catering order, no stated slot.
    # -> the order is big enough that only one nearby route can still take
    #    it, and even that route ends up quite full -> its own total score
    #    lands below the auto-assign bar -> escalate for a specialist's
    #    sanity check (a genuine quality concern, not a tie-breaking artifact).
    CustomerProfile(
        name="Galleria Grill & Catering",
        address="5085 Westheimer Rd, Houston, TX 77056",
        order_quantity_cases=400,
        preferred_slot=None,
    ),
    # Far-west Katy steakhouse, large order, prefers Tuesday early morning.
    # -> nearest routes are out of serviceable range or over capacity -> escalate.
    CustomerProfile(
        name="Katy Prairie Steakhouse",
        address="24600 Katy Fwy, Katy, TX 77494",
        order_quantity_cases=260,
        preferred_slot=PreferredSlot(DayOfWeek.TUE, (time(6, 0), time(8, 0))),
    ),
    # The Woodlands cafe, mid-size order, prefers Thursday late-morning.
    # -> the lightly-booked North route fits well -> clean recommend.
    CustomerProfile(
        name="Woodlands Fresh Cafe",
        address="1201 Lake Woodlands Dr, The Woodlands, TX 77380",
        order_quantity_cases=150,
        preferred_slot=PreferredSlot(DayOfWeek.THU, (time(9, 0), time(12, 0))),
    ),
]
