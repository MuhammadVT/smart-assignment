"""A wall-clock ceiling on a live eval run, so a hung backend fails instead of hanging.

Nothing else bounds a run. ``SAGE_TIMEOUT`` bounds one request in principle, but a
request has been observed running far past it, and retries multiply whatever that
costs -- with the network down overnight, one full eval run took **9.8 hours**
before failing, occupying the machine the whole time and saying nothing useful.

Damage control, not a fix: no run gets greener. An unbounded hang just becomes a
prompt, legible failure.

The default is deliberately loose. Healthy full runs measure 77-161s, so 1200s
carries roughly an order of magnitude of headroom and should never fire on a run
that is merely slow (more cases, ``num_runs`` > 1, a slower CI runner, a retried
request). ``.github/workflows/ci.yml``'s per-job ``timeout-minutes`` is the coarser
backstop for the one case this cannot catch -- a wedged event loop never lets the
timeout fire -- and is set higher on purpose, so the legible failure below is what
a developer normally sees.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from typing import AsyncIterator

BUDGET_ENV = "SMART_ASSIGNMENT_EVAL_BUDGET_SECONDS"
# 20 minutes. See the module docstring for why this is deliberately generous.
DEFAULT_BUDGET_SECONDS = 1200.0


class EvalRunBudgetExceeded(AssertionError):
    """A live eval run outran its wall-clock budget and was aborted. Subclasses
    ``AssertionError`` so pytest reports a test failure rather than an
    infrastructure error -- the message is explicit that nothing scored wrong."""


def resolve_budget_seconds() -> float:
    """The budget in seconds: ``SMART_ASSIGNMENT_EVAL_BUDGET_SECONDS`` when set,
    else :data:`DEFAULT_BUDGET_SECONDS`. A non-numeric value raises immediately,
    which is the right time to find out about a typo."""
    raw = os.environ.get(BUDGET_ENV)
    return float(raw) if raw and raw.strip() else DEFAULT_BUDGET_SECONDS


def _explain(budget: float) -> str:
    """Written for someone who has just watched the suite go red: what happened,
    what it is *not*, what to grep for, and exactly how to change it."""
    return (
        f"\nThe eval run was aborted after {budget:.0f}s ({budget / 60:.0f} min).\n"
        "\n"
        "This is a BUDGET STOP, not a scoring failure -- no case was judged wrong. "
        "The run stopped finishing in the time allowed, which in practice means the "
        "environment rather than the agent. Grep the captured log above for "
        "'timed out' (a slow or wedged backend) or 'getaddrinfo' (DNS, i.e. no "
        "network). A real outage once produced a 9.8-hour run; that is why this "
        "ceiling exists.\n"
        "\n"
        f"If the run was genuinely just slow, raise it:  {BUDGET_ENV}=2400\n"
        f"The default is {DEFAULT_BUDGET_SECONDS:.0f}s and healthy full runs measure "
        "77-161s, so it already carries ~10x headroom -- hitting it on a working "
        "backend is worth investigating, not raising. See eval/run_budget.py."
    )


@contextlib.asynccontextmanager
async def run_budget() -> AsyncIterator[float]:
    """Abort the awaited block if it outlasts the run budget.

    Yields the budget actually applied. Raises :class:`EvalRunBudgetExceeded` with
    an actionable explanation on expiry; any other exception propagates untouched,
    so a real scoring failure is never reshaped into a timeout.
    """
    budget = resolve_budget_seconds()
    try:
        async with asyncio.timeout(budget):
            yield budget
    except asyncio.TimeoutError:
        raise EvalRunBudgetExceeded(_explain(budget)) from None
