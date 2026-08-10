"""How many times a live eval replays each case -- and why the default is once.

ADK's ``AgentEvaluator.evaluate`` defaults to ``num_runs=2``: it replays every
eval case twice and averages the metric results. That default is wrong for this
suite on three counts:

* **It costs double.** Every run is a full agent conversation against a live LLM
  backend, so the committed golden set is 8 conversations per invocation instead
  of 4 -- paid on every credentialed CI run, for a signal nobody was reading.
* **Averaging hides the thing worth seeing.** A case that passes once and fails
  once averages to a middling score that may still clear the threshold, so an
  intermittent trajectory regression reports green. One run per case fails
  outright instead. That is *redder*, deliberately: an eval that smooths over
  flakiness is telling you less than it appears to.
* **It makes "what did the agent say?" ambiguous.** Two runs produce two final
  responses per case, so anything recording that text (see eval/capture.py) has
  to pick one and explain why. At one run there is nothing to pick.

Raising it is still the right move when run-to-run *variance* is the actual
question -- ``SMART_ASSIGNMENT_EVAL_NUM_RUNS=3`` to see how stable a score is.
That is a deliberate act now rather than a silent default.

Kept in its own module, next to ``eval/run_budget.py``, for the same reason that
one exists: the eval entry points import ``google-adk[eval]`` (the ``eval``
extra, deliberately absent from ``dev``), so the hermetic ``tests/`` suite cannot
import them -- but it can import this, and does.
"""

from __future__ import annotations

import os

NUM_RUNS_ENV = "SMART_ASSIGNMENT_EVAL_NUM_RUNS"
# Once per case. See the module docstring for why this overrides ADK's own 2.
DEFAULT_NUM_RUNS = 1


def resolve_num_runs() -> int:
    """Replays per eval case: ``SMART_ASSIGNMENT_EVAL_NUM_RUNS`` when set, else
    :data:`DEFAULT_NUM_RUNS`.

    A non-integer value raises ``ValueError`` immediately -- the right time to
    find out about a typo, same discipline as ``run_budget.resolve_budget_seconds``.
    A value below 1 is rejected explicitly rather than passed through: ADK would
    take ``0`` as "run nothing", producing a green result over zero cases, which
    is precisely the silent no-op this suite exists to prevent.
    """
    raw = os.environ.get(NUM_RUNS_ENV)
    if not raw or not raw.strip():
        return DEFAULT_NUM_RUNS

    num_runs = int(raw)
    if num_runs < 1:
        raise ValueError(
            f"{NUM_RUNS_ENV}={raw!r} is not a usable number of runs. It must be at least 1 "
            "(0 would score zero cases and still report success)."
        )
    return num_runs
