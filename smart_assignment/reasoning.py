"""
Escalation gating and the pluggable *reasoning* layer (spec step 5: "output
the top-ranked slot with a full reasoning trace").

Two reasoners are provided behind a common `Reasoner` protocol:

  - `DeterministicReasoner` (default): builds an auditable natural-language
    trace directly from the weighted-score breakdown. No API key, fully
    reproducible — this is what the demo uses so you can see output offline.

  - `LLMReasoner` (optional): hands the same structured facts to Gemini for a
    more fluent narrative. Requires GOOGLE_API_KEY / Vertex config; falls
    back to the deterministic trace on any error.

Swapping reasoners is a one-argument change to the pipeline — the ranking
itself is unaffected, so the *decision* stays deterministic and testable
either way.
"""

from __future__ import annotations

from typing import Optional, Protocol

from smart_assignment.shared.config import (
    FACTOR_CAPACITY_BUFFER,
    FACTOR_GEO_CLUSTERING,
    FACTOR_WINDOW_MATCH,
    Config,
)
from smart_assignment.shared.constraints import CONSTRAINT_LABEL
from smart_assignment.shared.models import (
    CandidateEvaluation,
    CustomerProfile,
    FactorScore,
)
from smart_assignment.shared.timeutils import day_label, fmt_window


def compute_total_score(ranked: list[CandidateEvaluation]) -> float:
    """
    The winning route's own total_score IS the decision-gating number -- there
    is no separate "confidence" computed from how close a runner-up scored.

    Earlier designs blended in a bonus/penalty for how clearly the winner beat
    the #2 option, which had a real flaw: two routes tied at a high score both
    got marked down as "uncertain," even though either one would serve the
    customer well. A route's own merit should stand on its own -- if it clears
    the bar, it clears it, regardless of what else was nearby. (The runner-up
    is still surfaced to the reader via `rejected_alternatives` -- it's just no
    longer an input to the number that gates auto-assignment.)
    """
    if not ranked:
        return 0.0
    return round(ranked[0].total_score, 2)


class Reasoner(Protocol):
    def explain(
        self,
        customer: CustomerProfile,
        ranked: list[CandidateEvaluation],
        infeasible: list[CandidateEvaluation],
        total_score: float,
        config: Config,
    ) -> str: ...


def _clustering_sentence(f: FactorScore) -> str:
    if f.value >= 0.85:
        opening = "it sits right in the middle of the stops already on this route"
    elif f.value >= 0.5:
        opening = "it lines up reasonably well with the stops already on this route"
    else:
        opening = "it sits a bit further out from the route's usual stops"
    return f"Geographically, {opening} — {f.detail}."


def _capacity_sentence(f: FactorScore) -> str:
    if f.value >= 0.999:
        body = "the truck still has plenty of room to spare after this order"
    elif f.value >= 0.5:
        body = "the truck is getting fuller, though it stays inside a comfortable enough margin"
    elif f.value > 0:
        body = (
            "the truck is getting quite full for this order — there's still some "
            "room, but not a lot of cushion"
        )
    else:
        body = "the truck would end up right at the edge of what it is allowed to carry"
    return f"On capacity, {body} — {f.detail}."


def _slot_sentence(f: FactorScore) -> str:
    if "no stated preference" in f.detail:
        return (
            f"The customer did not name a preferred day or time, so I treated every "
            f"option evenly on that front — {f.detail}."
        )
    if f.value >= 0.999:
        body = "both the day and the time line up exactly with what the customer asked for"
    elif f.value > 0:
        body = "the day lines up and it covers part of the preferred window, though not all of it"
    else:
        body = "this doesn't line up with what the customer asked for, on the day or the time"
    return f"On timing, {body} — {f.detail}."


_FACTOR_SENTENCE = {
    FACTOR_GEO_CLUSTERING: _clustering_sentence,
    FACTOR_CAPACITY_BUFFER: _capacity_sentence,
    FACTOR_WINDOW_MATCH: _slot_sentence,
}


def _factor_sentence(f: FactorScore) -> str:
    builder = _FACTOR_SENTENCE.get(f.name)
    return builder(f) if builder else f.detail


class DeterministicReasoner:
    """
    Builds a natural-language reasoning trace straight from the score
    breakdown — written to read like a colleague explaining their own
    decision, not a log line. It never changes the decision or the numbers,
    only how they're narrated, so it stays fully reproducible.
    """

    def explain(
        self,
        customer: CustomerProfile,
        ranked: list[CandidateEvaluation],
        infeasible: list[CandidateEvaluation],
        total_score: float,
        config: Config,
    ) -> str:
        if not ranked:
            return self._no_feasible_slot(customer, infeasible)
        return self._recommendation(customer, ranked, total_score, config)

    def _no_feasible_slot(
        self, customer: CustomerProfile, infeasible: list[CandidateEvaluation]
    ) -> str:
        if not infeasible:
            return (
                f"I wasn't able to find any candidate route anywhere near {customer.name} to "
                f"even consider. That points to a gap in route coverage rather than a capacity "
                f"or timing problem, so I'm handing this over to a routing specialist who can "
                f"look at options like standing up a new route or adjusting an existing one."
            )
        clauses = []
        for cand in infeasible:
            reasons = [CONSTRAINT_LABEL.get(c.name, c.name) for c in cand.failed_constraints]
            clauses.append(
                f"route {cand.route.route_id} on {day_label(cand.route.day)} didn't work "
                f"because of {' and '.join(reasons)}"
            )
        joined = "; ".join(clauses)
        return (
            f"I looked at every nearby route for {customer.name}, and none of them could take "
            f"this order: {joined}. Since nothing cleared the basic requirements, I'm escalating "
            f"this to a routing specialist rather than force a fit. This will most likely need a "
            f"manual decision, such as opening a new route, shifting a schedule, or freeing up "
            f"capacity somewhere else."
        )

    def _recommendation(
        self,
        customer: CustomerProfile,
        ranked: list[CandidateEvaluation],
        total_score: float,
        config: Config,
    ) -> str:
        winner = ranked[0]
        route = winner.route
        opening = (
            f"For {customer.name}, I recommend route {route.route_id} ({route.name}), "
            f"delivering on {day_label(route.day)} between {fmt_window(winner.chosen_window)}."
        )

        ordered_factors = sorted(winner.factor_scores, key=lambda f: f.weighted, reverse=True)
        factor_text = " ".join(_factor_sentence(f) for f in ordered_factors)

        if len(ranked) > 1:
            runner_up = ranked[1]
            gap = winner.total_score - runner_up.total_score
            if gap < 0.05:
                compare = (
                    f"I did weigh this against route {runner_up.route.route_id} on "
                    f"{day_label(runner_up.route.day)}, and honestly the two were close — it "
                    f"trailed by only {gap:.2f} points, so this wasn't an obvious choice."
                )
            else:
                compare = (
                    f"The next best option was route {runner_up.route.route_id} on "
                    f"{day_label(runner_up.route.day)}, but it trailed by a clearer margin "
                    f"({gap:.2f} points), so {route.route_id} was the stronger pick."
                )
        else:
            compare = (
                "It was also the only route that cleared every requirement, so there wasn't "
                "anything else to weigh it against."
            )

        if total_score < config.total_score_threshold:
            closing = (
                f"Putting all of that together, this pick's total score comes out to "
                f"{total_score:.0%}, which falls short of the {config.total_score_threshold:.0%} "
                f"bar I use before auto-assigning. Rather than commit on my own, I'd like a "
                f"specialist to take a quick look before this goes out."
            )
        else:
            closing = (
                f"Putting all of that together, this pick earns a total score of "
                f"{total_score:.0%} — comfortably above the {config.total_score_threshold:.0%} "
                f"bar I use before auto-assigning, so I'm comfortable moving ahead without a "
                f"specialist review."
            )

        return " ".join([opening, factor_text, compare, closing])


class LLMReasoner:
    """
    Optional Gemini-backed narrative. Uses the same structured facts as the
    deterministic reasoner; falls back to it if the model/credentials are
    unavailable. Kept import-light so nothing here requires an API key at
    import time.
    """

    def __init__(self, config: Optional[Config] = None):
        self._config = config or Config.from_env()
        self._fallback = DeterministicReasoner()

    def explain(
        self,
        customer: CustomerProfile,
        ranked: list[CandidateEvaluation],
        infeasible: list[CandidateEvaluation],
        total_score: float,
        config: Config,
    ) -> str:
        facts = self._fallback.explain(customer, ranked, infeasible, total_score, config)
        try:
            from smart_assignment.reasoning_prompts import build_reasoning_prompt
            from smart_assignment.shared.config import ROLE_REASONING
            from smart_assignment.shared.llm import generate_text

            text = generate_text(
                config.for_role(ROLE_REASONING), build_reasoning_prompt(facts)
            ).strip()
            return text or facts
        except Exception:
            # No credentials, no network, or SDK unavailable -> deterministic trace still works.
            return facts
