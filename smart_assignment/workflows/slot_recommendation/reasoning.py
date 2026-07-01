"""
Confidence scoring and the pluggable *reasoning* layer (spec step 5:
"output the top-ranked slot with a full reasoning trace").

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

from smart_assignment.shared.config import Config
from smart_assignment.shared.models import (
    CandidateEvaluation,
    CustomerProfile,
)
from smart_assignment.shared.timeutils import fmt_window


def compute_confidence(ranked: list[CandidateEvaluation], config: Config) -> float:
    """
    Confidence blends how *good* the top option is with how *clearly* it beats
    the runner-up. A strong, clearly-separated winner scores high; a mediocre
    option or a near-tie scores low (and will trip the escalation threshold).
    """
    if not ranked:
        return 0.0
    top = ranked[0].total_score
    if len(ranked) == 1:
        # Only one feasible slot: fairly certain, but capped by its own quality.
        return round(0.5 + 0.5 * top, 2)
    margin = top - ranked[1].total_score
    separation = max(0.0, min(1.0, margin / config.confidence_separation_ref))
    return round(0.6 * top + 0.4 * separation, 2)


class Reasoner(Protocol):
    def explain(
        self,
        customer: CustomerProfile,
        ranked: list[CandidateEvaluation],
        infeasible: list[CandidateEvaluation],
        confidence: float,
        config: Config,
    ) -> str: ...


def _describe_winner(winner: CandidateEvaluation) -> str:
    top_factors = sorted(winner.factor_scores, key=lambda f: f.weighted, reverse=True)
    parts = [f"{f.name} ({f.detail})" for f in top_factors]
    return "; ".join(parts)


class DeterministicReasoner:
    """Builds the reasoning trace straight from the score breakdown."""

    def explain(
        self,
        customer: CustomerProfile,
        ranked: list[CandidateEvaluation],
        infeasible: list[CandidateEvaluation],
        confidence: float,
        config: Config,
    ) -> str:
        if not ranked:
            reasons = []
            for cand in infeasible:
                failed = ", ".join(c.name for c in cand.failed_constraints)
                reasons.append(f"{cand.route.route_id} ({cand.route.day.value}): failed {failed}")
            joined = "; ".join(reasons) if reasons else "no candidate routes found nearby"
            return (
                f"No feasible slot for {customer.name}: every candidate route was "
                f"ruled out by a hard constraint [{joined}]. Escalating to a routing "
                f"specialist for a manual decision (new route, schedule change, or "
                f"capacity reallocation)."
            )

        winner = ranked[0]
        lead = (
            f"Recommending {winner.route.route_id} ({winner.route.name}) on "
            f"{winner.route.day.value}, window {fmt_window(winner.chosen_window)}, "
            f"score {winner.total_score:.2f}. Key factors: {_describe_winner(winner)}."
        )

        alt_lines = []
        for cand in ranked[1:]:
            gap = winner.total_score - cand.total_score
            alt_lines.append(
                f"{cand.route.route_id}/{cand.route.day.value} scored "
                f"{cand.total_score:.2f} (−{gap:.2f})"
            )
        alts = (" Passed over: " + "; ".join(alt_lines) + ".") if alt_lines else ""

        if confidence < config.confidence_threshold:
            tail = (
                f" Confidence {confidence:.0%} is below the "
                f"{config.confidence_threshold:.0%} threshold (options are close / "
                f"scores modest) — flagging for human review before committing."
            )
        else:
            tail = f" Confidence {confidence:.0%}."

        return lead + alts + tail


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
        confidence: float,
        config: Config,
    ) -> str:
        facts = self._fallback.explain(customer, ranked, infeasible, confidence, config)
        try:
            from google import genai  # imported lazily; only needed when used

            from smart_assignment.workflows.slot_recommendation.prompts import (
                build_reasoning_prompt,
            )

            client = genai.Client()
            resp = client.models.generate_content(
                model=config.model, contents=build_reasoning_prompt(facts)
            )
            text = (resp.text or "").strip()
            return text or facts
        except Exception:
            # No key, no network, or SDK change -> deterministic trace still works.
            return facts
