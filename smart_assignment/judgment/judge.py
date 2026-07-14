"""
Decision *strategies* behind step 5 (recommend-or-escalate), and the
orchestration of the grounded-judgment one.

Two interchangeable strategies implement the `Judge` protocol:

  - `WeightedScoreJudge` — the existing behavior: rank feasible candidates by
    the weighted total_score and gate on `Config.total_score_threshold`. Thin
    wrapper over `pipeline.decide`, so nothing about today's path changes.

  - `GroundedJudge` — the LLM reasons over the evidence packet and makes the
    call itself (this module's reason for existing). Sampling + consensus +
    verification live here.

`default_judge(config)` picks between them from `config.use_grounded_judgment`,
so callers (the pipeline, the conversational tool) stay agnostic.

Both return a `SlotRecommendation`, so everything downstream (reporting, the web
app, the tools) is untouched regardless of which strategy ran.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Callable, Optional, Protocol

from smart_assignment.judgment.evidence import EvidencePacket, build_evidence_packet
from smart_assignment.judgment.llm import generate_judgment
from smart_assignment.judgment.prompts import build_judgment_prompt, build_retry_prompt
from smart_assignment.judgment.schema import (
    Confidence,
    JudgmentDecision,
    JudgmentOutput,
    parse_judgment,
)
from smart_assignment.judgment.verifier import verify
from smart_assignment.reasoning import DeterministicReasoner, LLMReasoner, Reasoner
from smart_assignment.shared.config import Config
from smart_assignment.shared.models import (
    CandidateEvaluation,
    CustomerProfile,
    Decision,
    SlotRecommendation,
)
from smart_assignment.shared.timeutils import fmt_window

logger = logging.getLogger(__name__)

# A judgment_fn turns (config, prompt) into a raw judgment dict. Injectable so
# tests drive the whole strategy with a fake and no network/credentials.
JudgmentFn = Callable[[Config, str], dict]


class Judge(Protocol):
    def decide(
        self,
        customer: CustomerProfile,
        evaluations: list[CandidateEvaluation],
        config: Config,
    ) -> SlotRecommendation: ...


# ---------------------------------------------------------------------------
# Strategy 1: the existing weighted-sum + threshold path (unchanged behavior)
# ---------------------------------------------------------------------------


class WeightedScoreJudge:
    """The default, deterministic-ranking strategy: delegates to `pipeline.decide`."""

    def __init__(self, reasoner: Optional[Reasoner] = None):
        self._reasoner = reasoner

    def decide(
        self,
        customer: CustomerProfile,
        evaluations: list[CandidateEvaluation],
        config: Config,
    ) -> SlotRecommendation:
        # Imported lazily to avoid a pipeline <-> judgment import cycle.
        from smart_assignment.pipeline import decide

        reasoner = self._reasoner or LLMReasoner(config)
        return decide(customer, evaluations, reasoner, config)


# ---------------------------------------------------------------------------
# Strategy 2: grounded LLM judgment over the evidence packet
# ---------------------------------------------------------------------------


class GroundedJudge:
    """Let an LLM make the recommend/escalate call over raw evidence.

    Flow (see the package docstring): build the evidence packet, draw one
    verified sample; if it's a confident recommendation, ship it; otherwise
    ("escalation-side") draw up to `judgment_sample_count` samples total and
    combine their decisions by the configured consensus rule. Any mechanical
    failure (unparseable/ungrounded output surviving one corrective retry) falls
    back to the deterministic weighted pick, so the result is never worse than
    today's.
    """

    def __init__(
        self,
        judgment_fn: Optional[JudgmentFn] = None,
        fallback_reasoner: Optional[Reasoner] = None,
    ):
        self._judgment_fn = judgment_fn or generate_judgment
        self._fallback_reasoner = fallback_reasoner or DeterministicReasoner()

    # -- public entry point --

    def decide(
        self,
        customer: CustomerProfile,
        evaluations: list[CandidateEvaluation],
        config: Config,
    ) -> SlotRecommendation:
        feasible = [e for e in evaluations if e.feasible]
        if not feasible:
            # No feasible route is a deterministic outcome -- no LLM needed, and
            # identical to today's ESCALATED_NO_FEASIBLE_SLOT.
            return self._deterministic_fallback(customer, evaluations, config)

        packet = build_evidence_packet(customer, evaluations, config)

        first = self._verified_sample(packet, config)
        if first is None:
            logger.warning(
                "Grounded judgment produced no verified result; falling back to the "
                "deterministic weighted pick (output will be identical to "
                "SMART_ASSIGNMENT_USE_GROUNDED_JUDGMENT=false). See the WARNING above "
                "for the underlying cause."
            )
            rec = self._deterministic_fallback(customer, evaluations, config)
            rec.grounded_fallback = True
            rec.grounded_fallback_reason = (
                "Grounded LLM reasoning was unavailable, so this shows the deterministic "
                "result. Check the LLM backend (SMART_ASSIGNMENT_LLM_BACKEND) and its "
                "credentials."
            )
            return rec

        if self._ships_on_first_call(first, config):
            return self._to_recommendation(customer, first, packet, samples=[first])

        # Escalation-side: spend the sampling budget.
        samples = [first]
        for _ in range(max(1, config.judgment_sample_count) - 1):
            extra = self._verified_sample(packet, config)
            if extra is not None:
                samples.append(extra)
        return self._resolve_escalation_side(customer, samples, packet, config)

    # -- sampling + verification --

    def _verified_sample(self, packet: EvidencePacket, config: Config) -> Optional[JudgmentOutput]:
        """One sample: call the model, parse, verify; one corrective retry on a
        verification failure; None on any mechanical failure.

        Every failure path is logged (WARNING) so a silent deterministic
        fallback is never a mystery -- a missing LLM backend/credentials, a
        malformed reply, or a persistent grounding failure all surface in the
        logs with their cause.
        """
        try:
            raw = self._judgment_fn(config, build_judgment_prompt(packet))
            output = parse_judgment(raw)
        except Exception as exc:
            logger.warning(
                "Grounded judgment LLM call/parse failed (%s: %s) -- falling back. "
                "Check SMART_ASSIGNMENT_LLM_BACKEND and its credentials.",
                type(exc).__name__,
                exc,
            )
            return None
        result = verify(output, packet)
        if result.ok:
            return output
        logger.info(
            "Grounded judgment failed verification; retrying once. Reasons: %s",
            result.as_feedback(),
        )
        try:
            raw2 = self._judgment_fn(config, build_retry_prompt(packet, result.as_feedback()))
            output2 = parse_judgment(raw2)
        except Exception as exc:
            logger.warning(
                "Grounded judgment retry call/parse failed (%s: %s) -- falling back.",
                type(exc).__name__,
                exc,
            )
            return None
        verdict = verify(output2, packet)
        if verdict.ok:
            return output2
        logger.warning(
            "Grounded judgment still ungrounded after one retry (%s) -- falling back.",
            verdict.as_feedback(),
        )
        return None

    def _ships_on_first_call(self, output: JudgmentOutput, config: Config) -> bool:
        """Only a confident recommendation ships on one call. A hard ESCALATE
        always resamples; a LOW-confidence recommend resamples too, unless the
        operator opted out via `judgment_retry_on_low_confidence_recommend`."""
        if output.decision is JudgmentDecision.ESCALATE:
            return False
        low_conf = output.confidence is Confidence.LOW
        if low_conf and config.judgment_retry_on_low_confidence_recommend:
            return False
        return True

    # -- consensus over the k samples' *decisions* (not their picks) --

    def _resolve_escalation_side(
        self,
        customer: CustomerProfile,
        samples: list[JudgmentOutput],
        packet: EvidencePacket,
        config: Config,
    ) -> SlotRecommendation:
        recommends = [s for s in samples if s.decision is JudgmentDecision.RECOMMEND]
        n = len(samples)
        if config.judgment_consensus == "majority":
            cleared = len(recommends) * 2 > n
        else:  # "unanimous" (default, precautionary)
            cleared = len(recommends) == n

        if cleared and recommends:
            representative = self._modal_recommend(recommends)
            return self._to_recommendation(customer, representative, packet, samples=samples)
        return self._to_escalation(customer, samples, packet, config)

    @staticmethod
    def _modal_recommend(recommends: list[JudgmentOutput]) -> JudgmentOutput:
        """The sample whose picked route is the most common among recommenders
        (ties broken by sample order) -- so differing-but-good picks don't count
        as disagreement, only the recommend/escalate decision does."""
        counts = Counter(s.recommended_route_id for s in recommends)
        modal_id, _ = counts.most_common(1)[0]
        for s in recommends:
            if s.recommended_route_id == modal_id:
                return s
        return recommends[0]

    # -- mapping JudgmentOutput -> SlotRecommendation --

    def _rejected_alternatives(self, packet: EvidencePacket, chosen_id: Optional[str]) -> list[str]:
        out: list[str] = []
        for c in packet.feasible_candidates:
            if c["route_id"] == chosen_id:
                continue
            score = c["facts"].get("reference_weighted_score")
            score_txt = f"{score:.2f}" if isinstance(score, (int, float)) else "n/a"
            out.append(f"{c['route_id']} ({c['day']}): feasible but scored {score_txt}")
        for c in packet.infeasible_candidates:
            failed = ", ".join(fc["name"] for fc in c.get("failed_constraints", []))
            out.append(f"{c['route_id']} ({c['day']}): infeasible — {failed}")
        return out

    @staticmethod
    def _takes(samples: list[JudgmentOutput]) -> list[str]:
        return [
            f"[{s.decision.value}/{s.confidence.value}] "
            f"{s.recommended_route_id or 'no pick'}: {s.rationale}"
            for s in samples
        ]

    def _to_recommendation(
        self,
        customer: CustomerProfile,
        output: JudgmentOutput,
        packet: EvidencePacket,
        samples: list[JudgmentOutput],
    ) -> SlotRecommendation:
        ev = packet.evaluation_for(output.recommended_route_id)
        route = ev.route
        return SlotRecommendation(
            customer_number=customer.customer_number,
            customer_address=customer.address,
            customer_name=customer.name,
            decision=Decision.RECOMMENDED,
            total_score=round(ev.total_score, 2),
            reasoning=output.rationale,
            recommended_route_id=route.route_id,
            recommended_route_name=route.name,
            recommended_day=route.day.value,
            recommended_window=fmt_window(ev.chosen_window),
            recommended_window_basis=ev.window_basis or None,
            factor_breakdown=ev.factor_scores,
            rejected_alternatives=self._rejected_alternatives(packet, route.route_id),
            review_reason=None,
            # Only meaningful when the recommend came out of a resample.
            alternative_takes=self._takes(samples) if len(samples) > 1 else [],
        )

    def _to_escalation(
        self,
        customer: CustomerProfile,
        samples: list[JudgmentOutput],
        packet: EvidencePacket,
        config: Config,
    ) -> SlotRecommendation:
        # Propose *something* for the specialist: the most-picked route across
        # samples, else the highest reference-score feasible candidate.
        picks = [s.recommended_route_id for s in samples if s.recommended_route_id]
        if picks:
            proposed_id = Counter(picks).most_common(1)[0][0]
        else:
            proposed_id = max(
                packet.feasible_candidates,
                key=lambda c: c["facts"].get("reference_weighted_score") or 0.0,
            )["route_id"]
        ev = packet.evaluation_for(proposed_id)
        recommends = sum(1 for s in samples if s.decision is JudgmentDecision.RECOMMEND)
        primary = samples[0]
        return SlotRecommendation(
            customer_number=customer.customer_number,
            customer_address=customer.address,
            customer_name=customer.name,
            decision=Decision.ESCALATED_LOW_SCORE,
            total_score=round(ev.total_score, 2),
            reasoning=primary.rationale,
            recommended_route_id=ev.route.route_id,
            recommended_route_name=ev.route.name,
            recommended_day=ev.route.day.value,
            recommended_window=fmt_window(ev.chosen_window),
            recommended_window_basis=ev.window_basis or None,
            factor_breakdown=ev.factor_scores,
            rejected_alternatives=self._rejected_alternatives(packet, ev.route.route_id),
            review_reason=(
                f"Grounded judgment did not clear this for auto-assign: "
                f"{recommends}/{len(samples)} sample(s) recommended "
                f"(consensus rule: {config.judgment_consensus}). Surfacing all "
                f"reasoned takes for a specialist."
            ),
            alternative_takes=self._takes(samples),
        )

    def _deterministic_fallback(
        self,
        customer: CustomerProfile,
        evaluations: list[CandidateEvaluation],
        config: Config,
    ) -> SlotRecommendation:
        """The existing weighted pick + deterministic reasoning -- the safety
        floor for a no-feasible case or any mechanical LLM failure."""
        from smart_assignment.pipeline import decide

        return decide(customer, evaluations, self._fallback_reasoner, config)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def default_judge(config: Config, reasoner: Optional[Reasoner] = None) -> Judge:
    """Pick the decision strategy from config.

    `use_grounded_judgment` on -> `GroundedJudge`; off -> `WeightedScoreJudge`
    (today's behavior). `reasoner` is used only by the weighted strategy and by
    the grounded strategy's fallback; it defaults appropriately in each.
    """
    if config.use_grounded_judgment:
        return GroundedJudge(fallback_reasoner=reasoner)
    return WeightedScoreJudge(reasoner=reasoner)
