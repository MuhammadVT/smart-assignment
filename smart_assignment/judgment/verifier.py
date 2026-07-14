"""
Deterministic verification of a parsed `JudgmentOutput` against its evidence
packet. No LLM call — pure Python — so it's cheap, fully testable, and it's what
actually *earns* the "grounded" claim rather than trusting a prompt instruction.

Three layers of checking, from Fable's adversarial review:

  1. Decision/pick coherence
     - a RECOMMEND must name a route that is in the feasible set;
     - an ESCALATE may name a feasible route (a proposed slot for the human) or
       none;
     - a pick is NEVER allowed to be an infeasible route (redundant with the
       output schema's enum, but enforced here too — this is the hard safety
       net that makes it impossible to auto-assign an over-capacity/out-of-area
       route no matter how the model reasons).

  2. Structured-citation grounding (primary, exact)
     - every `FactCitation` must resolve to a real fact in the packet and match
       its value (percent-vs-fraction normalized, so "0.87" and "87%" both pass
       against a stored 0.87);
     - every `ComparisonCitation` must be arithmetically true of the two cited
       facts.

  3. Free-text prose scan (secondary, tolerant)
     - numeric tokens, route-ids, and day names that appear in the rationale
       prose must be grounded in the packet. This is deliberately *tolerant*
       (it normalizes percents and only fails on a token it cannot ground at
       all) so a faithful paraphrase like "roughly 87%" passes while an invented
       "91%" that appears nowhere in the packet fails.

On any failure the judge does one corrective retry, then falls back to the
deterministic weighted pick — so a verifier false-negative costs latency, never
a wrong answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from smart_assignment.judgment.evidence import NUMERIC_FACT_KEYS, EvidencePacket
from smart_assignment.judgment.schema import (
    ComparisonCitation,
    FactCitation,
    JudgmentDecision,
    JudgmentOutput,
)

# Absolute tolerance for matching a cited number against a packet value. Facts
# are rounded to at most 4 dp when serialized, so this comfortably absorbs
# rounding without admitting a genuinely different number.
_TOL = 0.02


@dataclass
class VerificationResult:
    ok: bool
    reasons: list[str] = field(default_factory=list)

    def as_feedback(self) -> str:
        return "; ".join(self.reasons)


def _facts_for(packet: EvidencePacket, route_id: str) -> dict | None:
    cand = packet.candidate_dict(route_id)
    return cand.get("facts") if cand else None


def _values_close(a: float, b: float) -> bool:
    if abs(a - b) <= _TOL:
        return True
    # percent-vs-fraction: a model may write 87 where the packet stores 0.87.
    if abs(a / 100.0 - b) <= _TOL or abs(a - b / 100.0) <= _TOL:
        return True
    return False


def _check_fact_citation(packet: EvidencePacket, c: FactCitation) -> str | None:
    facts = _facts_for(packet, c.route_id)
    if facts is None:
        return f"citation references unknown route {c.route_id!r}"
    if c.field not in NUMERIC_FACT_KEYS:
        return f"citation field {c.field!r} is not a citable fact"
    actual = facts.get(c.field)
    if actual is None:
        return f"{c.route_id}.{c.field} has no value in the evidence packet"
    if not _values_close(float(c.value), float(actual)):
        return f"{c.route_id}.{c.field}={actual} but citation claims {c.value}"
    return None


def _check_comparison_citation(packet: EvidencePacket, c: ComparisonCitation) -> str | None:
    fa, fb = _facts_for(packet, c.route_id_a), _facts_for(packet, c.route_id_b)
    if fa is None:
        return f"comparison references unknown route {c.route_id_a!r}"
    if fb is None:
        return f"comparison references unknown route {c.route_id_b!r}"
    if c.field not in NUMERIC_FACT_KEYS:
        return f"comparison field {c.field!r} is not a citable fact"
    va, vb = fa.get(c.field), fb.get(c.field)
    if va is None or vb is None:
        return f"comparison field {c.field!r} missing on one of the routes"
    va, vb = float(va), float(vb)
    ok = (
        (c.relation == "greater" and va > vb + _TOL)
        or (c.relation == "less" and va < vb - _TOL)
        or (c.relation == "equal" and abs(va - vb) <= _TOL)
    )
    if not ok:
        return (
            f"comparison claims {c.route_id_a}.{c.field} {c.relation} "
            f"{c.route_id_b}.{c.field} but values are {va} and {vb}"
        )
    return None


def _collect_groundable_numbers(packet: EvidencePacket) -> list[float]:
    nums: list[float] = []
    for cand in packet.feasible_candidates + packet.infeasible_candidates:
        for v in (cand.get("facts") or {}).values():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                nums.append(float(v))
    nums.append(float(packet.customer.get("order_quantity_cases", 0)))
    return nums


_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
_WORD_RE = re.compile(r"[A-Za-z]+(?:-[A-Za-z0-9]+)*")


def _scan_prose(packet: EvidencePacket, output: JudgmentOutput) -> list[str]:
    """Tolerant scan: every load-bearing number / route-id in the rationale must
    ground in the evidence packet.

    Deliberately tolerant to avoid false rejections of faithful prose:
      - route-ids and HH:MM clock times are scrubbed first, so their digits are
        never mistaken for a numeric fact (e.g. the "4200" in "RTE-4200", or the
        "7" in "07:00");
      - percent-vs-fraction is normalized, so "roughly 87%" grounds against a
        stored 0.87;
      - trivially small bare integers (< 10, e.g. "the other 2 routes") are
        ignored -- too generic to be a hallucinated decision fact.
    A number that still can't be grounded (e.g. an invented "91%") fails.
    """
    reasons: list[str] = []
    text = output.rationale
    known_ids = {c["route_id"] for c in packet.feasible_candidates + packet.infeasible_candidates}
    groundable = _collect_groundable_numbers(packet)

    scrubbed = text
    for rid in known_ids:
        scrubbed = scrubbed.replace(rid, " ")
    scrubbed = re.sub(r"\d{1,2}:\d{2}", " ", scrubbed)  # clock times

    for tok in _NUMBER_RE.findall(scrubbed):
        val = float(tok)
        if "." not in tok and val < 10:  # ignore small bare counts
            continue
        candidates = [val, val / 100.0, val * 100.0]
        if not any(any(_values_close(cv, g) for g in groundable) for cv in candidates):
            reasons.append(f"rationale cites number {tok!r} that is not grounded in the evidence")

    # Route-id-shaped tokens (contain a digit and a hyphen) must be real routes.
    for tok in _WORD_RE.findall(text):
        looks_like_route_id = "-" in tok and any(ch.isdigit() for ch in tok)
        if looks_like_route_id and tok not in known_ids:
            reasons.append(f"rationale references route {tok!r} that is not among the candidates")

    return reasons


def verify(output: JudgmentOutput, packet: EvidencePacket) -> VerificationResult:
    """Verify a parsed judgment against its evidence packet (deterministic)."""
    reasons: list[str] = []
    feasible_ids = set(packet.feasible_route_ids)

    # Layer 1: decision / pick coherence.
    pick = output.recommended_route_id
    if pick is not None and pick not in feasible_ids:
        reasons.append(
            f"recommended_route_id {pick!r} is not in the feasible set {sorted(feasible_ids)}"
        )
    if output.decision is JudgmentDecision.RECOMMEND and pick is None:
        reasons.append("decision is RECOMMEND but no recommended_route_id was given")

    # Layer 2: structured citations (exact).
    for c in output.fact_citations:
        problem = _check_fact_citation(packet, c)
        if problem:
            reasons.append(problem)
    for c in output.comparison_citations:
        problem = _check_comparison_citation(packet, c)
        if problem:
            reasons.append(problem)

    # Layer 3: tolerant prose scan.
    reasons.extend(_scan_prose(packet, output))

    return VerificationResult(ok=not reasons, reasons=reasons)
