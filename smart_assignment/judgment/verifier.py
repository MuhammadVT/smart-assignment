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
       route no matter how the model reasons);
     - a RECOMMEND must be *supported*: at least one citation (fact or
       comparison) must reference the picked route on a route-varying fact.
       Without this, a model could "ground" a pick entirely with true facts
       about a different route, or with constants that are identical for every
       candidate (order size, config ceilings) — citation padding.

  2. Structured-citation grounding (primary, exact)
     - every `FactCitation` must resolve to a real fact in the packet and match
       its value. Percent-vs-fraction is normalized ONLY for fraction-valued
       fields (so "0.87" and "87%" both pass against a stored
       utilization_after of 0.87) — never for counts/distances/minutes, where
       "value/100" would let a claim that is off by 100x slip through (e.g.
       remaining_capacity_after=4000 against a stored 40);
     - every `ComparisonCitation` must compare two *different* routes (a route
       compared against itself is trivially "equal" — padding, not grounding)
       and be arithmetically true of the two cited facts.

  3. Free-text prose scan (secondary, tolerant)
     - numeric tokens (including "1,234"-style thousands), route-ids,
       "route <id>" mentions, day names, and HH:MM clock times that appear in
       the rationale must be grounded in the packet. This is deliberately
       *tolerant* (percent phrasings are normalized, trivially small bare
       counts are ignored) so a faithful paraphrase like "roughly 87%" passes
       while an invented "91%" that appears nowhere in the packet fails.
       Percent normalization in prose only applies against fraction-scale
       packet values, and never to a token carrying a concrete unit ("cases",
       "miles", "minutes") — a unit-bearing figure must match a packet value
       at face value, so "84 miles" can't launder through a stored 0.84.

Known residual limits (deterministically uncheckable, mitigated elsewhere):
a rationale can attach a *correct* number to the wrong noun ("only 0.57
utilization" quoting the reference score), and a comparison citation can be
true yet support a different sentence than the one the model wrote. Those are
semantic, not arithmetic, gaps — the RECOMMEND-support rule, the resample
consensus in `judge.py`, and the human escalation path bound their blast
radius.

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
# are serialized at <=4 dp and a whole-percent paraphrase of a fraction ("87%"
# for 0.8712) is off by at most 0.005, so this absorbs faithful rounding while
# rejecting a *neighboring but different* number (the old 0.02 let "88%" pass
# against a stored 0.87).
_TOL = 0.005

# Fields whose values are fractions of 1 (shares/ratios). Only these may be
# cited in percent form; for every other field a citation must match the stored
# value at face value.
_FRACTION_FIELDS = frozenset(
    {"utilization_after", "capacity_ceiling", "reference_weighted_score"}
)

# Facts that actually differ between routes. A RECOMMEND must be backed by a
# citation on one of these for the picked route — citing order_quantity_cases
# or a config constant is true for every candidate and supports nothing.
_ROUTE_VARYING_FACT_KEYS = tuple(
    k
    for k in NUMERIC_FACT_KEYS
    if k not in ("cluster_reference_miles", "capacity_ceiling", "order_quantity_cases")
)

# Fraction-scale packet values (utilization can legitimately exceed 1 for an
# overloaded infeasible route, e.g. 1.46 -> "146%").
_FRACTION_SCALE_MAX = 2.0

_DAY_CODES = ("MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN")
_DAY_NAME_TO_CODE = {
    "monday": "MON",
    "tuesday": "TUE",
    "wednesday": "WED",
    "thursday": "THU",
    "friday": "FRI",
    "saturday": "SAT",
    "sunday": "SUN",
}


@dataclass
class VerificationResult:
    ok: bool
    reasons: list[str] = field(default_factory=list)

    def as_feedback(self) -> str:
        return "; ".join(self.reasons)


def _facts_for(packet: EvidencePacket, route_id: str) -> dict | None:
    cand = packet.candidate_dict(route_id)
    return cand.get("facts") if cand else None


def _fact_value_matches(field_name: str, cited: float, actual: float) -> bool:
    if abs(cited - actual) <= _TOL:
        return True
    # Percent phrasing of a fraction-valued fact: "87" (or "87%") for a stored
    # 0.87. Gated on the field being a fraction and the cited value actually
    # looking like a percent, so a count/distance can never shift magnitude.
    if field_name in _FRACTION_FIELDS and cited > 1.5:
        return abs(cited / 100.0 - actual) <= _TOL
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
    if not _fact_value_matches(c.field, float(c.value), float(actual)):
        return f"{c.route_id}.{c.field}={actual} but citation claims {c.value}"
    return None


def _check_comparison_citation(packet: EvidencePacket, c: ComparisonCitation) -> str | None:
    if c.route_id_a == c.route_id_b:
        return (
            f"comparison cites route {c.route_id_a!r} against itself -- "
            f"a comparison must reference two different routes"
        )
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


def _pick_is_cited(output: JudgmentOutput) -> bool:
    """True when at least one citation backs the recommended route with a
    route-varying fact (see `_ROUTE_VARYING_FACT_KEYS`)."""
    pick = output.recommended_route_id
    for c in output.fact_citations:
        if c.route_id == pick and c.field in _ROUTE_VARYING_FACT_KEYS:
            return True
    for c in output.comparison_citations:
        if pick in (c.route_id_a, c.route_id_b) and c.field in _ROUTE_VARYING_FACT_KEYS:
            return True
    return False


def _collect_groundable_numbers(packet: EvidencePacket) -> list[float]:
    nums: list[float] = []
    for cand in packet.feasible_candidates + packet.infeasible_candidates:
        for v in (cand.get("facts") or {}).values():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                nums.append(float(v))
    nums.append(float(packet.customer.get("order_quantity_cases", 0)))
    return nums


def _collect_allowed_days(packet: EvidencePacket) -> set[str]:
    days = {
        c.get("day")
        for c in packet.feasible_candidates + packet.infeasible_candidates
        if c.get("day")
    }
    pref = packet.customer.get("preferred_slot")
    if isinstance(pref, str) and pref.split() and pref.split()[0] in _DAY_CODES:
        days.add(pref.split()[0])
    return days


def _collect_allowed_times(packet: EvidencePacket) -> set[tuple[int, int]]:
    """Every HH:MM that appears in a candidate window or the preferred slot."""
    times: set[tuple[int, int]] = set()
    sources = [
        c.get("window") for c in packet.feasible_candidates + packet.infeasible_candidates
    ]
    pref = packet.customer.get("preferred_slot")
    if isinstance(pref, str):
        sources.append(pref)
    for text in sources:
        if isinstance(text, str):
            for m in _TIME_RE.finditer(text):
                times.add((int(m.group(1)), int(m.group(2))))
    return times


# "1,234"-style thousands groups are captured whole, so a fabricated "1,150"
# can't pass by splitting into a discarded "1" and a coincidentally-grounded
# "150".
_NUMBER_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")
_TIME_RE = re.compile(r"\b(\d{1,2}):(\d{2})\b")
_WORD_RE = re.compile(r"[A-Za-z]+(?:-[A-Za-z0-9]+)*")
# What immediately follows a number decides how it may be normalized.
_PERCENT_AFTER_RE = re.compile(r"\s*(?:%|percent\b|pct\b)", re.IGNORECASE)
_UNIT_AFTER_RE = re.compile(r"\s*(?:cases?\b|miles?\b|minutes?\b|mins?\b)", re.IGNORECASE)
# Only full day names are scanned case-insensitively; 3-letter codes are
# matched case-sensitively in upper case so prose like "sat at 81%" or
# "sun exposure" can never false-positive.
_DAY_NAME_RE = re.compile(
    r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)s?\b", re.IGNORECASE
)
_DAY_CODE_RE = re.compile(r"\b(MON|TUE|WED|THU|FRI|SAT|SUN)\b")
# "route 4200" / "Rte #12" style mentions -- the id-shaped-token scan below
# misses these because the bare token has no hyphen.
_ROUTE_MENTION_RE = re.compile(r"\b(?:route|rte)s?\s+#?([A-Za-z0-9][A-Za-z0-9-]*)", re.IGNORECASE)


def _prose_number_grounded(
    val: float, groundable: list[float], is_percent: bool, has_unit: bool
) -> bool:
    if any(abs(val - g) <= _TOL for g in groundable):
        return True
    # Percent-vs-fraction, gated three ways: the token must look like a percent
    # (>1.5), the packet value must be fraction-scale, and the token must not
    # carry a concrete unit ("84 miles" may not ground against a stored 0.84).
    if val > 1.5 and (is_percent or not has_unit):
        return any(
            0.0 <= g <= _FRACTION_SCALE_MAX and abs(val / 100.0 - g) <= _TOL
            for g in groundable
        )
    return False


def _scan_prose(packet: EvidencePacket, output: JudgmentOutput) -> list[str]:
    """Tolerant scan: every load-bearing number / route-id / day / clock time in
    the rationale must ground in the evidence packet.

    Deliberately tolerant to avoid false rejections of faithful prose:
      - route-ids are scrubbed first (longest first, so one id being a prefix of
        another can't leave stray digits), so their digits are never mistaken
        for a numeric fact (e.g. the "4200" in "RTE-4200");
      - percent-vs-fraction is normalized (with the gates in
        `_prose_number_grounded`), so "roughly 87%" grounds against a stored
        0.87 while "84 miles" cannot ground against it;
      - trivially small bare integers (< 10, e.g. "the other 2 routes") are
        ignored -- UNLESS they carry a unit or percent sign ("only 5 cases of
        headroom", "5% utilization" are decision facts and must ground);
      - clock times must match a candidate window or the preferred slot, so an
        invented "13:00-15:00" fails while a quoted real window passes;
      - day names must belong to a candidate route or the preferred slot.
    A number that still can't be grounded (e.g. an invented "91%") fails.
    """
    reasons: list[str] = []
    text = output.rationale
    known_ids = {c["route_id"] for c in packet.feasible_candidates + packet.infeasible_candidates}
    groundable = _collect_groundable_numbers(packet)
    allowed_days = _collect_allowed_days(packet)
    allowed_times = _collect_allowed_times(packet)

    scrubbed = text
    for rid in sorted(known_ids, key=len, reverse=True):
        scrubbed = scrubbed.replace(rid, " ")

    # Clock times: verify against the real windows, then scrub them so their
    # digits aren't re-read as numeric facts.
    for m in _TIME_RE.finditer(scrubbed):
        if (int(m.group(1)), int(m.group(2))) not in allowed_times:
            reasons.append(
                f"rationale cites time {m.group(0)!r} that matches no candidate window "
                f"or preferred slot"
            )
    scrubbed = _TIME_RE.sub(" ", scrubbed)

    for m in _NUMBER_RE.finditer(scrubbed):
        tok = m.group(0)
        val = float(tok.replace(",", ""))
        tail = scrubbed[m.end():]
        is_percent = bool(_PERCENT_AFTER_RE.match(tail))
        has_unit = bool(_UNIT_AFTER_RE.match(tail))
        if "." not in tok and "," not in tok and val < 10 and not is_percent and not has_unit:
            continue  # generic small count ("the other 2 routes")
        if not _prose_number_grounded(val, groundable, is_percent, has_unit):
            reasons.append(f"rationale cites number {tok!r} that is not grounded in the evidence")

    for m in _DAY_NAME_RE.finditer(scrubbed):
        code = _DAY_NAME_TO_CODE[m.group(1).lower()]
        if code not in allowed_days:
            reasons.append(
                f"rationale mentions {m.group(0)!r} but no candidate route or preferred "
                f"slot falls on that day"
            )
    for m in _DAY_CODE_RE.finditer(scrubbed):
        if m.group(1) not in allowed_days:
            reasons.append(
                f"rationale mentions {m.group(1)!r} but no candidate route or preferred "
                f"slot falls on that day"
            )

    # Route-id-shaped tokens (contain a digit and a hyphen) must be real routes.
    for tok in _WORD_RE.findall(text):
        looks_like_route_id = "-" in tok and any(ch.isdigit() for ch in tok)
        if looks_like_route_id and tok not in known_ids:
            reasons.append(f"rationale references route {tok!r} that is not among the candidates")
    # "route 4200"-style mentions: real ids were scrubbed above, so whatever
    # digit-bearing token still follows "route"/"rte" must at least be part of
    # a real id ("route 4200" for RTE-4200 is a fine paraphrase; "route 40"
    # naming a nonexistent route is not, even if 40 grounds as a number).
    lowered_ids = [rid.lower() for rid in known_ids]
    for m in _ROUTE_MENTION_RE.finditer(scrubbed):
        tok = m.group(1)
        if any(ch.isdigit() for ch in tok) and not any(
            tok.lower() in rid for rid in lowered_ids
        ):
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
    if (
        output.decision is JudgmentDecision.RECOMMEND
        and pick is not None
        and not _pick_is_cited(output)
    ):
        reasons.append(
            f"decision is RECOMMEND but no citation backs {pick} with a route-specific "
            f"fact (one of: {', '.join(_ROUTE_VARYING_FACT_KEYS)}) -- cite the facts "
            f"the pick rests on"
        )

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
