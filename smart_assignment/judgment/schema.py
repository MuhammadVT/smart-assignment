"""
The judgment output contract — what the grounded-judgment LLM must return, and
how a raw dict is parsed into typed objects.

The model returns JSON shaped like::

    {
      "candidate_notes": [{"route_id": "RTE-4200", "note": "..."}],
      "recommended_route_id": "RTE-4200" | null,
      "decision": "RECOMMEND" | "ESCALATE",
      "confidence": "HIGH" | "LOW",
      "rationale": "free text explanation",
      "citations": [
        {"kind": "fact", "route_id": "RTE-4200",
         "field": "utilization_after", "value": 0.87},
        {"kind": "comparison", "field": "remaining_capacity_after",
         "route_id_a": "RTE-4200", "route_id_b": "RTE-4400",
         "relation": "greater" | "less" | "equal"}
      ]
    }

`parse_judgment` is intentionally strict about *shape* (raising `JudgmentParseError`
on anything it can't turn into the typed objects) but does NOT check whether the
cited values are actually correct — that grounding check is the verifier's job
(`verifier.py`), kept separate so a shape problem and a grounding problem are
distinguishable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class JudgmentParseError(ValueError):
    """Raised when a raw judgment dict can't be parsed into the typed schema."""


class JudgmentDecision(str, Enum):
    RECOMMEND = "RECOMMEND"
    ESCALATE = "ESCALATE"


class Confidence(str, Enum):
    HIGH = "HIGH"
    LOW = "LOW"


@dataclass(frozen=True)
class FactCitation:
    """A claim that `feasible/infeasible candidate[route_id].facts[field] == value`."""

    route_id: str
    field: str
    value: float


@dataclass(frozen=True)
class ComparisonCitation:
    """A claim comparing one numeric fact across two candidates."""

    field: str
    route_id_a: str
    route_id_b: str
    relation: str  # "greater" | "less" | "equal"


@dataclass(frozen=True)
class CandidateNote:
    route_id: str
    note: str


@dataclass
class JudgmentOutput:
    decision: JudgmentDecision
    confidence: Confidence
    rationale: str
    recommended_route_id: Optional[str] = None
    candidate_notes: list[CandidateNote] = field(default_factory=list)
    fact_citations: list[FactCitation] = field(default_factory=list)
    comparison_citations: list[ComparisonCitation] = field(default_factory=list)

    @property
    def is_confident_recommend(self) -> bool:
        return self.decision is JudgmentDecision.RECOMMEND and self.confidence is Confidence.HIGH


_VALID_RELATIONS = {"greater", "less", "equal"}


def _require(cond: bool, message: str) -> None:
    if not cond:
        raise JudgmentParseError(message)


def _as_float(raw: object, where: str) -> float:
    # Accept ints, floats, and numeric strings (incl. a trailing "%"), so a
    # model that writes "87%" or "0.87" both parse; the verifier normalizes
    # percent-vs-fraction when it checks the value against the packet.
    if isinstance(raw, bool):
        raise JudgmentParseError(f"{where}: expected a number, got a boolean")
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str):
        token = raw.strip().rstrip("%").strip()
        try:
            return float(token)
        except ValueError as exc:
            raise JudgmentParseError(f"{where}: {raw!r} is not numeric") from exc
    raise JudgmentParseError(f"{where}: expected a number, got {type(raw).__name__}")


def _parse_citation(raw: object, idx: int):
    _require(isinstance(raw, dict), f"citations[{idx}] must be an object")
    kind = str(raw.get("kind", "fact")).strip().lower()
    if kind == "comparison":
        for k in ("field", "route_id_a", "route_id_b", "relation"):
            _require(raw.get(k) is not None, f"citations[{idx}] (comparison) missing {k!r}")
        relation = str(raw["relation"]).strip().lower()
        _require(
            relation in _VALID_RELATIONS,
            f"citations[{idx}] relation must be one of {sorted(_VALID_RELATIONS)}",
        )
        return ComparisonCitation(
            field=str(raw["field"]).strip(),
            route_id_a=str(raw["route_id_a"]).strip(),
            route_id_b=str(raw["route_id_b"]).strip(),
            relation=relation,
        )
    # default: fact citation
    for k in ("route_id", "field", "value"):
        _require(raw.get(k) is not None, f"citations[{idx}] (fact) missing {k!r}")
    return FactCitation(
        route_id=str(raw["route_id"]).strip(),
        field=str(raw["field"]).strip(),
        value=_as_float(raw["value"], f"citations[{idx}].value"),
    )


def parse_judgment(raw: object) -> JudgmentOutput:
    """Parse a raw judgment dict (already JSON-decoded) into a `JudgmentOutput`.

    Raises `JudgmentParseError` on any structural problem. Does not check that
    citations are *correct* — that's `verifier.verify`.
    """
    _require(isinstance(raw, dict), "judgment must be a JSON object")

    decision_raw = str(raw.get("decision", "")).strip().upper()
    _require(
        decision_raw in JudgmentDecision.__members__,
        f"decision must be one of {list(JudgmentDecision.__members__)}, got {decision_raw!r}",
    )
    decision = JudgmentDecision[decision_raw]

    confidence_raw = str(raw.get("confidence", "")).strip().upper()
    _require(
        confidence_raw in Confidence.__members__,
        f"confidence must be one of {list(Confidence.__members__)}, got {confidence_raw!r}",
    )
    confidence = Confidence[confidence_raw]

    rationale = str(raw.get("rationale", "")).strip()
    _require(bool(rationale), "rationale must be a non-empty string")

    rec_raw = raw.get("recommended_route_id")
    recommended_route_id = None if rec_raw in (None, "", "null") else str(rec_raw).strip()

    notes: list[CandidateNote] = []
    for i, n in enumerate(raw.get("candidate_notes") or []):
        _require(isinstance(n, dict), f"candidate_notes[{i}] must be an object")
        rid, note = n.get("route_id"), n.get("note")
        if rid and note:
            notes.append(CandidateNote(route_id=str(rid).strip(), note=str(note).strip()))

    fact_citations: list[FactCitation] = []
    comparison_citations: list[ComparisonCitation] = []
    for i, c in enumerate(raw.get("citations") or []):
        parsed = _parse_citation(c, i)
        if isinstance(parsed, FactCitation):
            fact_citations.append(parsed)
        else:
            comparison_citations.append(parsed)

    return JudgmentOutput(
        decision=decision,
        confidence=confidence,
        rationale=rationale,
        recommended_route_id=recommended_route_id,
        candidate_notes=notes,
        fact_citations=fact_citations,
        comparison_citations=comparison_citations,
    )
