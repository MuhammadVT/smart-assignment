"""
The route-slot choice contract. The model returns a *structured* explanation, not
a one-liner, so an ops manager gets the rationale AND the trade-off behind the
pick::

    {
      "chosen_index": 2,
      "decision_summary": "Assign BT149361 · WED · 09:10-12:10.",
      "primary_reasons": [
        "Tightest fit of any option -- geographic_clustering 0.88.",
        "Slot fully open -- slot_availability 1.0; no tier-4/5 stop shares it."
      ],
      "key_tradeoff": "Option 0 clusters marginally tighter (0.91 vs 0.88) but its "
                      "slot is contended (0.55); trading a hair of clustering for an "
                      "open slot is the better overall assignment.",
      "runner_up": {"index": 0, "why_not": "Better clustering, but slot 0.55 crowds a "
                                            "high-tier incumbent."},
      "vs_deterministic_default": {"verdict": "DIVERGE", "note": "The blend picks on "
                                   "clustering alone; it can't see option 0's slot is "
                                   "contended."},
      "citations": [
        {"index": 2, "field": "slot_availability", "value": 1.0},
        {"index": 0, "field": "geographic_clustering", "value": 0.91}
      ]
    }

Only ``chosen_index`` is *actionable* (it selects a real route-slot from the
enumerated menu); every other field is grounded explanation. ``parse_route_slot_choice``
is strict about SHAPE only -- whether the index is valid, the verdict is consistent,
the trade-off is present when it's owed, and every stated number is grounded is the
verifier's job (`verifier.py`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class RouteSlotChoiceParseError(ValueError):
    """Raised when a raw route-slot-choice dict can't be parsed into the schema."""


# The two allowed verdicts for the model's self-assessment against the
# deterministic weighted default.
VERDICT_AGREE = "AGREE"
VERDICT_DIVERGE = "DIVERGE"
VALID_VERDICTS = (VERDICT_AGREE, VERDICT_DIVERGE)


class RSDecision(str, Enum):
    """The model's own recommend-vs-escalate call (only consulted when
    `Config.use_grounded_route_slot_escalation` is on). Kept local so the
    routeslot package stays decoupled from the judgment package."""

    RECOMMEND = "RECOMMEND"
    ESCALATE = "ESCALATE"


class RSConfidence(str, Enum):
    HIGH = "HIGH"
    LOW = "LOW"


@dataclass(frozen=True)
class RouteSlotCitation:
    index: int
    field: str
    value: float


@dataclass(frozen=True)
class RunnerUp:
    """The next-best route-slot and the specific fact that tips the pick away
    from it -- forces an explicit 'why not the alternative'."""

    index: int
    why_not: str


@dataclass(frozen=True)
class DefaultComparison:
    """The model's self-assessment against the deterministic weighted default:
    did it AGREE with the blend's pick, or DIVERGE (with a justification)?"""

    verdict: str
    note: str = ""


@dataclass
class RouteSlotChoice:
    chosen_index: int
    decision_summary: str
    primary_reasons: list[str] = field(default_factory=list)
    key_tradeoff: str = ""
    runner_up: Optional[RunnerUp] = None
    vs_deterministic_default: Optional[DefaultComparison] = None
    citations: list[RouteSlotCitation] = field(default_factory=list)
    # The model's own recommend-vs-escalate call + confidence. Only meaningful on
    # the grounded-escalation path (Config.use_grounded_route_slot_escalation);
    # they default to a confident RECOMMEND so the pick-only path (and its prompt,
    # which doesn't ask for them) is unaffected. ``chosen_index`` is always the
    # best option -- on ESCALATE it is the strongest-but-insufficient one the
    # specialist reviews.
    decision: RSDecision = RSDecision.RECOMMEND
    confidence: RSConfidence = RSConfidence.HIGH

    def prose_fields(self) -> list[str]:
        """Every free-text field the verifier's prose scan must ground -- so a
        number stated anywhere the ops manager reads it is checked, not just the
        citation list."""
        parts = [self.decision_summary, self.key_tradeoff, *self.primary_reasons]
        if self.runner_up is not None:
            parts.append(self.runner_up.why_not)
        if self.vs_deterministic_default is not None:
            parts.append(self.vs_deterministic_default.note)
        return [p for p in parts if p]


def _require(cond: bool, message: str) -> None:
    if not cond:
        raise RouteSlotChoiceParseError(message)


def _as_int(raw: object, where: str) -> int:
    if isinstance(raw, bool):
        raise RouteSlotChoiceParseError(f"{where}: expected an integer, got a boolean")
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str) and raw.strip().lstrip("-").isdigit():
        return int(raw.strip())
    if isinstance(raw, float) and raw.is_integer():
        return int(raw)
    raise RouteSlotChoiceParseError(f"{where}: {raw!r} is not an integer")


def _as_float(raw: object, where: str) -> float:
    if isinstance(raw, bool):
        raise RouteSlotChoiceParseError(f"{where}: expected a number, got a boolean")
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str):
        token = raw.strip().rstrip("%").strip()
        try:
            return float(token)
        except ValueError as exc:
            raise RouteSlotChoiceParseError(f"{where}: {raw!r} is not numeric") from exc
    raise RouteSlotChoiceParseError(f"{where}: expected a number, got {type(raw).__name__}")


def _as_nonempty_str(raw: object, where: str) -> str:
    text = str(raw).strip() if raw is not None else ""
    _require(bool(text), f"{where} must be a non-empty string")
    return text


def _parse_primary_reasons(raw: object) -> list[str]:
    _require(isinstance(raw, list), "'primary_reasons' must be a list")
    reasons = [str(r).strip() for r in raw if str(r).strip()]
    _require(bool(reasons), "'primary_reasons' must have at least one non-empty entry")
    return reasons


def _parse_runner_up(raw: object) -> Optional[RunnerUp]:
    if raw is None:
        return None
    _require(isinstance(raw, dict), "'runner_up' must be an object")
    _require(raw.get("index") is not None, "runner_up missing 'index'")
    return RunnerUp(
        index=_as_int(raw["index"], "runner_up.index"),
        why_not=_as_nonempty_str(raw.get("why_not"), "runner_up.why_not"),
    )


def _parse_default_comparison(raw: object) -> Optional[DefaultComparison]:
    if raw is None:
        return None
    _require(isinstance(raw, dict), "'vs_deterministic_default' must be an object")
    verdict = str(raw.get("verdict", "")).strip().upper()
    _require(
        verdict in VALID_VERDICTS,
        f"vs_deterministic_default.verdict must be one of {VALID_VERDICTS}, got {verdict!r}",
    )
    return DefaultComparison(verdict=verdict, note=str(raw.get("note", "")).strip())


def _parse_citations(raw: object) -> list[RouteSlotCitation]:
    citations: list[RouteSlotCitation] = []
    for i, c in enumerate(raw or []):
        _require(isinstance(c, dict), f"citations[{i}] must be an object")
        for k in ("index", "field", "value"):
            _require(c.get(k) is not None, f"citations[{i}] missing {k!r}")
        citations.append(
            RouteSlotCitation(
                index=_as_int(c["index"], f"citations[{i}].index"),
                field=str(c["field"]).strip(),
                value=_as_float(c["value"], f"citations[{i}].value"),
            )
        )
    return citations


def _parse_enum(raw: object, enum_cls, default, where: str):
    """Read an optional enum field. Absent/blank -> default (so the pick-only path,
    whose prompt doesn't ask for it, is unaffected); a present-but-invalid value is
    a shape error."""
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return default
    token = str(raw).strip().upper()
    _require(
        token in enum_cls.__members__,
        f"{where} must be one of {list(enum_cls.__members__)}, got {token!r}",
    )
    return enum_cls[token]


def parse_route_slot_choice(raw: object) -> RouteSlotChoice:
    _require(isinstance(raw, dict), "route-slot choice must be a JSON object")
    _require("chosen_index" in raw, "missing 'chosen_index'")
    _require("primary_reasons" in raw, "missing 'primary_reasons'")

    return RouteSlotChoice(
        chosen_index=_as_int(raw["chosen_index"], "chosen_index"),
        decision_summary=_as_nonempty_str(raw.get("decision_summary"), "decision_summary"),
        primary_reasons=_parse_primary_reasons(raw["primary_reasons"]),
        key_tradeoff=str(raw.get("key_tradeoff", "")).strip(),
        runner_up=_parse_runner_up(raw.get("runner_up")),
        vs_deterministic_default=_parse_default_comparison(raw.get("vs_deterministic_default")),
        citations=_parse_citations(raw.get("citations")),
        decision=_parse_enum(raw.get("decision"), RSDecision, RSDecision.RECOMMEND, "decision"),
        confidence=_parse_enum(
            raw.get("confidence"), RSConfidence, RSConfidence.HIGH, "confidence"
        ),
    )
