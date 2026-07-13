"""
The route-slot choice contract. The model returns::

    {
      "chosen_index": 2,
      "rationale": "why this route-slot is the best overall pick",
      "citations": [
        {"index": 2, "field": "slot_availability", "value": 0.91}
      ]
    }

`parse_route_slot_choice` is strict about SHAPE only; whether the index is valid
and the cited values are correct is the verifier's job (`verifier.py`).
"""

from __future__ import annotations

from dataclasses import dataclass, field


class RouteSlotChoiceParseError(ValueError):
    """Raised when a raw route-slot-choice dict can't be parsed into the schema."""


@dataclass(frozen=True)
class RouteSlotCitation:
    index: int
    field: str
    value: float


@dataclass
class RouteSlotChoice:
    chosen_index: int
    rationale: str
    citations: list[RouteSlotCitation] = field(default_factory=list)


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


def parse_route_slot_choice(raw: object) -> RouteSlotChoice:
    _require(isinstance(raw, dict), "route-slot choice must be a JSON object")
    _require("chosen_index" in raw, "missing 'chosen_index'")
    chosen_index = _as_int(raw["chosen_index"], "chosen_index")

    rationale = str(raw.get("rationale", "")).strip()
    _require(bool(rationale), "rationale must be a non-empty string")

    citations: list[RouteSlotCitation] = []
    for i, c in enumerate(raw.get("citations") or []):
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
    return RouteSlotChoice(chosen_index=chosen_index, rationale=rationale, citations=citations)
