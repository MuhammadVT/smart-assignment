"""
Capture a complete, end-to-end execution trace of `root_agent` for ONE prospect.

Read-only with respect to the repository: it imports the real package, drives the
real ADK agent through a real `Runner`, and writes its output to a scratch
directory. It changes no project file and no session state that outlives the run.

What it captures, in one JSON document:

  * the resolved `Config` (every knob the decision actually used) and the model
    chosen for each role;
  * step 2 in full -- the geocode, the whole proximity ranking with the Top-N cut
    and any preferred-day keep marked, and per candidate route the committed
    stops, neighbor weights, time clusters, anchors and slot menu, each with the
    arithmetic spelled out as a string;
  * steps 3-4 -- every hard-constraint check and every (route, slot) factor value
    and weighted total, again with the substitution shown;
  * the evidence packet and the exact prompt text the grounded judgment call
    receives;
  * every LLM request/response on the conversation loop (via ADK model
    callbacks), the nested grounded call (by wrapping `shared.llm`), and the
    escalation-triage sub-agent's own traffic when that path fires;
  * the tool results, the final session state, and the reply the user sees.

Nothing here is re-derived by hand: the numbers in the trace come from the same
functions the product runs, so a page built from this file cannot drift from the
system it documents.

Usage
-----
    python .github/skills/prospect-trace/capture_trace.py \
        --address "1201 Lake Woodlands Dr, The Woodlands, TX 77380" \
        --cases 150 --day THU --window 09:00-12:00 \
        --out .trace/woodlands

    # wrap a body-only HTML fragment into a standalone page (non-Claude harnesses)
    python .github/skills/prospect-trace/capture_trace.py \
        --wrap-html .trace/woodlands/page.fragment.html \
        --out-html .trace/woodlands/page.html
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

SKILL_DIR = Path(__file__).resolve().parent
REPO_ROOT = SKILL_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("capture_trace")


# --------------------------------------------------------------------------
# standalone-page wrapper (for harnesses with no artifact publishing)
# --------------------------------------------------------------------------


def wrap_html(fragment_path: Path, out_path: Path) -> None:
    """Wrap a body-only fragment into a standalone HTML document.

    The Claude Code artifact publisher supplies its own document skeleton, so the
    page is authored as a fragment. Every other harness needs a real file to open
    in a browser -- this adds the skeleton (and the theme toggle the artifact
    host would otherwise provide) without touching the fragment's markup."""
    import re

    fragment = fragment_path.read_text(encoding="utf-8")

    # Lift the fragment's <title> into <head>. `document.title` would resolve it
    # either way, but a title element in the body is invalid HTML and the file is
    # meant to be opened, inspected and shared as-is.
    title = "Prospect trace"
    match = re.search(r"<title>(.*?)</title>", fragment, re.S | re.I)
    if match:
        title = match.group(1).strip()
        fragment = fragment[: match.start()] + fragment[match.end():]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "<!doctype html>\n"
        '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{title}</title>\n"
        "</head>\n<body>\n" + fragment.lstrip("\n") + "\n</body>\n</html>\n",
        encoding="utf-8",
    )
    print(f"wrote standalone page: {out_path}")


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001 - provenance is best-effort
        return None


def _part_repr(part: Any) -> dict:
    """A JSON-safe view of one ADK content part (text / call / response)."""
    out: dict = {}
    if getattr(part, "text", None):
        out["text"] = part.text
    call = getattr(part, "function_call", None)
    if call is not None:
        out["function_call"] = {"name": call.name, "args": dict(call.args or {})}
    resp = getattr(part, "function_response", None)
    if resp is not None:
        out["function_response"] = {"name": resp.name, "response": resp.response}
    return out


def _content_repr(content: Any) -> list:
    return [_part_repr(p) for p in (getattr(content, "parts", None) or [])]


def _usage(response: Any) -> dict:
    meta = getattr(response, "usage_metadata", None)
    if meta is None:
        return {}
    return {
        "prompt_tokens": getattr(meta, "prompt_token_count", None),
        "output_tokens": getattr(meta, "candidates_token_count", None),
    }


# --------------------------------------------------------------------------
# the capture
# --------------------------------------------------------------------------


class TraceCapture:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.trace: dict = {}
        self.llm_calls: list = []          # nested grounded calls (shared.llm)
        self.agent_traffic: list = []      # root_agent model traffic
        self.triage_traffic: list = []     # escalation-triage sub-agent traffic

    # -- prospect ---------------------------------------------------------

    def build_customer(self):
        from smart_assignment.shared.models import (
            CustomerProfile,
            DayOfWeek,
            PreferredSlot,
        )
        from smart_assignment.shared.timeutils import parse_time

        slot = None
        if self.args.day and self.args.window:
            start, _, end = self.args.window.partition("-")
            slot = PreferredSlot(
                DayOfWeek(self.args.day.strip().upper()),
                (parse_time(start.strip()), parse_time(end.strip())),
            )
        return CustomerProfile(
            name=self.args.name or "New prospect",
            address=self.args.address,
            order_quantity_cases=self.args.cases,
            customer_number=self.args.customer_number,
            preferred_slot=slot,
        )

    def user_message(self) -> str:
        """The sentence a sales rep would type -- what the agent actually gets."""
        if self.args.message:
            return self.args.message
        parts = [self.args.address, f"{self.args.cases} cases"]
        if self.args.day and self.args.window:
            parts.append(f"{self.args.day.strip().upper()} {self.args.window.strip()}")
        return ", ".join(parts)

    # -- config -----------------------------------------------------------

    def capture_config(self) -> None:
        from smart_assignment.shared.config import DEFAULT_CONFIG

        cfg = DEFAULT_CONFIG
        self.trace["config"] = {
            k: (v if isinstance(v, (str, int, float, bool, type(None))) else str(v))
            for k, v in vars(cfg).items()
        }
        self.trace["resolved_models"] = {
            role: cfg.resolved_model(role)
            for role in ("root_agent", "judgment", "triage", "address_resolve")
        }

    # -- steps 1-4, with the arithmetic ------------------------------------

    def capture_deterministic(self, customer) -> None:
        """Run intake -> geo -> evaluate directly, recording every intermediate
        quantity. Runs BEFORE the agent so a geocode failure is diagnosed here
        rather than only showing up as a tool error."""
        from smart_assignment.integrations.route_capacity_client import (
            fetch_candidate_routes,
        )
        from smart_assignment.pipeline import (
            _preferred_day_candidate,
            evaluate_candidates,
            intake,
        )
        from smart_assignment.shared.config import DEFAULT_CONFIG as cfg
        from smart_assignment.shared.constraints import (
            applicable_preferred_window,
            build_context,
            service_distance_limit,
        )
        from smart_assignment.shared.geo import GeocodingError, haversine_miles
        from smart_assignment.shared import slot_selection as ss
        from smart_assignment.shared.timeutils import (
            fmt_time,
            fmt_window,
            overlap_minutes,
            window_midpoint,
        )
        from smart_assignment.tools.slot_recommendation import _GEOCODER

        intake(customer)
        self.trace["intake"] = {
            "name": customer.name,
            "address": customer.address,
            "order_quantity_cases": customer.order_quantity_cases,
            "customer_number": customer.customer_number,
            "preferred_day": self.args.day,
            "preferred_window": self.args.window,
        }

        t0 = time.time()
        all_routes = fetch_candidate_routes()
        self.trace["universe"] = {
            "total_routes_in_source": len(all_routes),
            "fetch_seconds": round(time.time() - t0, 2),
        }

        t0 = time.time()
        try:
            customer.location = _GEOCODER.geocode(customer.address)
        except GeocodingError as exc:
            self.trace["geo"] = {
                "ok": False,
                "geocoder": type(_GEOCODER).__name__,
                "error": f"{type(exc).__name__}: {exc}",
                "seconds": round(time.time() - t0, 2),
            }
            self.trace["deterministic_skipped"] = (
                "The address did not geocode, so steps 2-5 could not be re-derived "
                "here. The agent run below shows how the workflow handled it."
            )
            return
        geo_seconds = round(time.time() - t0, 2)

        ranked = sorted(
            all_routes, key=lambda r: haversine_miles(customer.location, r.service_center)
        )
        top_n = ranked[: cfg.top_n_candidate_routes]
        extra = (
            _preferred_day_candidate(customer, ranked, top_n, cfg)
            if cfg.use_preferred_day_candidate
            else None
        )
        candidates = list(top_n) + ([extra] if extra is not None else [])

        self.trace["geo"] = {
            "ok": True,
            "geocoder": type(_GEOCODER).__name__,
            "latitude": customer.location.latitude,
            "longitude": customer.location.longitude,
            "seconds": geo_seconds,
            "top_n": cfg.top_n_candidate_routes,
            "preferred_day_keep": (
                {
                    "route_id": extra.route_id,
                    "name": extra.name,
                    "day": extra.day.value,
                    "rank": ranked.index(extra) + 1,
                    "distance_miles": round(
                        haversine_miles(customer.location, extra.service_center), 3
                    ),
                    "service_distance_limit": service_distance_limit(extra, cfg),
                }
                if extra is not None
                else None
            ),
            "ranking": [
                {
                    "rank": i + 1,
                    "route_id": r.route_id,
                    "name": r.name,
                    "day": r.day.value,
                    "distance_miles": round(
                        haversine_miles(customer.location, r.service_center), 3
                    ),
                    "role": (
                        "top_n"
                        if r in top_n
                        else "preferred_day_keep"
                        if extra is not None and r.route_id == extra.route_id
                        else "cut"
                    ),
                }
                for i, r in enumerate(ranked[: max(12, cfg.top_n_candidate_routes + 8)])
            ],
        }
        if customer.preferred_slot is not None:
            day = customer.preferred_slot.day
            self.trace["geo"]["preferred_day_routes_in_source"] = sum(
                1 for r in all_routes if r.day == day
            )

        # --- per-route arithmetic ---------------------------------------
        routes_detail = []
        for route in candidates:
            ctx = build_context(customer, route, cfg)
            pref_window = applicable_preferred_window(customer, route)
            entry: dict = {
                "route_id": route.route_id,
                "name": route.name,
                "day": route.day.value,
                "in_candidate_set_because": (
                    "preferred_day_keep"
                    if extra is not None and route.route_id == extra.route_id
                    else "top_n"
                ),
                "distance_to_center_miles": haversine_miles(
                    customer.location, route.service_center
                ),
                "service_distance_limit": service_distance_limit(route, cfg),
                "capacity_cases": route.vehicle_capacity_cases,
                "committed_volume_cases": route.committed_volume_cases,
                "order_cases": customer.order_quantity_cases,
                "remaining_after": ctx.remaining_capacity_after,
                "utilization_after": ctx.utilization_after,
                "utilization_calc": (
                    f"({route.committed_volume_cases} + {customer.order_quantity_cases})"
                    f" / {route.vehicle_capacity_cases} = {ctx.utilization_after:.6f}"
                ),
                "avg_stop_distance_miles": ctx.avg_stop_distance_miles,
                "geo_clustering_calc": (
                    f"1 - {ctx.avg_stop_distance_miles:.4f} / {cfg.cluster_reference_miles}"
                ),
                "preferred_window_applies": (
                    fmt_window(pref_window) if pref_window else None
                ),
                "best_preference_overlap_minutes": ctx.window_overlap_minutes,
                "committed_stops": [
                    {
                        "customer_number": s.customer_number,
                        "tier": s.customer_tier,
                        "harm_weight": cfg.tier_harm_weight(s.customer_tier),
                        "window": (
                            fmt_window(s.delivery_time_window)
                            if s.delivery_time_window
                            else None
                        ),
                        "reference_time": (
                            fmt_time(window_midpoint(s.delivery_time_window))
                            if s.delivery_time_window
                            else None
                        ),
                        "distance_miles": haversine_miles(customer.location, s.location),
                    }
                    for s in route.committed_stops
                ],
                "neighbors": [],
                "clusters": [],
                "menu": [],
            }

            neighbors = ss.nearest_neighbors(
                customer.location,
                route.committed_stops,
                cfg.slot_neighbor_count,
                cfg.slot_neighbor_max_miles,
            )
            timed = [
                (n, ss._minutes(ref))
                for n in neighbors
                if (ref := ss.stop_reference_time(n.stop)) is not None
            ]
            total_w = sum(ss._weight(n.distance_miles) for n, _ in timed)
            entry["total_inverse_distance_weight"] = total_w
            for n, minutes in timed:
                entry["neighbors"].append(
                    {
                        "customer_number": n.stop.customer_number,
                        "distance_miles": n.distance_miles,
                        "inverse_distance_weight": ss._weight(n.distance_miles),
                        "reference_minutes": minutes,
                        "reference_time": fmt_time(ss._time_from_minutes(minutes)),
                    }
                )
            for cluster in ss._cluster_by_time(timed, cfg.slot_cluster_gap_minutes):
                w_sum = sum(ss._weight(n.distance_miles) for n, _ in cluster)
                anchor = sum(ss._weight(n.distance_miles) * m for n, m in cluster) / w_sum
                window = ss.centered_window(anchor, cfg.slot_window_minutes)
                entry["clusters"].append(
                    {
                        "members": [n.stop.customer_number for n, _ in cluster],
                        "member_times": [
                            fmt_time(ss._time_from_minutes(m)) for _, m in cluster
                        ],
                        "weight_sum": w_sum,
                        "anchor_minutes": anchor,
                        "anchor_time": fmt_time(ss._time_from_minutes(anchor)),
                        "window": fmt_window(window),
                        "fit_calc": f"{w_sum:.6f} / {total_w:.6f} = {w_sum / total_w:.6f}",
                    }
                )
            for option in ctx.available_slots:
                overlapping = [
                    {
                        "customer_number": s.customer_number,
                        "tier": s.customer_tier,
                        "harm": cfg.tier_harm_weight(s.customer_tier),
                        "overlap_minutes": overlap_minutes(
                            option.window, s.delivery_time_window
                        ),
                    }
                    for s in route.committed_stops
                    if s.delivery_time_window is not None
                    and overlap_minutes(option.window, s.delivery_time_window) > 0
                ]
                harm = sum(x["harm"] for x in overlapping)
                entry["menu"].append(
                    {
                        "window": fmt_window(option.window),
                        "anchor_time": (
                            fmt_time(option.anchor_time) if option.anchor_time else None
                        ),
                        "fit_score": option.fit_score,
                        "committed_overlap": option.committed_overlap,
                        "basis": option.basis,
                        "menu_quality_rank_score": ss._quality(option, cfg),
                        "overlapping_stops": overlapping,
                        "harm_sum": harm,
                        "openness_calc": f"1 / (1 + {harm:.2f}) = {1 / (1 + harm):.6f}",
                    }
                )
            routes_detail.append(entry)
        self.trace["routes"] = routes_detail

        # --- constraints + scores ---------------------------------------
        evaluations = evaluate_candidates(customer, candidates, cfg)
        self.trace["evaluations"] = []
        totals = []
        for ev in evaluations:
            self.trace["evaluations"].append(
                {
                    "route_id": ev.route.route_id,
                    "name": ev.route.name,
                    "day": ev.route.day.value,
                    "feasible": ev.feasible,
                    "distance_miles": round(ev.distance_miles, 3),
                    "utilization_after": round(ev.utilization_after, 4),
                    "remaining_capacity_after": ev.remaining_capacity_after,
                    "constraints": [
                        {"name": c.name, "passed": c.passed, "detail": c.detail}
                        for c in ev.constraint_outcomes
                    ],
                    "chosen_window": (
                        fmt_window(ev.chosen_window) if ev.chosen_window else None
                    ),
                    "window_basis": ev.window_basis,
                }
            )
            for scored in ev.scored_slots:
                terms = [
                    {
                        "name": f.name,
                        "weight": f.weight,
                        "value": round(f.value, 4),
                        "weighted": f.weight * f.value,
                        "detail": f.detail,
                    }
                    for f in scored.factor_scores
                ]
                weight_sum = sum(t["weight"] for t in terms) or 1.0
                numerator = sum(t["weighted"] for t in terms)
                totals.append(
                    {
                        "route_id": ev.route.route_id,
                        "route_name": ev.route.name,
                        "day": ev.route.day.value,
                        "window": fmt_window(scored.slot.window),
                        "basis": scored.slot.basis,
                        "feasible": ev.feasible,
                        "terms": terms,
                        "calc": (
                            " + ".join(
                                f"{t['weight']}*{t['value']:.4f}" for t in terms
                            )
                            + f" = {numerator:.6f}; / {weight_sum} = {numerator / weight_sum:.6f}"
                        ),
                        "total_score": scored.total_score,
                    }
                )
        totals.sort(key=lambda t: (-t["total_score"]))
        self.trace["route_slot_totals"] = totals

        # --- the evidence packet + the exact judgment prompt -------------
        from smart_assignment.routeslot.evidence import build_route_slot_packet
        from smart_assignment.routeslot.prompts import (
            build_route_slot_decision_prompt,
            build_route_slot_prompt,
        )

        if cfg.use_grounded_route_slot_escalation:
            packet = build_route_slot_packet(
                customer,
                evaluations,
                cfg,
                auto_assign_threshold=cfg.route_slot_score_threshold,
            )
            prompt = build_route_slot_decision_prompt(packet)
            path = "grounded_escalation (the model decides recommend vs escalate)"
        else:
            packet = build_route_slot_packet(
                customer, evaluations, cfg, min_score=cfg.route_slot_score_threshold
            )
            prompt = build_route_slot_prompt(packet)
            path = "threshold-gated (the bar decides; the model only picks)"
        self.trace["packet"] = packet.as_dict()
        self.trace["decision_path"] = path
        self.trace["judgment_prompt"] = prompt
        self.trace["gate"] = {
            "threshold": cfg.route_slot_score_threshold,
            "route_slots_enumerated": len(totals),
            "feasible": sum(1 for t in totals if t["feasible"]),
            "eligible": len(packet.options),
        }

    # -- the live agent turn ----------------------------------------------

    def _install_llm_wrappers(self) -> None:
        """Record every nested grounded call (the judgment pick, a batch-style
        triage compose, an address resolution) with its full prompt and reply."""
        import smart_assignment.shared.llm as shared_llm

        original_tool_call = shared_llm.generate_tool_call
        original_text = shared_llm.generate_text

        def wrapped_tool_call(config, prompt, tool, role=None):
            started = time.time()
            record: dict = {
                "kind": "generate_tool_call",
                "role": role,
                "model": (
                    config.sage_model if config.llm_backend == "sage" else config.model
                ),
                "tool_offered": tool.get("name"),
                "prompt": prompt,
            }
            try:
                call_args, text = original_tool_call(config, prompt, tool, role=role)
                record.update(call_args=call_args, text=text)
                return call_args, text
            except Exception as exc:  # noqa: BLE001 - record then re-raise
                record["error"] = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                record["seconds"] = round(time.time() - started, 2)
                self.llm_calls.append(record)

        def wrapped_text(config, prompt, role=None):
            started = time.time()
            record: dict = {
                "kind": "generate_text",
                "role": role,
                "model": (
                    config.sage_model if config.llm_backend == "sage" else config.model
                ),
                "prompt": prompt,
            }
            try:
                text = original_text(config, prompt, role=role)
                record["text"] = text
                return text
            except Exception as exc:  # noqa: BLE001
                record["error"] = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                record["seconds"] = round(time.time() - started, 2)
                self.llm_calls.append(record)

        shared_llm.generate_tool_call = wrapped_tool_call
        shared_llm.generate_text = wrapped_text

    def _model_callbacks(self, sink: list):
        """ADK before/after model callbacks that append to `sink`."""

        def before_model(callback_context, llm_request):
            sink.append(
                {
                    "direction": "request",
                    "model": getattr(llm_request, "model", None),
                    "system_instruction": getattr(
                        getattr(llm_request, "config", None), "system_instruction", None
                    ),
                    "tools_declared": sorted((llm_request.tools_dict or {}).keys()),
                    "contents": [
                        {"role": c.role, "parts": _content_repr(c)}
                        for c in (llm_request.contents or [])
                    ],
                    "at": round(time.time(), 3),
                }
            )
            return None

        def after_model(callback_context, llm_response):
            sink.append(
                {
                    "direction": "response",
                    "parts": _content_repr(getattr(llm_response, "content", None)),
                    "usage": _usage(llm_response),
                    "at": round(time.time(), 3),
                }
            )
            return None

        return before_model, after_model

    async def run_agent(self) -> None:
        from google.adk.runners import Runner
        from google.adk.sessions import InMemorySessionService
        from google.genai import types

        from smart_assignment.agent import root_agent

        before, after = self._model_callbacks(self.agent_traffic)
        root_agent.before_model_callback = before
        root_agent.after_model_callback = after

        # The escalation-triage AgentTool wraps its own LlmAgent, whose traffic
        # never passes through root_agent's callbacks. Instrument it too, so an
        # escalation trace shows the brief being drafted rather than a gap.
        for tool in root_agent.tools:
            sub_agent = getattr(tool, "agent", None)
            if sub_agent is not None:
                t_before, t_after = self._model_callbacks(self.triage_traffic)
                sub_agent.before_model_callback = t_before
                # The triage agent already uses after_model_callback to finalize
                # its brief; ADK accepts a list, so append rather than replace.
                existing = sub_agent.after_model_callback
                sub_agent.after_model_callback = (
                    [existing, t_after] if existing else t_after
                )

        self.trace["agent"] = {
            "name": root_agent.name,
            "description": root_agent.description,
            "instruction": root_agent.instruction,
            "instruction_chars": len(root_agent.instruction or ""),
            "tools": [getattr(t, "name", type(t).__name__) for t in root_agent.tools],
        }

        sessions = InMemorySessionService()
        await sessions.create_session(
            app_name="prospect_trace", user_id="trace", session_id="s1"
        )
        runner = Runner(
            app_name="prospect_trace", agent=root_agent, session_service=sessions
        )

        events = []
        started = time.time()
        async for event in runner.run_async(
            user_id="trace",
            session_id="s1",
            new_message=types.Content(
                role="user", parts=[types.Part(text=self.user_message())]
            ),
        ):
            events.append(
                {
                    "author": event.author,
                    "parts": _content_repr(getattr(event, "content", None)),
                    "is_final": event.is_final_response(),
                }
            )
        self.trace["agent_seconds"] = round(time.time() - started, 2)
        self.trace["events"] = events

        session = await sessions.get_session(
            app_name="prospect_trace", user_id="trace", session_id="s1"
        )
        self.trace["session_state"] = dict(session.state)

        finals = [
            part["text"]
            for event in events
            if event["is_final"]
            for part in event["parts"]
            if part.get("text")
        ]
        self.trace["final_reply"] = "\n".join(finals) if finals else None

    # -- verification replay ----------------------------------------------

    def capture_verification(self) -> None:
        """Re-run the deterministic verifier over the grounded reply that was
        actually returned, so the trace records the verdict the decision layer
        reached rather than an assumption about it."""
        judgment = next(
            (c for c in self.llm_calls if c.get("role") == "judgment" and "error" not in c),
            None,
        )
        if judgment is None or "packet" not in self.trace:
            return
        try:
            from smart_assignment.routeslot.evidence import build_route_slot_packet  # noqa: F401
            from smart_assignment.routeslot.llm import _extract_json
            from smart_assignment.routeslot.schema import parse_route_slot_choice
            from smart_assignment.routeslot.verifier import verify_choice

            raw = judgment.get("call_args") or _extract_json(judgment.get("text") or "")
            choice = parse_route_slot_choice(raw)
            packet = self._rebuild_packet()
            result = verify_choice(choice, packet)
            self.trace["verification"] = {
                "ok": result.ok,
                "feedback": None if result.ok else result.as_feedback(),
                "chosen_index": choice.chosen_index,
                "decision": choice.decision.value,
                "confidence": choice.confidence.value,
                "citations": len(choice.citations or []),
                "answered_via": "tool_call" if judgment.get("call_args") else "text_json",
            }
        except Exception as exc:  # noqa: BLE001 - a replay failure is itself a finding
            self.trace["verification"] = {"error": f"{type(exc).__name__}: {exc}"}

    def _rebuild_packet(self):
        """The same packet the decision built, reconstructed for the replay."""
        from smart_assignment.integrations.route_capacity_client import (
            fetch_candidate_routes,
        )
        from smart_assignment.pipeline import evaluate_candidates, geo_lookup
        from smart_assignment.routeslot.evidence import build_route_slot_packet
        from smart_assignment.shared.config import DEFAULT_CONFIG as cfg
        from smart_assignment.tools.slot_recommendation import _GEOCODER

        customer = self.build_customer()
        candidates = geo_lookup(customer, fetch_candidate_routes(), _GEOCODER, cfg)
        evaluations = evaluate_candidates(customer, candidates, cfg)
        if cfg.use_grounded_route_slot_escalation:
            return build_route_slot_packet(
                customer,
                evaluations,
                cfg,
                auto_assign_threshold=cfg.route_slot_score_threshold,
            )
        return build_route_slot_packet(
            customer, evaluations, cfg, min_score=cfg.route_slot_score_threshold
        )

    # -- the ledger --------------------------------------------------------

    def capture_ledger(self) -> None:
        """Per-call tokens and latency, plus the split between model time and
        deterministic time -- the numbers the page's summary table reports."""
        ledger = []
        pending = None
        for frame in self.agent_traffic:
            if frame["direction"] == "request":
                pending = frame
            elif pending is not None:
                ledger.append(
                    {
                        "surface": "root_agent turn",
                        "prompt_tokens": frame["usage"].get("prompt_tokens"),
                        "output_tokens": frame["usage"].get("output_tokens"),
                        "seconds": round(frame["at"] - pending["at"], 2),
                    }
                )
                pending = None
        for frame in self.triage_traffic:
            if frame["direction"] == "response":
                ledger.append(
                    {
                        "surface": "escalation_triage turn",
                        "prompt_tokens": frame["usage"].get("prompt_tokens"),
                        "output_tokens": frame["usage"].get("output_tokens"),
                        "seconds": None,
                    }
                )
        for call in self.llm_calls:
            ledger.append(
                {
                    "surface": call["kind"],
                    "role": call.get("role"),
                    "prompt_chars": len(call.get("prompt") or ""),
                    "seconds": call.get("seconds"),
                    "error": call.get("error"),
                }
            )
        self.trace["ledger"] = ledger

    # -- orchestration -----------------------------------------------------

    def run(self) -> dict:
        import asyncio

        self.trace["meta"] = {
            "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": _git_commit(),
            "repo_root": str(REPO_ROOT),
            "user_message": self.user_message(),
        }
        self.capture_config()

        customer = self.build_customer()
        try:
            self.capture_deterministic(customer)
        except Exception as exc:  # noqa: BLE001 - never lose the agent run
            import traceback

            self.trace["deterministic_error"] = traceback.format_exc()
            logger.warning("Deterministic capture failed: %s", exc)

        self._install_llm_wrappers()
        try:
            asyncio.run(self.run_agent())
            self.trace["agent_run_ok"] = True
        except Exception:  # noqa: BLE001
            import traceback

            self.trace["agent_run_ok"] = False
            self.trace["agent_error"] = traceback.format_exc()

        self.trace["agent_llm_traffic"] = self.agent_traffic
        self.trace["triage_llm_traffic"] = self.triage_traffic
        self.trace["nested_llm_calls"] = self.llm_calls
        self.capture_verification()
        self.capture_ledger()
        return self.trace


# --------------------------------------------------------------------------
# digest
# --------------------------------------------------------------------------


def write_digest(trace: dict, path: Path) -> None:
    """A compact Markdown digest of the trace.

    The JSON is the source of truth and can run to a hundred kilobytes; this is
    the orientation pass -- what happened, in what order, with the headline
    numbers -- so a reader (or an agent authoring the page) can see the shape of
    the run before pulling details out of the JSON."""
    out: list[str] = []
    meta = trace.get("meta", {})
    cfg = trace.get("config", {})
    out.append(f"# Prospect trace — {meta.get('user_message', '')}")
    out.append("")
    out.append(f"- captured: {meta.get('captured_at')} at commit `{meta.get('git_commit')}`")
    out.append(
        f"- backend `{cfg.get('llm_backend')}` · model `{cfg.get('model')}` · "
        f"geocoder `{trace.get('geo', {}).get('geocoder')}`"
    )
    out.append(f"- decision path: {trace.get('decision_path', 'n/a')}")
    out.append(f"- agent run ok: {trace.get('agent_run_ok')} ({trace.get('agent_seconds')}s)")
    out.append("")

    flags = {k: v for k, v in cfg.items() if k.startswith("use_")}
    out.append("## Flags")
    out.append("")
    for key, value in sorted(flags.items()):
        out.append(f"- `{key}` = {value}")
    out.append("")

    geo = trace.get("geo") or {}
    if geo.get("ok"):
        out.append("## Step 2 — candidate routes")
        out.append("")
        out.append(f"{trace['universe']['total_routes_in_source']} routes in source; "
                   f"top_n = {geo['top_n']}.")
        keep = geo.get("preferred_day_keep")
        if keep:
            out.append(
                f"Preferred-day keep: **{keep['route_id']} {keep['name']} ({keep['day']})** "
                f"at {keep['distance_miles']} mi, global rank {keep['rank']}, "
                f"limit {keep['service_distance_limit']} mi."
            )
        out.append("")
        out.append("| rank | route | day | miles | role |")
        out.append("|---|---|---|---:|---|")
        for row in geo.get("ranking", []):
            out.append(
                f"| {row['rank']} | {row['route_id']} {row['name']} | {row['day']} "
                f"| {row['distance_miles']} | {row['role']} |"
            )
        out.append("")
    elif geo:
        out.append("## Step 2 — geocoding FAILED")
        out.append("")
        out.append(f"`{geo.get('error')}`")
        out.append("")

    if trace.get("evaluations"):
        out.append("## Step 3 — hard constraints")
        out.append("")
        for ev in trace["evaluations"]:
            verdict = "feasible" if ev["feasible"] else "INFEASIBLE"
            out.append(f"- **{ev['route_id']} {ev['name']} ({ev['day']})** — {verdict}")
            for c in ev["constraints"]:
                mark = "pass" if c["passed"] else "FAIL"
                out.append(f"  - {c['name']}: {mark} — {c['detail']}")
        out.append("")

    if trace.get("route_slot_totals"):
        out.append("## Step 4 — every (route, slot) scored")
        out.append("")
        out.append("| total | route | day | window | basis | feasible | calc |")
        out.append("|---:|---|---|---|---|---|---|")
        for t in trace["route_slot_totals"]:
            out.append(
                f"| {t['total_score']:.4f} | {t['route_id']} {t['route_name']} | {t['day']} "
                f"| {t['window']} | {t['basis']} | {t['feasible']} | `{t['calc']}` |"
            )
        out.append("")

    gate = trace.get("gate")
    if gate:
        out.append("## Step 5 — the gate")
        out.append("")
        out.append(
            f"{gate['route_slots_enumerated']} enumerated → {gate['feasible']} feasible "
            f"→ {gate['eligible']} eligible at the {gate['threshold']} bar."
        )
        out.append("")

    if trace.get("nested_llm_calls"):
        out.append("## Nested LLM calls")
        out.append("")
        for call in trace["nested_llm_calls"]:
            head = (
                f"- `{call['kind']}` role=`{call.get('role')}` "
                f"prompt={len(call.get('prompt') or '')} chars, {call.get('seconds')}s"
            )
            if call.get("error"):
                head += f" — ERROR {call['error']}"
            elif call.get("call_args"):
                head += " — answered via tool call"
            else:
                head += " — answered as text"
            out.append(head)
        out.append("")

    verification = trace.get("verification")
    if verification:
        out.append("## Verification")
        out.append("")
        out.append(f"```json\n{json.dumps(verification, indent=2)}\n```")
        out.append("")

    out.append("## Model traffic")
    out.append("")
    out.append(f"- root_agent frames: {len(trace.get('agent_llm_traffic', []))}")
    out.append(f"- escalation_triage frames: {len(trace.get('triage_llm_traffic', []))}")
    calls = [
        part["function_call"]["name"]
        for frame in trace.get("agent_llm_traffic", [])
        if frame["direction"] == "response"
        for part in frame["parts"]
        if part.get("function_call")
    ]
    out.append(f"- tools the agent called, in order: {' -> '.join(calls) or '(none)'}")
    out.append("")

    if trace.get("final_reply"):
        out.append("## Final reply to the user")
        out.append("")
        out.append("```")
        out.append(trace["final_reply"])
        out.append("```")
        out.append("")

    path.write_text("\n".join(out), encoding="utf-8")


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Capture an end-to-end root_agent trace for one prospect."
    )
    parser.add_argument("--address", help="Prospect street address (required to capture).")
    parser.add_argument("--cases", type=int, help="Order quantity in cases.")
    parser.add_argument("--day", help="Preferred day: MON/TUE/WED/THU/FRI/SAT.")
    parser.add_argument("--window", help='Preferred window, e.g. "09:00-12:00".')
    parser.add_argument("--name", help="Business name, if known.")
    parser.add_argument("--customer-number", help="Existing Sysco number (NNN-NNNNNN).")
    parser.add_argument(
        "--message",
        help="Override the exact sentence sent to the agent (defaults to the "
        "address/cases/slot joined as a sales rep would type it).",
    )
    parser.add_argument("--out", help="Output directory for trace.json + digest.md.")
    parser.add_argument("--wrap-html", help="Wrap this body-only fragment into a page.")
    parser.add_argument("--out-html", help="Destination for --wrap-html.")
    parser.add_argument("--quiet", action="store_true", help="Suppress INFO logging.")
    args = parser.parse_args()

    if args.wrap_html:
        if not args.out_html:
            parser.error("--wrap-html requires --out-html")
        wrap_html(Path(args.wrap_html), Path(args.out_html))
        return 0

    if not args.address or not args.cases:
        parser.error("--address and --cases are required (or use --wrap-html)")
    if bool(args.day) != bool(args.window):
        parser.error("--day and --window must be given together, or not at all")

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        stream=sys.stderr,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if args.quiet:
        # These libraries attach their own handlers/levels, so the root level
        # alone does not silence them.
        for noisy in ("google_adk", "google_genai", "httpx", "smart_assignment"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    out_dir = Path(args.out) if args.out else REPO_ROOT / ".trace" / "prospect"
    out_dir.mkdir(parents=True, exist_ok=True)

    trace = TraceCapture(args).run()

    trace_path = out_dir / "trace.json"
    trace_path.write_text(
        json.dumps(trace, indent=2, default=str, ensure_ascii=False), encoding="utf-8"
    )
    digest_path = out_dir / "digest.md"
    write_digest(trace, digest_path)

    print(f"\ntrace : {trace_path}  ({trace_path.stat().st_size:,} bytes)")
    print(f"digest: {digest_path}")
    print(f"agent run ok: {trace.get('agent_run_ok')}")
    if not trace.get("agent_run_ok"):
        print("NOTE: the agent turn failed; see agent_error in the trace.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
