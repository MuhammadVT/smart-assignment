"""
Generate the GitHub Pages overview site (``docs/index.html``) directly from
live workflow output, so the content can never drift from the code.

The page is a four-tab single-page site:
  1. Overview     — what the agent does, the steps, the rules, the outcomes.
  2. Architecture — the agentic workflow diagram + how the pieces fit.
  3. Simulator    — how scoring is computed (with the real formulas), plus an
                    interactive runner and the per-customer results.
  4. Frontend     — the Salesforce-embedded "Choose a delivery slot" view a sales
                    consultant confirms (a prospect picker over the same output).

All example/interactive data is precomputed here from the real pipeline
(``mock_customers.SAMPLE_CUSTOMERS``) and embedded as JSON, so the static site
needs no backend. Reasoning uses the DeterministicReasoner so the page is
reproducible offline and regenerating with no code change produces no diff.

CLI: ``python3 scripts/generate_page.py``.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Optional

from smart_assignment.mock_customers import SAMPLE_CUSTOMERS
from smart_assignment.pipeline import run_slot_recommendation
from smart_assignment.reasoning import DeterministicReasoner
from smart_assignment.shared.config import (
    DEFAULT_CONFIG,
    FACTOR_CAPACITY_BUFFER,
    FACTOR_GEO_CLUSTERING,
    FACTOR_SLOT_AVAILABILITY,
    FACTOR_WINDOW_MATCH,
    Config,
)
from smart_assignment.shared.constraints import CONSTRAINT_LABEL, build_context
from smart_assignment.shared.models import (
    CandidateEvaluation,
    Decision,
    RecommendationResult,
    SlotRecommendation,
)
from smart_assignment.shared.slot_selection import nearest_neighbors
from smart_assignment.shared.timeutils import (
    duration_minutes,
    fmt_time,
    fmt_window,
    overlap_minutes,
)

DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "docs" / "index.html"

FACTOR_LABEL = {
    FACTOR_GEO_CLUSTERING: "Geographic clustering",
    FACTOR_CAPACITY_BUFFER: "Capacity buffer",
    FACTOR_WINDOW_MATCH: "Slot match (day + time)",
    FACTOR_SLOT_AVAILABILITY: "Slot availability (openness)",
}
DECISION_PILL = {
    Decision.RECOMMENDED: ("rec", "✔ Recommended"),
    Decision.ESCALATED_LOW_SCORE: ("low", "⚠ Low score — human review"),
    Decision.ESCALATED_NO_FEASIBLE_SLOT: ("no", "✖ No feasible slot — specialist"),
}
DECISION_SHORT = {
    Decision.RECOMMENDED: "Recommended — auto-assign",
    Decision.ESCALATED_LOW_SCORE: "Escalate — low score",
    Decision.ESCALATED_NO_FEASIBLE_SLOT: "Escalate — no feasible slot",
}
SCORE_CLASS = {
    Decision.RECOMMENDED: "g",
    Decision.ESCALATED_LOW_SCORE: "a",
    Decision.ESCALATED_NO_FEASIBLE_SLOT: "r",
}
SCORE_TEXT_COLOR = {
    Decision.RECOMMENDED: "var(--green)",
    Decision.ESCALATED_LOW_SCORE: "var(--amber)",
    Decision.ESCALATED_NO_FEASIBLE_SLOT: "var(--red)",
}

_STYLE = """
  :root {
    --navy: #0b2e59; --blue: #1257a6; --blue-soft: #e8f0fb; --ink: #1a2233;
    --muted: #5b6675; --line: #e3e8ef; --bg: #f6f8fb; --card: #ffffff;
    --green: #1a7f37; --green-soft: #e7f4ea; --amber: #9a6700; --amber-soft: #fdf3d8;
    --red: #b42318; --red-soft: #fbe9e7; --violet: #5b3fb0; --violet-soft: #efeafb;
    --radius: 14px; --shadow: 0 1px 2px rgba(16,32,64,.06), 0 6px 20px rgba(16,32,64,.06);
    --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  }
  * { box-sizing: border-box; }
  html { scroll-behavior: smooth; }
  body { margin: 0; color: var(--ink); background: var(--bg);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    line-height: 1.55; }
  a { color: var(--blue); }
  .wrap { max-width: 1080px; margin: 0 auto; padding: 0 20px; }
  .chip { display: inline-flex; align-items: center; gap: 6px; font-size: 12px; font-weight: 600;
    letter-spacing: .02em; padding: 5px 11px; border-radius: 999px; background: rgba(255,255,255,.15);
    border: 1px solid rgba(255,255,255,.3); }
  header.hero { background: linear-gradient(135deg, var(--navy), var(--blue)); color: #fff; padding: 48px 0 52px; }
  .hero h1 { margin: 14px 0 8px; font-size: 40px; line-height: 1.1; letter-spacing: -.02em; }
  .hero p.lead { margin: 0; font-size: 18px; max-width: 660px; color: #dbe6f5; }
  .hero .meta { margin-top: 22px; display: flex; flex-wrap: wrap; gap: 10px; }
  /* tabs */
  .tabbar { position: sticky; top: 0; z-index: 30; background: #fff; border-bottom: 1px solid var(--line);
    box-shadow: 0 2px 8px rgba(16,32,64,.04); }
  .tabbar .wrap { display: flex; gap: 4px; }
  .tabbtn { appearance: none; border: 0; background: transparent; padding: 15px 18px; font-size: 14.5px;
    font-weight: 700; color: var(--muted); cursor: pointer; border-bottom: 3px solid transparent; }
  .tabbtn:hover { color: var(--ink); }
  .tabbtn.active { color: var(--blue); border-bottom-color: var(--blue); }
  .tabpanel { display: none; }
  .tabpanel.active { display: block; }
  section { padding: 46px 0; }
  section h2 { font-size: 26px; letter-spacing: -.01em; margin: 0 0 6px; }
  section .sub { color: var(--muted); margin: 0 0 26px; max-width: 720px; }
  .eyebrow { color: var(--blue); font-weight: 700; font-size: 13px; letter-spacing: .08em; text-transform: uppercase; }
  .agent-banner { display: flex; gap: 12px; align-items: center; background: var(--violet-soft);
    border: 1px solid #ddd2f4; border-radius: var(--radius); padding: 14px 18px; margin-bottom: 24px;
    color: #3a2b6b; font-size: 14px; }
  .agent-banner .big { font-size: 22px; }
  .flow { display: grid; grid-template-columns: repeat(5, 1fr); gap: 14px; }
  .step { background: var(--card); border: 1px solid var(--line); border-radius: var(--radius);
    padding: 18px 16px; box-shadow: var(--shadow); position: relative; }
  .step .abadge { position: absolute; top: 12px; right: 12px; font-size: 10px; font-weight: 700;
    color: var(--violet); background: var(--violet-soft); border-radius: 999px; padding: 3px 8px; }
  .step .num { width: 30px; height: 30px; border-radius: 8px; background: var(--blue-soft); color: var(--blue);
    display: grid; place-items: center; font-weight: 700; font-size: 14px; margin-bottom: 10px; }
  .step h3 { margin: 0 0 4px; font-size: 15px; }
  .step p { margin: 0; font-size: 13px; color: var(--muted); }
  .step .action { margin-top: 8px; font-size: 12px; color: var(--violet); font-weight: 600; }
  .grid-2 { display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; }
  .grid-3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }
  .card { background: var(--card); border: 1px solid var(--line); border-radius: var(--radius);
    padding: 20px; box-shadow: var(--shadow); }
  .card .icon { font-size: 22px; }
  .card h3 { margin: 8px 0 4px; font-size: 16px; }
  .card h4 { margin: 4px 0 3px; font-size: 14px; }
  .card p { margin: 0; font-size: 14px; color: var(--muted); }
  .tag { display: inline-block; font-size: 11px; font-weight: 700; color: var(--blue);
    background: var(--blue-soft); padding: 3px 8px; border-radius: 6px; margin-top: 10px; }
  .bar { height: 8px; border-radius: 6px; background: #eef1f6; overflow: hidden; margin-top: 8px; }
  .bar > span { display: block; height: 100%; background: var(--blue); border-radius: 6px; }
  .formula { font-family: var(--mono); background: #f4f7fc; border: 1px solid #dde6f2; border-radius: 9px;
    padding: 11px 13px; font-size: 12.5px; color: #22364f; margin-top: 12px; overflow-x: auto; }
  .formula b { color: var(--navy); }
  .srccard h4 { display: flex; align-items: center; gap: 8px; }
  .srccard .badge { font-size: 10px; font-weight: 700; padding: 2px 7px; border-radius: 999px; }
  .srccard .badge.cfg { background: var(--blue-soft); color: var(--blue); }
  .srccard .badge.route { background: #eef7ef; color: var(--green); }
  .srccard .badge.intake { background: var(--violet-soft); color: var(--violet); }
  .srclist { margin: 10px 0 0; padding: 0; list-style: none; display: grid; gap: 7px; font-size: 12.5px; }
  .srclist li { display: flex; justify-content: space-between; gap: 10px; }
  .srclist .k { color: #33415c; }
  .srclist .v { font-family: var(--mono); color: var(--navy); font-weight: 600; white-space: nowrap; }
  .srclist .src { color: var(--muted); }
  .arch { background: #fff; border: 1px solid var(--line); border-radius: var(--radius);
    padding: 20px; box-shadow: var(--shadow); }
  .arch svg { width: 100%; height: auto; display: block; }
  .arch-legend { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-top: 22px; }
  .arch-legend .card { padding: 16px; }
  .arch-legend p { font-size: 12.5px; }
  /* LLM & agent touchpoints — every place a model is in the loop. */
  .tp { display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; }
  .tpcard { position: relative; }
  .tptype { display: inline-block; font-size: 10px; font-weight: 800; letter-spacing: .05em;
    text-transform: uppercase; padding: 3px 9px; border-radius: 999px; }
  .tptype.agent { background: var(--violet-soft); color: var(--violet); }
  .tptype.sub   { background: #e6ecfb; color: #274bbd; }
  .tptype.call  { background: var(--blue-soft); color: var(--blue); }
  .tptype.narr  { background: var(--green-soft); color: var(--green); }
  .tpcard h4 { margin: 10px 0 2px; font-size: 15.5px; }
  .tpcard .where { font-family: var(--mono); font-size: 11.5px; color: var(--muted); }
  .tpcard p { margin: 8px 0 0; font-size: 13px; color: var(--muted); }
  .tpcard .guard { margin-top: 10px; font-size: 12.5px; color: #33415c; background: #f4f7fc;
    border: 1px solid #dde6f2; border-radius: 8px; padding: 8px 10px; }
  .tpcard .guard b { color: var(--navy); }
  .tpmeta { margin-top: 10px; display: flex; flex-wrap: wrap; gap: 8px; align-items: center; font-size: 11.5px; }
  .tpmeta .flag { font-family: var(--mono); background: #f1f4f9; border-radius: 6px; padding: 2px 7px; color: #33415c; }
  .tpmeta .state { font-weight: 700; padding: 2px 8px; border-radius: 999px; }
  .tpmeta .state.on  { background: var(--green-soft); color: var(--green); }
  .tpmeta .state.off { background: #eef1f6; color: var(--muted); }
  .tpmeta .state.always { background: var(--violet-soft); color: var(--violet); }
  .guarantee { margin-top: 18px; background: var(--violet-soft); border: 1px solid #ddd2f4;
    border-radius: var(--radius); padding: 16px 18px; color: #3a2b6b; font-size: 13.5px; }
  .guarantee b { color: var(--navy); }
  .legend { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }
  .pill { display: inline-flex; align-items: center; gap: 8px; font-weight: 700; font-size: 13px;
    padding: 6px 12px; border-radius: 999px; }
  .pill.rec { background: var(--green-soft); color: var(--green); }
  .pill.low { background: var(--amber-soft); color: var(--amber); }
  .pill.no  { background: var(--red-soft); color: var(--red); }
  /* interactive simulator */
  .sim { background: var(--card); border: 1px solid var(--line); border-radius: var(--radius);
    box-shadow: var(--shadow); padding: 22px; }
  .sim-controls { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }
  .sim input { font-size: 15px; padding: 11px 13px; border: 1px solid var(--line); border-radius: 9px;
    min-width: 200px; font-family: var(--mono); }
  .sim button.run { background: var(--blue); color: #fff; border: 0; border-radius: 9px; padding: 12px 20px;
    font-weight: 700; font-size: 14.5px; cursor: pointer; box-shadow: 0 2px 6px rgba(18,87,166,.25); }
  .sim button.run:disabled { opacity: .55; cursor: default; box-shadow: none; }
  .sim input.flash { animation: flash .8s ease; }
  @keyframes flash { 0% { box-shadow: 0 0 0 3px rgba(18,87,166,.45); border-color: var(--blue); }
    100% { box-shadow: 0 0 0 0 rgba(18,87,166,0); } }
  .picker { margin-top: 16px; padding: 14px 16px; border: 1px dashed #c7d6ea; border-radius: 11px; background: #fbfcfe; }
  .picker-label { display: block; font-size: 12.5px; color: var(--muted); font-weight: 600; margin-bottom: 10px; }
  .chips { display: flex; gap: 8px; flex-wrap: wrap; }
  .chip-btn { font-family: var(--mono); font-size: 12px; border: 1px solid #c7d6ea; background: #fff;
    color: var(--blue); border-radius: 8px; padding: 7px 11px; cursor: pointer; }
  .chip-btn:hover { background: var(--blue-soft); }
  .chip-btn.selected { background: var(--blue); color: #fff; border-color: var(--blue); }
  .sim-hint { margin-top: 10px; font-size: 12.5px; color: var(--green); min-height: 16px; font-weight: 600; }
  .sim-error { color: var(--red); font-size: 13px; margin-top: 6px; min-height: 16px; }
  .sim-cust { margin: 14px 0 4px; font-size: 14.5px; }
  .sim-cust .cnum { font-family: var(--mono); color: var(--muted); font-size: 12px; }
  .sim-step { display: grid; grid-template-columns: 26px 1fr; gap: 12px; padding: 11px 0; border-top: 1px solid var(--line); }
  .sim-dot { width: 18px; height: 18px; border-radius: 50%; margin-top: 3px; border: 2px solid var(--line); background: #fff; }
  .sim-step.running .sim-dot { border-color: var(--blue); animation: pulse 1s infinite; }
  .sim-step.done .sim-dot { border-color: var(--green); background: var(--green); }
  @keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(18,87,166,.35); } 70% { box-shadow: 0 0 0 8px rgba(18,87,166,0); } 100% { box-shadow: 0 0 0 0 rgba(18,87,166,0); } }
  .sim-title { font-weight: 700; font-size: 14px; }
  .sim-state { font-weight: 700; font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: .05em; margin-left: 6px; }
  .sim-step.done .sim-state { color: var(--green); }
  .sim-action { font-size: 13px; color: var(--violet); margin-top: 2px; }
  .sim-lines { margin-top: 7px; font-size: 12.5px; color: #33415c; display: grid; gap: 3px; }
  .sim-line .ok { color: var(--green); font-weight: 700; }
  .sim-line .no { color: var(--red); font-weight: 700; }
  .sim-line .calc { font-family: var(--mono); font-size: 11.5px; color: #5b6675; }
  .sim-line .calc b { color: var(--navy); }
  /* Score & Rank: each (route, slot) option is a collapsible <details> -- a
     compact score line (the summary) with its full per-factor math tucked inside,
     hidden by default. Same numbers as before, laid out to be scannable. */
  .sim-line .rs-details { border: 1px solid var(--line); border-radius: 11px; background: var(--card);
    box-shadow: var(--shadow); overflow: hidden; }
  .sim-line .rs-details.winner { border-color: var(--green); }
  .sim-line .rs-head { list-style: none; cursor: pointer; user-select: none; display: flex;
    align-items: center; gap: 9px; padding: 9px 12px; }
  .sim-line .rs-head::-webkit-details-marker { display: none; }
  .sim-line .rs-caret { flex: none; width: 11px; text-align: center; color: var(--muted);
    font-size: 9px; transition: transform .15s; }
  .sim-line .rs-details[open] .rs-caret { transform: rotate(90deg); color: var(--blue); }
  .sim-line .rs-id { font-weight: 700; font-size: 13.5px; font-variant-numeric: tabular-nums; }
  .sim-line .rs-name { color: var(--muted); font-size: 12px; }
  .sim-line .rs-win { font-family: var(--mono); font-size: 11.5px; color: var(--muted); }
  .sim-line .rs-spacer { flex: 1 1 auto; min-width: 6px; }
  .sim-line .rs-showmath { font-size: 11px; font-weight: 600; color: var(--blue); white-space: nowrap; }
  .sim-line .rs-showmath::after { content: "show the math"; }
  .sim-line .rs-details[open] .rs-showmath { color: var(--muted); }
  .sim-line .rs-details[open] .rs-showmath::after { content: "hide the math"; }
  .sim-line .rs-score { font-family: var(--mono); font-weight: 700; font-variant-numeric: tabular-nums;
    font-size: 13.5px; border-radius: 8px; padding: 3px 8px; line-height: 1; background: #eef2f7; color: var(--navy); }
  .sim-line .rs-details.winner .rs-score { background: var(--green-soft); color: var(--green); }
  .sim-line .rs-star { color: var(--green); }
  .sim-line .rs-math { border-top: 1px solid var(--line); background: #f8fafd; padding: 10px 12px 11px; }
  .sim-line .rs-factor { display: grid; grid-template-columns: 4px 1fr auto; gap: 3px 10px;
    padding: 8px 0; border-bottom: 1px dashed var(--line); }
  .sim-line .rs-factor:first-child { padding-top: 1px; }
  .sim-line .rs-factor:last-child { border-bottom: 0; }
  .sim-line .rs-ftick { grid-row: 1 / span 2; border-radius: 3px; align-self: stretch; }
  .sim-line .rs-fname { font-size: 12.5px; font-weight: 600; color: var(--ink); }
  .sim-line .rs-fnums { display: inline-flex; align-items: baseline; gap: 7px; justify-self: end; white-space: nowrap; }
  .sim-line .rs-fval { font-family: var(--mono); font-weight: 700; font-size: 12.5px;
    font-variant-numeric: tabular-nums; color: var(--navy); }
  .sim-line .rs-fwt { font-family: var(--mono); font-size: 10.5px; color: var(--muted); }
  .sim-line .rs-fcontrib { font-family: var(--mono); font-size: 10.5px; color: var(--muted); }
  .sim-line .rs-fdetail { grid-column: 2 / span 2; font-size: 11.5px; color: var(--muted); line-height: 1.5; }
  .sim-line .rs-fformula { font-family: var(--mono); font-size: 10.5px; color: var(--ink);
    background: #eef2f7; border-radius: 5px; padding: 1px 6px; margin-right: 6px; }
  .sim-line .rs-total { display: flex; align-items: baseline; flex-wrap: wrap; gap: 7px;
    margin-top: 9px; padding-top: 9px; border-top: 1px solid var(--line); }
  .sim-line .rs-total-lbl { font-size: 11.5px; font-weight: 700; color: var(--ink); }
  .sim-line .rs-total-expr { font-family: var(--mono); font-size: 10.5px; color: var(--muted); }
  .sim-line .rs-total-eq { font-family: var(--mono); font-size: 13px; font-weight: 700; color: var(--navy); margin-left: auto; }
  .sim-output { margin-top: 18px; }
  .examples { display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; }
  .result { background: var(--card); border: 1px solid var(--line); border-radius: var(--radius);
    box-shadow: var(--shadow); overflow: hidden; }
  .result .rhead { padding: 18px 20px; border-bottom: 1px solid var(--line); }
  .result .cnum { font-family: var(--mono); font-size: 12px; color: var(--muted); }
  .result h3 { margin: 2px 0 2px; font-size: 18px; }
  .result .addr { font-size: 13px; color: var(--muted); }
  .result .facts { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
  .fact { font-size: 12px; background: #f1f4f9; color: #33415c; border-radius: 7px; padding: 4px 9px; }
  .result .rbody { padding: 18px 20px; }
  .decision-row { display: flex; align-items: center; justify-content: space-between; gap: 12px; flex-wrap: wrap; }
  .slot { margin: 12px 0 4px; font-size: 15px; }
  .slot b { color: var(--navy); }
  .score { margin-top: 4px; }
  .score .bar { height: 10px; }
  .score .bar > span.g { background: var(--green); }
  .score .bar > span.a { background: var(--amber); }
  .score .bar > span.r { background: var(--red); }
  .score small { color: var(--muted); font-size: 12px; }
  .factors { margin-top: 14px; display: grid; gap: 9px; }
  .factor { display: grid; grid-template-columns: 150px 1fr 42px; align-items: center; gap: 10px; font-size: 12.5px; }
  .factor .fname { color: #33415c; }
  .factor .fval { text-align: right; color: var(--muted); font-variant-numeric: tabular-nums; }
  .reason { margin-top: 14px; background: var(--blue-soft); border-left: 3px solid var(--blue);
    border-radius: 8px; padding: 12px 14px; font-size: 13.5px; color: #22364f; }
  .reason .lbl { font-weight: 700; color: var(--blue); font-size: 11px; letter-spacing: .06em;
    text-transform: uppercase; display: block; margin-bottom: 4px; }
  .reason .summary { font-weight: 600; display: block; margin-bottom: 6px; }
  .reason ul.reasons { margin: 4px 0 0; padding-left: 18px; }
  .reason ul.reasons li { margin: 2px 0; }
  .reason .sub { margin-top: 8px; }
  .reason .sub .k { font-weight: 700; color: var(--blue); font-size: 10.5px;
    letter-spacing: .05em; text-transform: uppercase; display: block; }
  .reason .vs { margin-top: 8px; font-size: 12px; color: var(--muted); font-style: italic; }
  details.routes { margin-top: 14px; border-top: 1px dashed var(--line); padding-top: 10px; }
  details.routes summary { cursor: pointer; font-size: 13px; font-weight: 600; color: var(--blue); }
  .routelist { margin-top: 10px; display: grid; gap: 8px; }
  .route { border: 1px solid var(--line); border-radius: 9px; padding: 9px 11px; font-size: 12.5px; }
  .route .rtitle { display: flex; justify-content: space-between; gap: 8px; font-weight: 600; }
  .route .status { font-size: 11px; font-weight: 700; padding: 1px 7px; border-radius: 5px; white-space: nowrap; }
  .route .status.ok { background: var(--green-soft); color: var(--green); }
  .route .status.bad { background: var(--red-soft); color: var(--red); }
  .route.routecard { padding: 12px 13px; }
  .route.routecard .score { margin-top: 8px; }
  .route.routecard .score .bar { height: 8px; }
  .route.routecard .factors { margin-top: 9px; }
  .rectag { font-size: 10px; font-weight: 700; color: var(--green); background: var(--green-soft);
    border-radius: 999px; padding: 2px 7px; margin-left: 6px; white-space: nowrap; }
  /* Candidate (route, slot) options listed inside a route card (route-slot path). */
  .slot-options { margin-top: 12px; display: grid; gap: 10px; }
  .slot-option { border: 1px solid var(--line); border-radius: 10px; padding: 9px 11px; background: var(--bg); }
  .slot-option.rec { border-color: var(--green); background: var(--green-soft); }
  .slot-head { display: flex; align-items: center; gap: 8px; font-size: 12.5px; }
  .slot-win { font-weight: 700; color: var(--navy); font-family: var(--mono); }
  .slot-score { margin-left: auto; color: var(--muted); font-variant-numeric: tabular-nums; }
  .slot-option .factors { margin-top: 8px; }
  /* A factor that's present but deliberately unscored (no stated preference). */
  .factor.na { grid-template-columns: 150px 1fr; }
  .factor.na .fname { color: var(--muted); }
  .factor.na .na-note { font-size: 11px; color: var(--muted); font-weight: 600; justify-self: start;
    background: #f1f4f9; border: 1px dashed #cfd8e6; border-radius: 6px; padding: 2px 8px; }
  .checks { margin: 6px 0 0; padding-left: 0; list-style: none; color: var(--muted); }
  .checks li { padding: 1px 0; }
  .checks li .ok { color: var(--green); font-weight: 700; }
  .checks li .no { color: var(--red); font-weight: 700; }
  .note { background: #fff8ec; border: 1px solid #f2e2bd; border-radius: var(--radius);
    padding: 18px 20px; color: #5c4b1f; font-size: 14px; }
  footer { padding: 34px 0 60px; color: var(--muted); font-size: 13px; border-top: 1px solid var(--line); background: #fff; }
  footer a { text-decoration: none; }
  @media (max-width: 860px) {
    .flow { grid-template-columns: repeat(2, 1fr); }
    .grid-2, .grid-3, .legend, .arch-legend, .tp { grid-template-columns: 1fr; }
    .examples { grid-template-columns: 1fr; }
    .hero h1 { font-size: 32px; }
    .factor { grid-template-columns: 120px 1fr 40px; }
    .tabbar .wrap { overflow-x: auto; }
  }
"""

# Styles for the Frontend tab -- the Salesforce-embedded "Choose a delivery slot"
# view a sales consultant confirms. Kept separate from _STYLE for readability;
# concatenated into the same <style>. Uses the site's tokens so it reads native,
# with violet as the agent accent (matches the LLM colour used elsewhere).
_FE_STYLE = """
  /* The Frontend tab breaks out of the site's 1080px .wrap to a wider container
     so the three columns aren't cramped. */
  .fe-wrap { max-width: 1320px; margin: 0 auto; padding: 0 24px; }
  .fe-eyebrow { color: var(--violet); font-weight: 800; font-size: 12px; letter-spacing: .12em;
    text-transform: uppercase; }
  .fe-status { display: inline-flex; gap: 8px; align-items: center; font-size: 12px; color: var(--muted);
    background: #eef2f8; border: 1px solid var(--line); border-radius: 999px; padding: 6px 13px; margin: 4px 0 22px; }
  .fe-status b { color: var(--navy); font-weight: 700; }
  /* align-items: stretch so the left profile column matches the (tall) middle
     column's height; the map column opts out via align-self so it stays compact. */
  .fe-grid { display: grid; grid-template-columns: 280px minmax(0,1fr) 360px; gap: 24px; align-items: stretch; }
  .fe-side { background: var(--card); border: 1px solid var(--line); border-radius: var(--radius);
    box-shadow: var(--shadow); padding: 20px; display: flex; flex-direction: column; }
  .fe-side h3 { margin: 0 0 2px; font-size: 15px; }
  .fe-side .kicker { color: var(--violet); font-weight: 700; font-size: 11px; letter-spacing: .06em;
    text-transform: uppercase; }
  .fe-field { margin-top: 14px; }
  .fe-field .l { font-size: 11px; color: var(--muted); font-weight: 600; letter-spacing: .02em; }
  .fe-field .v { font-size: 14px; color: var(--ink); font-weight: 600; margin-top: 2px; }
  .fe-field .v.addr { display: flex; gap: 6px; font-weight: 500; }
  .fe-field .v .pin { color: var(--violet); }
  /* The note follows the profile fields; the stretched card's leftover space
     falls BELOW it (not between the fields and the note). */
  .fe-note { margin-top: 18px; background: var(--violet-soft); border: 1px solid #ddd2f4; border-radius: 11px;
    padding: 12px 13px; color: #3a2b6b; font-size: 12px; }
  .fe-note .h { font-weight: 700; display: flex; gap: 6px; align-items: center; margin-bottom: 3px; }
  .fe-main { display: grid; gap: 14px; }
  .fe-banner { border-radius: var(--radius); padding: 13px 16px; font-size: 13px; font-weight: 600; }
  .fe-banner.warn { background: var(--amber-soft); border: 1px solid #f2e2bd; color: #7a5c12; }
  .fe-banner.stop { background: var(--red-soft); border: 1px solid #f2cfc9; color: #7a2b22; }
  .fe-opt { background: var(--card); border: 1px solid var(--line); border-radius: var(--radius);
    box-shadow: var(--shadow); padding: 18px 20px; transition: box-shadow .15s, border-color .15s; }
  .fe-opt.selectable { cursor: pointer; }
  .fe-opt.selectable:hover { border-color: #c9bcec; }
  .fe-opt.selected { border: 2px solid var(--violet); box-shadow: 0 0 0 4px rgba(91,63,176,.10), var(--shadow); }
  .fe-opt.dim { opacity: .92; }
  .fe-opt-head { display: flex; align-items: flex-start; gap: 12px; }
  .fe-radio { width: 20px; height: 20px; border-radius: 50%; border: 2px solid #cdd6e4; flex: none; margin-top: 2px; }
  .fe-opt.selected .fe-radio { border-color: var(--violet);
    background: radial-gradient(circle at center, var(--violet) 0 6px, #fff 7px 20px); }
  .fe-radio.no { border-color: #e0b6b0; }
  .fe-opt-title { flex: 1; min-width: 0; }
  .fe-opt-title .when { font-size: 16.5px; font-weight: 700; color: var(--ink); }
  .fe-opt-title .when b { color: var(--navy); }
  .fe-opt-title .nowrap { white-space: nowrap; }
  .fe-opt-title .fe-route { font-size: 12.5px; color: var(--muted); margin-top: 4px; }
  .fe-opt-title .fe-route b { color: #33415c; font-family: var(--mono); font-weight: 600; }
  /* Quality-rank chip (replaces raw scores): High confidence / Medium / Low feasible. */
  .fe-rank { display: inline-flex; align-items: center; gap: 6px; font-size: 11.5px; font-weight: 800;
    padding: 5px 11px; border-radius: 999px; white-space: nowrap; }
  .fe-rank .d { width: 7px; height: 7px; border-radius: 50%; }
  .fe-rank.hi  { background: var(--green-soft); color: var(--green); }
  .fe-rank.hi .d  { background: var(--green); }
  .fe-rank.med { background: var(--blue-soft); color: var(--blue); }
  .fe-rank.med .d { background: var(--blue); }
  .fe-rank.lo  { background: var(--amber-soft); color: var(--amber); }
  .fe-rank.lo .d  { background: var(--amber); }
  .fe-rank.no  { background: var(--red-soft); color: var(--red); }
  .fe-rank.no .d  { background: var(--red); }
  .fe-tag { font-size: 10.5px; font-weight: 700; color: var(--muted); margin-left: 8px; white-space: nowrap; }
  /* An explicit, always-visible select affordance so it's obvious each slot is
     clickable -- the label flips to "Selected" via the .selected class. */
  .fe-selrow { display: flex; justify-content: flex-end; margin-top: 14px; }
  .fe-selmark { display: inline-flex; align-items: center; font-size: 12px; font-weight: 700;
    color: var(--violet); border: 1px solid #c9bcec; background: #fff; border-radius: 8px; padding: 6px 13px; }
  .fe-selmark::after { content: "Select this slot"; }
  .fe-opt.selectable:hover .fe-selmark { background: var(--violet-soft); }
  .fe-opt.selected .fe-selmark { background: var(--violet); color: #fff; border-color: var(--violet); }
  .fe-opt.selected .fe-selmark::after { content: "\\2713  Selected"; }
  .fe-opt.selectable:focus-visible { outline: 2px solid var(--violet); outline-offset: 2px; }
  .fe-why { margin: 12px 0 0; font-size: 13.5px; color: #33415c; line-height: 1.5; }
  .fe-tiles { display: grid; grid-template-columns: repeat(var(--fe-cols, 4), minmax(0,1fr)); gap: 10px; margin-top: 14px; }
  .fe-tile { background: var(--bg); border: 1px solid var(--line); border-radius: 10px; padding: 10px 11px; }
  .fe-tile .tl { font-size: 10.5px; color: var(--muted); font-weight: 600; text-transform: uppercase; letter-spacing: .03em; }
  .fe-tile .tv { font-size: 15px; font-weight: 700; color: var(--navy); margin-top: 4px; font-variant-numeric: tabular-nums; }
  .fe-tile .ts { font-size: 10.5px; color: var(--muted); margin-top: 3px; }
  .fe-tile .fs { display: inline-block; font-family: var(--mono); font-size: 10px; font-weight: 700; color: var(--violet);
    background: var(--violet-soft); border-radius: 5px; padding: 1px 5px; margin-top: 5px; }
  .fe-tile.miss .tv { color: var(--amber); }
  .fe-unavail { margin-top: 10px; font-size: 13px; color: #7a2b22; background: var(--red-soft);
    border: 1px solid #f2cfc9; border-radius: 9px; padding: 10px 12px; }
  .fe-unavail .x { color: var(--red); font-weight: 800; }
  .fe-tradeoff { margin-top: 12px; font-size: 12.5px; color: #5b4a1f; background: #fff8ec;
    border: 1px solid #f2e2bd; border-radius: 9px; padding: 9px 12px; }
  .fe-tradeoff b { color: #7a5c12; }
  .fe-escalate { background: var(--violet-soft); border: 1px solid #ddd2f4; border-radius: var(--radius); padding: 16px 18px; }
  .fe-escalate h4 { margin: 0 0 4px; font-size: 14px; color: #3a2b6b; }
  .fe-escalate p { margin: 0 0 12px; font-size: 12.5px; color: #4a3b7b; }
  .fe-escalate button { background: #fff; border: 1px solid #c9bcec; color: var(--violet); font-weight: 700;
    font-size: 13px; border-radius: 9px; padding: 9px 15px; cursor: pointer; }
  .fe-confirm { display: flex; align-items: center; gap: 14px; flex-wrap: wrap; justify-content: space-between;
    background: var(--card); border: 1px solid var(--line); border-radius: var(--radius); box-shadow: var(--shadow); padding: 16px 18px; }
  .fe-confirm .log { font-size: 12.5px; color: var(--muted); max-width: 480px; }
  .fe-confirm .log b { color: var(--navy); }
  .fe-confirm .btns { display: flex; gap: 10px; }
  .fe-btn-ghost { background: #fff; border: 1px solid var(--line); color: var(--ink); font-weight: 700;
    font-size: 14px; border-radius: 9px; padding: 10px 18px; cursor: pointer; }
  .fe-btn-primary { background: var(--violet); border: 0; color: #fff; font-weight: 700; font-size: 14px;
    border-radius: 9px; padding: 11px 22px; cursor: pointer; box-shadow: 0 2px 8px rgba(91,63,176,.3); }
  .fe-btn-primary:disabled { opacity: .5; cursor: default; box-shadow: none; }
  .fe-map { background: var(--card); border: 1px solid var(--line); border-radius: var(--radius);
    box-shadow: var(--shadow); padding: 16px; align-self: start; position: sticky; top: 70px; }
  .fe-map .mt { font-size: 13px; font-weight: 700; color: var(--navy); }
  .fe-map .ms { font-size: 11px; color: var(--muted); margin-top: 2px; }
  .fe-map svg { width: 100%; height: auto; display: block; margin-top: 8px; border-radius: 10px;
    border: 1px solid var(--line); }
  .fe-legend { display: flex; flex-wrap: wrap; gap: 8px 14px; margin-top: 12px; font-size: 11.5px; color: #33415c; }
  .fe-legend span { display: inline-flex; align-items: center; gap: 6px; }
  .fe-dot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; }
  .fe-gain { margin-top: 10px; font-size: 12px; color: var(--green); font-weight: 700; }
  .fe-map .nomap { font-size: 12.5px; color: var(--muted); margin-top: 10px; line-height: 1.5; }
  .fe-picker { margin: 0 0 20px; padding: 14px 16px; border: 1px dashed #c7d6ea; border-radius: 11px; background: #fbfcfe; }
  .fe-picker .picker-label { display: block; font-size: 12.5px; color: var(--muted); font-weight: 600; margin-bottom: 10px; }
  @media (max-width: 1040px) {
    .fe-grid { grid-template-columns: 1fr; }
    .fe-side, .fe-map { position: static; }
    /* Set the property directly (not the --fe-cols var) so it wins over the
       inline var each tile row carries. */
    .fe-tiles { grid-template-columns: repeat(2, minmax(0,1fr)); }
  }
"""

# Static architecture diagram (SVG). No curly braces -> safe inside the f-string.
_ARCH_SVG = """
<svg viewBox="0 0 980 668" role="img" aria-label="Agentic workflow architecture diagram">
  <defs>
    <marker id="arw" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#1257a6"/></marker>
    <marker id="arwg" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#1a7f37"/></marker>
    <marker id="arwa" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#9a6700"/></marker>
    <marker id="arwd" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#7c8aa0"/></marker>
  </defs>

  <rect x="326" y="8" width="328" height="54" rx="22" fill="#0b2e59"/>
  <text x="490" y="30" text-anchor="middle" fill="#fff" font-size="13.5" font-weight="700">New prospect intake</text>
  <text x="490" y="48" text-anchor="middle" fill="#cfe0f5" font-size="11" font-weight="500">address · # cases · preferred slot (if stated)</text>
  <line x1="490" y1="62" x2="490" y2="86" stroke="#1257a6" stroke-width="2" marker-end="url(#arw)"/>

  <rect x="24" y="88" width="620" height="440" rx="18" fill="#eef5fd" stroke="#1257a6" stroke-width="2"/>
  <text x="44" y="118" fill="#1257a6" font-size="15" font-weight="800">🤖 AI AGENT — conversational ADK LlmAgent orchestrator</text>

  <rect x="44" y="146" width="136" height="80" rx="12" fill="#fff" stroke="#c7d6ea"/>
  <text x="112" y="180" text-anchor="middle" font-size="13" font-weight="700" fill="#0b2e59">1 · Intake</text>
  <text x="112" y="199" text-anchor="middle" font-size="10.5" fill="#5b6675">validate profile</text>

  <rect x="194" y="146" width="136" height="80" rx="12" fill="#fff" stroke="#c7d6ea"/>
  <text x="262" y="180" text-anchor="middle" font-size="13" font-weight="700" fill="#0b2e59">2 · Geo-Lookup</text>
  <text x="262" y="199" text-anchor="middle" font-size="10.5" fill="#5b6675">geocode + Top-N</text>

  <rect x="344" y="146" width="136" height="80" rx="12" fill="#fff" stroke="#c7d6ea"/>
  <text x="412" y="176" text-anchor="middle" font-size="13" font-weight="700" fill="#0b2e59">3 · Constraint</text>
  <text x="412" y="192" text-anchor="middle" font-size="13" font-weight="700" fill="#0b2e59">Check</text>
  <text x="412" y="210" text-anchor="middle" font-size="10.5" fill="#5b6675">apply hard rules</text>

  <rect x="494" y="146" width="136" height="80" rx="12" fill="#fff" stroke="#c7d6ea"/>
  <text x="562" y="176" text-anchor="middle" font-size="13" font-weight="700" fill="#0b2e59">4 · Score &amp;</text>
  <text x="562" y="192" text-anchor="middle" font-size="13" font-weight="700" fill="#0b2e59">Rank</text>
  <text x="562" y="210" text-anchor="middle" font-size="10.5" fill="#5b6675">weighted scoring</text>

  <line x1="180" y1="186" x2="192" y2="186" stroke="#1257a6" stroke-width="2" marker-end="url(#arw)"/>
  <line x1="330" y1="186" x2="342" y2="186" stroke="#1257a6" stroke-width="2" marker-end="url(#arw)"/>
  <line x1="480" y1="186" x2="492" y2="186" stroke="#1257a6" stroke-width="2" marker-end="url(#arw)"/>

  <rect x="48" y="250" width="204" height="74" rx="12" fill="#efeafb" stroke="#cfc2f0"/>
  <text x="150" y="277" text-anchor="middle" font-size="11.5" font-weight="700" fill="#5b3fb0">🧠 Address resolver (LLM)</text>
  <text x="150" y="295" text-anchor="middle" font-size="9.5" fill="#6b5aa0">on a geocode miss: picks from the</text>
  <text x="150" y="308" text-anchor="middle" font-size="9.5" fill="#6b5aa0">geocoder's suggested matches</text>
  <text x="150" y="319" text-anchor="middle" font-size="8.5" fill="#8a7ec0">— for the user to confirm</text>
  <path d="M244,226 C220,238 182,240 158,250" fill="none" stroke="#7c8aa0" stroke-width="1.6" stroke-dasharray="5 4" marker-end="url(#arwd)"/>

  <text x="334" y="288" text-anchor="middle" font-size="11" font-weight="700" fill="#0b2e59">5 · Decide</text>
  <polygon points="334,300 412,342 334,384 256,342" fill="#fff" stroke="#9a6700" stroke-width="2"/>
  <text x="334" y="339" text-anchor="middle" font-size="11.5" font-weight="700" fill="#9a6700">recommend?</text>
  <text x="334" y="354" text-anchor="middle" font-size="9.5" font-weight="700" fill="#5b3fb0">grounded LLM call</text>
  <path d="M562,226 L562,342 L414,342" fill="none" stroke="#1257a6" stroke-width="2" marker-end="url(#arw)"/>

  <rect x="150" y="560" width="250" height="62" rx="12" fill="#e7f4ea" stroke="#1a7f37"/>
  <text x="275" y="588" text-anchor="middle" font-size="13" font-weight="700" fill="#1a7f37">✅ Recommend &amp; auto-assign</text>
  <text x="275" y="606" text-anchor="middle" font-size="10.5" fill="#3b7a4e">no human needed</text>

  <rect x="452" y="560" width="270" height="62" rx="12" fill="#fdf3d8" stroke="#9a6700"/>
  <text x="587" y="588" text-anchor="middle" font-size="13" font-weight="700" fill="#9a6700">🙋 Escalate to specialist</text>
  <text x="587" y="606" text-anchor="middle" font-size="10.5" fill="#8a6a1f">human reviews &amp; decides</text>

  <path d="M320,382 L275,560" fill="none" stroke="#1a7f37" stroke-width="2" marker-end="url(#arwg)"/>
  <text x="286" y="474" fill="#1a7f37" font-size="11" font-weight="700">yes</text>
  <path d="M352,378 L585,558" fill="none" stroke="#9a6700" stroke-width="2" marker-end="url(#arwa)"/>
  <text x="474" y="474" fill="#9a6700" font-size="11" font-weight="700">no</text>

  <rect x="690" y="150" width="266" height="80" rx="12" fill="#f4f7fb" stroke="#c7d6ea"/>
  <text x="823" y="182" text-anchor="middle" font-size="12.5" font-weight="700" fill="#0b2e59">🗄️ Mocked source systems</text>
  <text x="823" y="201" text-anchor="middle" font-size="10.5" fill="#5b6675">route capacity · geocoding</text>
  <line x1="690" y1="190" x2="648" y2="190" stroke="#7c8aa0" stroke-width="1.6" stroke-dasharray="5 4" marker-end="url(#arwd)"/>

  <rect x="690" y="300" width="266" height="80" rx="12" fill="#efeafb" stroke="#cfc2f0"/>
  <text x="823" y="332" text-anchor="middle" font-size="12.5" font-weight="700" fill="#5b3fb0">🧠 LLM decision &amp; reasoning</text>
  <text x="823" y="351" text-anchor="middle" font-size="10.5" fill="#6b5aa0">grounded recommend/escalate + pick · writes the rationale</text>
  <line x1="690" y1="340" x2="648" y2="340" stroke="#7c8aa0" stroke-width="1.6" stroke-dasharray="5 4" marker-end="url(#arwd)"/>

  <rect x="690" y="452" width="266" height="86" rx="12" fill="#e6ecfb" stroke="#a9bbe8"/>
  <text x="823" y="482" text-anchor="middle" font-size="12.5" font-weight="700" fill="#274bbd">🤝 Escalation-triage sub-agent</text>
  <text x="823" y="501" text-anchor="middle" font-size="10.5" fill="#3a52a8">an LlmAgent, exposed as an ADK AgentTool</text>
  <text x="823" y="518" text-anchor="middle" font-size="10.5" fill="#3a52a8">consulted only on escalation · composes the brief</text>
  <path d="M640,585 C690,585 690,520 688,510" fill="none" stroke="#274bbd" stroke-width="1.6" stroke-dasharray="5 4" marker-end="url(#arw)"/>

  <text x="24" y="656" font-size="11.5" fill="#5b6675">→ agent flow &#160;&#160;·&#160;&#160; ⇢ agent calls a service / sub-agent &#160;&#160;·&#160;&#160; ◇ grounded decision point (LLM)</text>
</svg>
"""

# Interactive simulator logic. Plain string (NOT an f-string) so its braces are safe.
_SIM_JS = """
(function () {
  var DATA = JSON.parse(document.getElementById('workflow-data').textContent);
  var input = document.getElementById('cust-input');
  var runBtn = document.getElementById('run-btn');
  var chipsEl = document.getElementById('chips');
  var stepsEl = document.getElementById('sim-steps');
  var outEl = document.getElementById('sim-output');
  var errEl = document.getElementById('sim-error');
  var hintEl = document.getElementById('sim-hint');
  var keys = Object.keys(DATA);  // each key is a customer's address (or Sysco number if on file)

  keys.forEach(function (key) {
    var b = document.createElement('button');
    b.className = 'chip-btn';
    b.innerHTML = DATA[key].name;
    b.title = DATA[key].address;
    // Clicking a sample loads its address INTO the input box (it does not
    // run) so the relationship is obvious; the user then presses Run.
    b.addEventListener('click', function () {
      input.value = key;
      chipsEl.querySelectorAll('.chip-btn').forEach(function (x) { x.classList.remove('selected'); });
      b.classList.add('selected');
      input.classList.remove('flash');
      void input.offsetWidth;  // restart the flash animation
      input.classList.add('flash');
      input.focus();
      errEl.textContent = '';
      hintEl.textContent = 'Loaded ' + DATA[key].name + '’s address into the box above — now press “Run agent workflow”.';
    });
    chipsEl.appendChild(b);
  });

  function sleep(ms) { return new Promise(function (r) { setTimeout(r, ms); }); }

  async function run() {
    var key = (input.value || '').trim();
    errEl.textContent = ''; hintEl.textContent = ''; outEl.innerHTML = ''; stepsEl.innerHTML = '';
    var d = DATA[key];
    if (!d) {
      errEl.textContent = 'No matching sample prospect. Click one of the sample cards below, '
        + 'or paste one of their addresses exactly.';
      return;
    }
    runBtn.disabled = true;
    var head = document.createElement('div');
    head.className = 'sim-cust';
    head.innerHTML = '🤖 Agent running for <b>' + d.name + '</b> <span class="cnum">' + d.address + '</span>';
    stepsEl.appendChild(head);
    for (var i = 0; i < d.steps.length; i++) {
      var s = d.steps[i];
      var el = document.createElement('div');
      el.className = 'sim-step running';
      el.innerHTML = '<div class="sim-dot"></div><div class="sim-body">' +
        '<div class="sim-title">Step ' + (i + 1) + ' · ' + s.title +
        ' <span class="sim-state">running…</span></div>' +
        '<div class="sim-action">' + s.action + '</div></div>';
      stepsEl.appendChild(el);
      await sleep(720);
      el.classList.remove('running'); el.classList.add('done');
      el.querySelector('.sim-state').textContent = 'done';
      var lines = s.lines.map(function (l) { return '<div class="sim-line">' + l + '</div>'; }).join('');
      el.querySelector('.sim-body').insertAdjacentHTML('beforeend', '<div class="sim-lines">' + lines + '</div>');
      await sleep(260);
    }
    await sleep(180);
    outEl.innerHTML = d.resultHtml + (d.routesHtml || '');
    runBtn.disabled = false;
  }

  runBtn.addEventListener('click', run);
  input.addEventListener('keydown', function (e) { if (e.key === 'Enter') { run(); } });
})();
"""

# Tab switching logic. Plain string (NOT an f-string).
_TABS_JS = """
(function () {
  var btns = document.querySelectorAll('.tabbtn');
  var panels = document.querySelectorAll('.tabpanel');
  var valid = { overview: 1, architecture: 1, simulator: 1, frontend: 1 };
  function activate(name, scroll) {
    btns.forEach(function (b) {
      var on = b.getAttribute('data-tab') === name;
      b.classList.toggle('active', on);
      b.setAttribute('aria-selected', on ? 'true' : 'false');
    });
    panels.forEach(function (p) { p.classList.toggle('active', p.id === 'tab-' + name); });
    if (scroll) {
      var bar = document.querySelector('.tabbar');
      if (bar) { window.scrollTo({ top: bar.offsetTop, behavior: 'smooth' }); }
    }
  }
  btns.forEach(function (b) {
    b.addEventListener('click', function () {
      var n = b.getAttribute('data-tab');
      activate(n, true);
      history.replaceState(null, '', '#' + n);
    });
  });
  var initial = (location.hash || '').replace('#', '');
  if (valid[initial]) { activate(initial, false); }
})();
"""

# Frontend tab: a prospect picker that swaps the server-rendered SC-facing view
# and makes the slot cards selectable. Reuses the same #workflow-data payload as
# the simulator (each entry carries a `frontendHtml`), so it can't drift from the
# pipeline output. Selection uses one delegated listener on the container, so it
# keeps working after the innerHTML is swapped for a different prospect.
_FRONTEND_JS = """
(function () {
  var host = document.getElementById('fe-view');
  var chipsEl = document.getElementById('fe-chips');
  if (!host || !chipsEl) { return; }
  var DATA = JSON.parse(document.getElementById('workflow-data').textContent);
  var keys = Object.keys(DATA);

  // Delegated so it survives host.innerHTML swaps: click (or Enter/Space on) a
  // selectable slot card to select it and update the confirm bar's label.
  function selectCard(card) {
    if (!card || !host.contains(card)) { return; }
    var cards = host.querySelectorAll('.fe-opt.selectable');
    for (var i = 0; i < cards.length; i++) {
      cards[i].classList.remove('selected');
      cards[i].setAttribute('aria-pressed', 'false');
    }
    card.classList.add('selected');
    card.setAttribute('aria-pressed', 'true');
    var sel = host.querySelector('#fe-sel');
    if (sel) { sel.textContent = card.getAttribute('data-when') || sel.textContent; }
  }
  host.addEventListener('click', function (ev) {
    selectCard(ev.target.closest ? ev.target.closest('.fe-opt.selectable') : null);
  });
  host.addEventListener('keydown', function (ev) {
    if (ev.key !== 'Enter' && ev.key !== ' ' && ev.key !== 'Spacebar') { return; }
    var card = ev.target.closest ? ev.target.closest('.fe-opt.selectable') : null;
    if (card) { ev.preventDefault(); selectCard(card); }
  });

  function show(key, btn) {
    host.innerHTML = DATA[key].frontendHtml || '';
    var chips = chipsEl.querySelectorAll('.chip-btn');
    for (var i = 0; i < chips.length; i++) { chips[i].classList.remove('selected'); }
    if (btn) { btn.classList.add('selected'); }
  }

  keys.forEach(function (key, i) {
    var b = document.createElement('button');
    b.className = 'chip-btn';
    b.innerHTML = DATA[key].name;  // server-escaped, so decode entities (e.g. &amp;)
    b.title = DATA[key].address;
    b.addEventListener('click', function () { show(key, b); });
    chipsEl.appendChild(b);
    if (i === 0) { show(key, b); }
  });
})();
"""


def _esc(text: object) -> str:
    return html.escape(str(text))


def _win(window_str: Optional[str]) -> str:
    if not window_str or window_str == "n/a":
        return "any"
    return window_str.replace("-", "–")


def _pref_fact(result: RecommendationResult) -> str:
    slot = result.customer.preferred_slot
    if slot is None:
        return "no slot preference"
    return f"prefers {slot.day.value} {_win(fmt_window(slot.window))}"


def _factor_value(cand: CandidateEvaluation, name: str) -> float:
    for f in cand.factor_scores:
        if f.name == name:
            return f.value
    return 0.0


def _factor_bars(factor_scores) -> str:
    rows = []
    for f in factor_scores:
        label = FACTOR_LABEL.get(f.name, f.name)
        pct = round(f.value * 100)
        rows.append(
            f'<div class="factor"><span class="fname">{_esc(label)}</span>'
            f'<div class="bar"><span style="width:{pct}%"></span></div>'
            f'<span class="fval">{f.value:.2f}</span></div>'
        )
    return "".join(rows)


def _route_checks(cand: CandidateEvaluation) -> str:
    checks = []
    for oc in cand.constraint_outcomes:
        sym = '<span class="ok">✓</span>' if oc.passed else '<span class="no">✗</span>'
        label = CONSTRAINT_LABEL.get(oc.name, oc.name)
        checks.append(f"<li>{sym} {_esc(label)}: {_esc(oc.detail)}</li>")
    return "".join(checks)


def _infeasible_card(cand: CandidateEvaluation) -> str:
    """A routecard for an infeasible route (no score/slots -- just why it failed)."""
    r = cand.route
    title = f"{_esc(r.route_id)} · {_esc(r.name)} · {r.day.value} · {cand.distance_miles:.1f} mi"
    return (
        f'<div class="route routecard" data-route-id="{_esc(r.route_id)}"><div class="rtitle">'
        f"<span>{title}</span><span class=\"status bad\">INFEASIBLE</span></div>"
        f'<ul class="checks">{_route_checks(cand)}</ul></div>'
    )


def _feasible_route_card(cand: CandidateEvaluation, winner_id, threshold) -> str:
    """A routecard for a feasible route scored at the ROUTE level (route-slot
    scoring off): score bar + the route's weighted-factor bars."""
    r = cand.route
    title = f"{_esc(r.route_id)} · {_esc(r.name)} · {r.day.value} · {cand.distance_miles:.1f} mi"
    rec_tag = ' <span class="rectag">★ recommended</span>' if r.route_id == winner_id else ""
    status = f'<span class="status ok">FEASIBLE · {cand.total_score:.2f}</span>'
    score_cls = "g" if cand.total_score >= threshold else "a"
    pct = round(cand.total_score * 100)
    body = (
        f'<div class="score"><div class="bar">'
        f'<span class="{score_cls}" style="width:{pct}%"></span></div></div>'
        f'<div class="factors">{_factor_bars(cand.factor_scores)}</div>'
    )
    return (
        f'<div class="route routecard" data-route-id="{_esc(r.route_id)}"><div class="rtitle">'
        f"<span>{title}{rec_tag}</span>{status}</div>"
        f'{body}<ul class="checks">{_route_checks(cand)}</ul></div>'
    )


# Canonical order of the route-slot factors, so "Slot match" always sits in the
# same place -- even when it's shown as un-scored (no stated preference).
_RS_FACTOR_ORDER = [
    FACTOR_GEO_CLUSTERING,
    FACTOR_CAPACITY_BUFFER,
    FACTOR_WINDOW_MATCH,
    FACTOR_SLOT_AVAILABILITY,
]


def _slot_factor_bars(factor_scores, has_preference: bool) -> str:
    """Factor bars for one route-slot, in canonical order. Slot match is always
    listed: when the prospect stated no preference it is *excluded from the
    weighted total* (rather than given an arbitrary neutral), so it's shown greyed
    with a clear 'not scored' note instead of a bar."""
    by_name = {f.name: f for f in factor_scores}
    rows = []
    for name in _RS_FACTOR_ORDER:
        f = by_name.get(name)
        label = FACTOR_LABEL.get(name, name)
        if f is not None:
            pct = round(f.value * 100)
            rows.append(
                f'<div class="factor"><span class="fname">{_esc(label)}</span>'
                f'<div class="bar"><span style="width:{pct}%"></span></div>'
                f'<span class="fval">{f.value:.2f}</span></div>'
            )
        elif name == FACTOR_WINDOW_MATCH and not has_preference:
            rows.append(
                f'<div class="factor na"><span class="fname">{_esc(label)}</span>'
                '<span class="na-note" title="The prospect stated no preferred slot, '
                'so slot match is excluded from the score and the other weights '
                'renormalize">not scored · no preferred slot given</span></div>'
            )
    return "".join(rows)


def _slot_option(cand: CandidateEvaluation, ss, winner_id, threshold, has_preference) -> str:
    """One candidate (route, slot) inside a route card: its window, its own
    route-slot score bar, and its per-slot factor bars. The recommended slot is
    highlighted + starred."""
    is_rec = cand.route.route_id == winner_id and ss.slot.window == cand.chosen_window
    win = _win(fmt_window(ss.slot.window))
    star = ' <span class="rectag">★ recommended</span>' if is_rec else ""
    score_cls = "g" if ss.total_score >= threshold else "a"
    pct = round(ss.total_score * 100)
    return (
        f'<div class="slot-option{" rec" if is_rec else ""}">'
        f'<div class="slot-head"><span class="slot-win">◆ {win}</span>{star}'
        f'<span class="slot-score">score {ss.total_score:.2f}</span></div>'
        f'<div class="score"><div class="bar">'
        f'<span class="{score_cls}" style="width:{pct}%"></span></div></div>'
        f'<div class="factors">{_slot_factor_bars(ss.factor_scores, has_preference)}</div></div>'
    )


def _route_slot_group_card(cand: CandidateEvaluation, winner_id, threshold, has_preference) -> str:
    """A routecard for one feasible ROUTE on the route-slot path: the route header
    + constraint checks once (no per-slot repetition), then its candidate slots
    listed inside (each with its own score + factor bars). Mirrors the per-route
    delivery-window panel."""
    r = cand.route
    title = f"{_esc(r.route_id)} · {_esc(r.name)} · {r.day.value} · {cand.distance_miles:.1f} mi"
    has_rec = r.route_id == winner_id and any(
        ss.slot.window == cand.chosen_window for ss in cand.scored_slots
    )
    rec_tag = ' <span class="rectag">★ recommended</span>' if has_rec else ""
    slots = "".join(
        _slot_option(cand, ss, winner_id, threshold, has_preference)
        for ss in sorted(cand.scored_slots, key=lambda s: s.total_score, reverse=True)
    )
    return (
        f'<div class="route routecard" data-route-id="{_esc(r.route_id)}"><div class="rtitle">'
        f'<span>{title}{rec_tag}</span><span class="status ok">FEASIBLE</span></div>'
        f'<ul class="checks">{_route_checks(cand)}</ul>'
        f'<div class="slot-options">{slots}</div></div>'
    )


def _route_cards(result: RecommendationResult, config: Config) -> str:
    """The "Routes the agent evaluated" section. Open by default (the
    ``<summary>`` still collapses it). Each card carries ``data-route-id`` so the
    web app can tint it to match the route's colour on the map.

    On the route-slot path it renders one card per **route** (feasible ranked
    first), with that route's candidate **slots** listed inside -- each slot with
    its own score and factor bars -- so the route info isn't repeated per slot and
    the section mirrors the per-route delivery-window panels. Off that path it
    keeps one card per route with the route's own score."""
    rec = result.recommendation
    winner_id = rec.recommended_route_id
    feasible = list(result.ranked_feasible)
    infeasible = [e for e in result.candidates_considered if not e.feasible]
    route_slot_mode = any(e.scored_slots for e in feasible)

    cards = []
    if route_slot_mode:
        threshold = config.route_slot_score_threshold
        has_pref = result.customer.preferred_slot is not None
        cards.extend(
            _route_slot_group_card(e, winner_id, threshold, has_pref) for e in feasible
        )
    else:
        threshold = config.total_score_threshold
        cards.extend(_feasible_route_card(e, winner_id, threshold) for e in feasible)
    cards.extend(_infeasible_card(e) for e in infeasible)
    n = len(result.candidates_considered)

    return (
        f'<details class="routes" open><summary>Routes the agent evaluated ({n})</summary>'
        f'<div class="routelist">{"".join(cards)}</div></details>'
    )


def _route_rows(candidates: list[CandidateEvaluation]) -> str:
    rows = []
    for cand in candidates:
        r = cand.route
        if cand.feasible:
            status_cls, status = "ok", f"FEASIBLE · {cand.total_score:.2f}"
        else:
            status_cls, status = "bad", "INFEASIBLE"
        title = f"{_esc(r.route_id)} · {_esc(r.name)} · {r.day.value} · {cand.distance_miles:.1f} mi"
        rows.append(
            f'<div class="route"><div class="rtitle"><span>{title}</span>'
            f'<span class="status {status_cls}">{status}</span></div>'
            f'<ul class="checks">{_route_checks(cand)}</ul></div>'
        )
    return "".join(rows)


def _reason_block(
    rec: SlotRecommendation, reason_label: str, reasoning_override: Optional[str] = None
) -> str:
    """The 'why' panel. When ``reasoning_override`` is given (the live agent's own
    recommendation narration), render exactly that text -- so the panel shows the
    same words the chat box did, instead of a separately-rendered version. When a
    grounded route-slot pick populated the structured fields, render them as
    distinct sections (summary · reasons · trade-off · runner-up · vs-default);
    otherwise fall back to the flat reasoning line."""
    if reasoning_override and reasoning_override.strip():
        body = _esc(reasoning_override.strip()).replace("\n", "<br>")
        return f'<div class="reason"><span class="lbl">{reason_label}</span>{body}</div>'
    if not rec.decision_summary:
        return f'<div class="reason"><span class="lbl">{reason_label}</span>{_esc(rec.reasoning)}</div>'

    parts = [
        f'<span class="lbl">{reason_label}</span>',
        f'<span class="summary">{_esc(rec.decision_summary)}</span>',
    ]
    if rec.primary_reasons:
        items = "".join(f"<li>{_esc(r)}</li>" for r in rec.primary_reasons)
        parts.append(f'<ul class="reasons">{items}</ul>')
    if rec.key_tradeoff:
        parts.append(
            f'<div class="sub"><span class="k">Trade-off</span>{_esc(rec.key_tradeoff)}</div>'
        )
    if rec.runner_up:
        parts.append(
            f'<div class="sub"><span class="k">Runner-up</span>{_esc(rec.runner_up)}</div>'
        )
    if rec.default_comparison:
        parts.append(f'<div class="vs">{_esc(rec.default_comparison)}</div>')
    return f'<div class="reason">{"".join(parts)}</div>'


def _example_card(
    result: RecommendationResult,
    include_routes: bool = True,
    reasoning_override: Optional[str] = None,
) -> str:
    c = result.customer
    rec = result.recommendation
    pill_cls, pill_text = DECISION_PILL[rec.decision]
    score_cls = SCORE_CLASS[rec.decision]
    score_color = SCORE_TEXT_COLOR[rec.decision]
    n_routes = len(result.candidates_considered)

    if rec.decision == Decision.ESCALATED_NO_FEASIBLE_SLOT:
        score_bar = (
            f'<div class="bar"><span class="{score_cls}" style="width:100%"></span></div>'
            f"<small>no candidate passed the hard rules</small>"
        )
    else:
        note = "<small>below the auto-assign bar</small>" if score_cls == "a" else ""
        score_bar = (
            f'<div class="bar"><span class="{score_cls}" style="width:{round(rec.total_score * 100)}%">'
            f"</span></div>{note}"
        )

    slot_html = ""
    if rec.recommended_route_id:
        prefix = "→ proposed: " if rec.requires_human_review else "→ "
        slot_html = (
            f'<p class="slot">{prefix}<b>{_esc(rec.recommended_route_id)} · '
            f"{_esc(rec.recommended_route_name)}</b> · {_esc(rec.recommended_day)} · "
            f'<b>{_win(rec.recommended_window)}</b> '
            f'<small style="color:var(--muted)">(score {rec.total_score:.2f})</small></p>'
        )

    # Show the winning route-slot's factor bars, always listing Slot match: when
    # the prospect stated no preference it's excluded from the score, so it shows
    # the same greyed "not scored" pill as the evaluated-route slot cards.
    has_pref = result.customer.preferred_slot is not None
    factors_html = (
        f'<div class="factors">{_slot_factor_bars(rec.factor_breakdown, has_pref)}</div>'
        if rec.factor_breakdown
        else ""
    )
    reason_label = "Why the agent escalated" if rec.requires_human_review else "Why the agent chose this"

    cnum_text = _esc(c.customer_number) if c.customer_number else "new prospect — no Sysco number yet"
    routes_html = (
        f'<details class="routes"><summary>Routes the agent evaluated ({n_routes})</summary>'
        f'<div class="routelist">{_route_rows(result.candidates_considered)}</div></details>'
        if include_routes
        else ""
    )
    return f"""
      <article class="result">
        <div class="rhead">
          <div class="cnum">{cnum_text}</div>
          <h3>{_esc(c.name)}</h3>
          <div class="addr">{_esc(c.address)}</div>
          <div class="facts">
            <span class="fact">{c.order_quantity_cases} cases</span>
            <span class="fact">{_esc(_pref_fact(result))}</span>
            <span class="fact">{n_routes} routes evaluated</span>
          </div>
        </div>
        <div class="rbody">
          <div class="decision-row">
            <span class="pill {pill_cls}">{pill_text}</span>
            <span style="font-size:13px;color:var(--muted)">total score
              <b style="color:{score_color}">{rec.total_score:.0%}</b></span>
          </div>
          <div class="score">{score_bar}</div>
          {slot_html}
          {factors_html}
          {_reason_block(rec, reason_label, reasoning_override)}
          {routes_html}
        </div>
      </article>"""


def _rs_factor_formula(name: str, config: Config) -> str:
    """The math skeleton for one route-slot factor, with the config constants
    plugged in -- so a reader can reproduce the value from the cited inputs."""
    cref = config.cluster_reference_miles
    ceiling = config.max_utilization_after_assignment
    margin = config.capacity_buffer_safety_margin
    safe = ceiling - margin
    if name == FACTOR_GEO_CLUSTERING:
        return f"clamp(1 − avg_mi ÷ {cref:.0f})"
    if name == FACTOR_CAPACITY_BUFFER:
        return f"1.00 if util ≤ {safe:.0%}, else clamp(({ceiling:.0%} − util) ÷ {margin:.0%})"
    if name == FACTOR_WINDOW_MATCH:
        return "overlap-min ÷ preferred-min (0 on wrong day)"
    if name == FACTOR_SLOT_AVAILABILITY:
        return "1 ÷ (1 + Σ tier-harm over overlapping stops)"
    return ""


# Per-dimension accent colours for the route-slot factor rows: each factor gets
# a small tick in its own hue so the four dimensions are told apart at a glance.
_RS_FACTOR_COLOR = {
    FACTOR_GEO_CLUSTERING: "#1257a6",
    FACTOR_CAPACITY_BUFFER: "#9a6700",
    FACTOR_WINDOW_MATCH: "#5b3fb0",
    FACTOR_SLOT_AVAILABILITY: "#0e7c7b",
}


def _route_slot_details(e, ss, config: Config, is_win: bool) -> str:
    """One (route, slot) option as a collapsible ``<details>``: a compact score
    line is the summary; the per-factor breakdown (formula + cited inputs +
    weight + contribution) and the weighted total live inside, collapsed by
    default. Same numbers as before -- only laid out to be scannable and
    checkable on demand rather than as a wall of text."""
    win = _win(fmt_window(ss.slot.window))
    star = ' <span class="rs-star">★</span>' if is_win else ""

    rows = []
    weight_sum = 0.0
    terms = []
    for fs in ss.factor_scores:
        label = FACTOR_LABEL.get(fs.name, fs.name)
        formula = _rs_factor_formula(fs.name, config)
        color = _RS_FACTOR_COLOR.get(fs.name, "var(--muted)")
        rows.append(
            f'<div class="rs-factor">'
            f'<span class="rs-ftick" style="background:{color}"></span>'
            f'<span class="rs-fname">{_esc(label)}</span>'
            f'<span class="rs-fnums"><span class="rs-fval">{fs.value:.2f}</span>'
            f'<span class="rs-fwt">× {fs.weight:.2f}</span>'
            f'<span class="rs-fcontrib">→ {fs.weighted:.2f}</span></span>'
            f'<span class="rs-fdetail"><span class="rs-fformula">{_esc(formula)}</span>'
            f"{_esc(fs.detail)}</span></div>"
        )
        weight_sum += fs.weight
        terms.append(f"{fs.weight:.2f}×{fs.value:.2f}")

    total = (
        '<div class="rs-total"><span class="rs-total-lbl">Route-slot score</span>'
        f'<span class="rs-total-expr">= ({" + ".join(terms)}) ÷ {weight_sum:.2f}</span>'
        f'<span class="rs-total-eq">= {ss.total_score:.2f}</span></div>'
    )

    return (
        f'<details class="rs-details{" winner" if is_win else ""}">'
        '<summary class="rs-head">'
        '<span class="rs-caret">▶</span>'
        f'<span class="rs-id">{_esc(e.route.route_id)}</span>'
        f'<span class="rs-name">{_esc(e.route.name)}</span>'
        f'<span class="rs-win">@ {win}</span>'
        '<span class="rs-spacer"></span>'
        '<span class="rs-showmath"></span>'
        f'<span class="rs-score">{ss.total_score:.2f}{star}</span>'
        "</summary>"
        f'<div class="rs-math">{"".join(rows)}{total}</div>'
        "</details>"
    )


def _route_slot_score_lines(ranked_feasible, config: Config, rec=None) -> list[str]:
    """Score & Rank narrative for the route-slot path: every candidate slot on
    every feasible route, scored as its own (route, slot) option. Each option is
    a collapsed ``<details>`` row -- score line visible, the full per-factor math
    (formula, cited inputs, weight, weighted total) one click away -- so the
    reader can still validate every number by hand without a wall of text. The
    recommended (route, slot) is flagged."""
    lines = [
        "Each candidate slot on each feasible route is scored as its own (route, slot) option. "
        "Open <b>show the math</b> on any row to see every factor's formula, the numbers it used, "
        "and its weight — so the math is checkable:"
    ]
    winner_id = rec.recommended_route_id if rec else None
    for e in ranked_feasible:
        for ss in sorted(e.scored_slots, key=lambda s: s.total_score, reverse=True):
            is_win = bool(winner_id) and e.route.route_id == winner_id and ss.slot.window == e.chosen_window
            lines.append(_route_slot_details(e, ss, config, is_win))
    return lines


def _sim_steps(result: RecommendationResult, config: Config) -> list[dict]:
    c = result.customer
    rec = result.recommendation
    loc = c.location
    cands = result.candidates_considered
    n = len(cands)
    feasible = [e for e in cands if e.feasible]
    slot = c.preferred_slot
    pref = f"{slot.day.value} {_win(fmt_window(slot.window))}" if slot else "any"
    gw = config.factor_weights[FACTOR_GEO_CLUSTERING]
    cw = config.factor_weights[FACTOR_CAPACITY_BUFFER]
    ww = config.factor_weights[FACTOR_WINDOW_MATCH]
    cref = config.cluster_reference_miles
    ceiling = config.max_utilization_after_assignment
    margin = config.capacity_buffer_safety_margin
    safe = ceiling - margin

    if c.customer_number:
        id_line = f'Customer number <b>{_esc(c.customer_number)}</b> <span class="ok">✓ valid</span>'
    else:
        id_line = (
            f'No Sysco customer number yet (new prospect) — using <b>address</b> as the '
            f'identifier: <b>{_esc(c.address)}</b>'
        )
    intake = [
        id_line,
        f"Order quantity: <b>{c.order_quantity_cases}</b> cases",
        f"Preferred slot (day + time): <b>{_esc(pref)}</b>",
    ]
    geo = [
        f"Geocoded address → <b>({loc.latitude:.4f}, {loc.longitude:.4f})</b>",
        f"Selected the {n} nearest routes:",
    ]
    for e in cands:
        geo.append(f"• {_esc(e.route.route_id)} · {_esc(e.route.name)} — {e.distance_miles:.1f} mi")

    con = [f"{len(feasible)} of {n} routes passed every hard rule:"]
    for e in cands:
        if e.feasible:
            con.append(f'• {_esc(e.route.route_id)}: <span class="ok">FEASIBLE</span>')
        else:
            failed = ", ".join(CONSTRAINT_LABEL.get(o.name, o.name) for o in e.failed_constraints)
            con.append(
                f'• {_esc(e.route.route_id)}: <span class="no">INFEASIBLE</span> — failed {_esc(failed)}'
            )

    if result.ranked_feasible and config.use_route_slot_scoring:
        score = _route_slot_score_lines(result.ranked_feasible, config, rec)
    elif result.ranked_feasible:
        score = ["Each dimension is normalized to 0–1, then combined by weight:"]
        for e in result.ranked_feasible:
            ctx = build_context(c, e.route, config)
            g = _factor_value(e, FACTOR_GEO_CLUSTERING)
            b = _factor_value(e, FACTOR_CAPACITY_BUFFER)
            w = _factor_value(e, FACTOR_WINDOW_MATCH)
            score.append(
                f"• <b>{_esc(e.route.route_id)} · {_esc(e.route.name)}</b> "
                f"→ weighted score <b>{e.total_score:.2f}</b>"
            )
            score.append(
                f'<span class="calc">↳ clustering = clamp(1 − {ctx.avg_stop_distance_miles:.1f} ÷ '
                f"{cref:.0f} mi) = <b>{g:.2f}</b> · weight {gw:.2f}</span>"
            )
            if ctx.utilization_after <= safe:
                score.append(
                    f'<span class="calc">↳ capacity buffer = 1.00 flat '
                    f"({ctx.utilization_after:.0%} full is under the {safe:.0%} safe line) = "
                    f"<b>{b:.2f}</b> · weight {cw:.2f}</span>"
                )
            else:
                score.append(
                    f'<span class="calc">↳ capacity buffer = clamp(({ceiling:.0%} − '
                    f"{ctx.utilization_after:.0%}) ÷ {margin:.0%}) = <b>{b:.2f}</b> · "
                    f"weight {cw:.2f}</span>"
                )
            if slot is None:
                score.append(
                    f'<span class="calc">↳ slot match = neutral (no preferred slot) = '
                    f"<b>{w:.2f}</b> · weight {ww:.2f}</span>"
                )
            else:
                pd = max(1, duration_minutes(slot.window))
                day_ok = e.route.day == slot.day
                day_sym = "✓" if day_ok else "✗"
                if day_ok:
                    score.append(
                        f'<span class="calc">↳ slot match = day({e.route.day.value}{day_sym}'
                        f"pref {slot.day.value}) → time({ctx.window_overlap_minutes}÷{pd} min) = "
                        f"<b>{w:.2f}</b> · weight {ww:.2f}</span>"
                    )
                else:
                    score.append(
                        f'<span class="calc">↳ slot match = day({e.route.day.value}{day_sym}'
                        f"pref {slot.day.value}) → wrong day, no credit = "
                        f"<b>{w:.2f}</b> · weight {ww:.2f}</span>"
                    )
            score.append(
                f'<span class="calc">↳ total = {gw:.2f}×{g:.2f} + {cw:.2f}×{b:.2f} + '
                f"{ww:.2f}×{w:.2f} = <b>{e.total_score:.2f}</b></span>"
            )
    else:
        score = ["No feasible routes survived the hard rules — nothing to score."]

    rs = config.use_route_slot_scoring
    bar = config.route_slot_score_threshold if rs else config.total_score_threshold
    winner_label = "winning route-slot" if rs else "winning route"
    decide = [
        f"Decision: <b>{DECISION_SHORT[rec.decision]}</b>",
        f"Total score for the {winner_label}: <b>{rec.total_score:.0%}</b> "
        f"(auto-assign bar {bar:.0%})",
    ]
    if rec.recommended_route_id:
        decide.append(
            f"Proposed slot: <b>{_esc(rec.recommended_route_id)} · "
            f"{_esc(rec.recommended_day)} · {_win(rec.recommended_window)}</b>"
        )
    else:
        decide.append("No valid slot — routing specialist required.")

    return [
        {
            "title": "Intake",
            "action": "The agent validates the intake profile (address, order quantity, preferred slot).",
            "lines": intake,
        },
        {
            "title": "Geo-Lookup",
            "action": "The agent geocodes the address and selects the nearest candidate routes.",
            "lines": geo,
        },
        {
            "title": "Constraint Check",
            "action": "The agent applies the two hard rules (serviceability, capacity) and removes infeasible routes.",
            "lines": con,
        },
        {
            "title": "Score & Rank",
            "action": (
                "The agent scores each feasible (route, slot) pair on the weighted factors "
                "— including slot availability — and ranks them."
                if config.use_route_slot_scoring
                else "The agent scores each feasible route on the weighted factors (with the math) "
                "and ranks them."
            ),
            "lines": score,
        },
        {
            "title": "Recommend / Decide",
            "action": (
                "The agent picks the best route-slot, checks its total against the auto-assign "
                "bar, and decides."
                if config.use_route_slot_scoring
                else "The agent picks the best slot, checks its total score against the auto-assign "
                "bar, and decides."
            ),
            "lines": decide,
        },
    ]


def _scoring_section_route_slot(config: Config) -> str:
    """The 'how the agent scores' explainer for the route-slot path: the unit is
    a (route, slot) pair, geo/capacity are route-level, window_match and slot
    availability are slot-level, and window_match is dropped without a preference."""
    gw = config.rs_weight_geo
    cw = config.rs_weight_capacity
    ww = config.rs_weight_window
    aw = config.rs_weight_availability
    cref = config.cluster_reference_miles
    thr = config.route_slot_score_threshold
    ceiling = config.max_utilization_after_assignment
    safe = ceiling - config.capacity_buffer_safety_margin
    hi = config.slot_tier_harm_high
    mid = config.slot_tier_harm_mid
    lo = config.slot_tier_harm_low
    unk = config.slot_tier_harm_unknown
    return f"""
    <span class="eyebrow">How the agent scores &amp; ranks</span>
    <h2>Every route-<em>slot</em> is scored on its own</h2>
    <p class="sub">The decision unit is the <b>(route, time-slot) pair</b>: each candidate slot on each
      feasible route is scored separately, so a route's <em>slot availability</em> — not just its capacity and
      location — decides which route wins. Two factors are <b>route-level</b> (shared by all of a route's
      slots) and two are <b>slot-level</b>. Each is normalized to 0–1 and combined by weight; deterministic
      Python, same inputs → same score.</p>
    <div class="grid-2">
      <div class="card"><div class="icon">🧭</div><h3>Geographic clustering · weight {gw:.2f} <span
        style="font-size:11px;color:var(--muted)">route-level</span></h3>
        <p>How tightly the customer sits within the route's existing cluster of stops. Closer = higher.</p>
        <div class="formula">score = clamp( 1 − <b>avg_miles_to_stops</b> ÷ {cref:.0f} , 0, 1 )</div></div>
      <div class="card"><div class="icon">🛡️</div><h3>Capacity buffer · weight {cw:.2f} <span
        style="font-size:11px;color:var(--muted)">route-level</span></h3>
        <p>How safely under the capacity ceiling the truck stays once this order is added. Flat while
          comfortably safe (≤ {safe:.0%} full), decaying to 0 at the {ceiling:.0%} ceiling.</p></div>
      <div class="card"><div class="icon">🎯</div><h3>Slot match (day + time) · weight {ww:.2f} <span
        style="font-size:11px;color:var(--muted)">slot-level</span></h3>
        <p>How much <b>this</b> slot covers the customer's preferred day + time window. The day is a gate;
          then it's the overlap fraction. <b>Dropped entirely</b> when the prospect states no preference — the
          score simply renormalizes over the other factors (no arbitrary neutral value).</p></div>
      <div class="card"><div class="icon">🪟</div><h3>Slot availability · weight {aw:.2f} <span
        style="font-size:11px;color:var(--muted)">slot-level</span></h3>
        <p>How <b>open</b> this slot is — few / low-tier committed stops already in it. Weighted so we avoid
          crowding the most valued customers.</p>
        <div class="formula">openness = 1 ÷ ( 1 + <b>Σ harm(incumbent)</b> over stops sharing the window )</div>
        <p style="margin-top:8px;font-size:12.5px">Harm per overlapping committed stop, by tier:
          <b>5 / Perks {hi:.1f}</b> &gt; <b>4 {mid:.1f}</b> &gt; the prospect &gt; <b>Other {lo:.1f}</b>
          (unknown {unk:.1f}). A window full of Other-tier stops still scores open; one shared by tier-5 /
          Perks incumbents scores contended.</p></div>
    </div>

    <div style="height:16px"></div>
    <div class="card">
      <h3 style="margin-top:0">Best route-slot, then the auto-assign decision</h3>
      <div class="formula">total = ( {gw:.2f}·clustering + {cw:.2f}·capacity + {ww:.2f}·window* + {aw:.2f}·availability ) ÷ Σ&nbsp;active&nbsp;weights</div>
      <p style="margin-top:14px"><span style="font-size:11px;color:var(--muted)">*window only when a
        preference is stated; otherwise it's absent and the remaining weights renormalize.</span> This
        weighted total ranks the route-slots; the {thr:.0%} bar is the deterministic reference. With the
        backend configured, an <b>LLM reasons over all feasible route-slots and makes the
        recommend-vs-escalate call itself</b> — a confident recommend ships on one verified call, while an
        escalate (or low-confidence recommend) is resampled and cleared by consensus before it may
        auto-assign. It picks the winning route-slot by index from the feasible set and cites the facts;
        every choice is verified in code, and on any failure it falls back to the deterministic
        {thr:.0%}-bar decision — the reproducible floor.</p>
    </div>"""


def _scoring_section(config: Config) -> str:
    if config.use_route_slot_scoring:
        return _scoring_section_route_slot(config)
    gw = config.factor_weights[FACTOR_GEO_CLUSTERING]
    cw = config.factor_weights[FACTOR_CAPACITY_BUFFER]
    ww = config.factor_weights[FACTOR_WINDOW_MATCH]
    total_w = gw + cw + ww
    cref = config.cluster_reference_miles
    neutral = config.window_neutral_score
    thr = config.total_score_threshold
    ceiling = config.max_utilization_after_assignment
    margin = config.capacity_buffer_safety_margin
    safe = ceiling - margin
    return f"""
    <span class="eyebrow">How the agent scores &amp; ranks</span>
    <h2>Exactly how each dimension is scored</h2>
    <p class="sub">Only routes that pass every hard rule reach this stage. The agent scores each on three
      dimensions, normalizes each to 0–1 (<span style="font-family:var(--mono)">clamp</span> keeps it in
      range), and combines them by weight. This is deterministic Python — the same inputs always give the
      same score.</p>
    <div class="grid-3">
      <div class="card"><div class="icon">🧭</div><h3>Geographic clustering · weight {gw:.2f}</h3>
        <p>How tightly the customer sits within the route's existing cluster of stops. Closer = higher.</p>
        <div class="formula">score = clamp( 1 − <b>avg_miles_to_stops</b> ÷ {cref:.0f} , 0, 1 )</div>
        <p style="margin-top:8px;font-size:12.5px">Distance is the average great-circle miles to the route's
          committed stops; at {cref:.0f} mi the score reaches 0.</p></div>
      <div class="card"><div class="icon">🛡️</div><h3>Capacity buffer · weight {cw:.2f}</h3>
        <p>How safely under the capacity ceiling the truck stays once this order is added. Flat while
          comfortably safe; only decays as the truck approaches its limit.</p>
        <div class="formula">score = 1.0                                     if utilization ≤ {safe:.0%}
<br/>score = clamp( ({ceiling:.0%} − utilization) ÷ {margin:.0%} , 0, 1 )   otherwise</div>
        <p style="margin-top:8px;font-size:12.5px">Utilization stays flat at 1.0 up to {safe:.0%} full (the
          {margin:.0%}-point safety margin below the {ceiling:.0%} ceiling); past that it falls straight to 0
          exactly at the ceiling. Two routes that are both comfortably safe score the same — only a route
          that is genuinely getting full is marked down.</p></div>
      <div class="card"><div class="icon">🎯</div><h3>Slot match (day + time) · weight {ww:.2f}</h3>
        <p>How well the route matches the customer's preferred <b>slot</b> — which always includes a
          <b>day of week</b> plus a time-of-day window. A <b>soft preference</b>: it shapes the score but
          never eliminates a route.</p>
        <div class="formula">score = 0                                                     if <b>route_day</b> ≠ <b>preferred_day</b>, or zero time overlap
<br/>score = clamp( <b>overlap_minutes</b> ÷ <b>preferred_window_minutes</b> , 0, 1 )        otherwise</div>
        <p style="margin-top:8px;font-size:12.5px">The day is a gate, not partial credit: a route only earns any
          score once it lands on the customer's preferred day, and then only for however much of the preferred
          window it actually covers. Wrong day, or right day with no time overlap, scores 0 — half-right isn't a
          match. If the customer states no slot, a neutral {neutral:.2f} is used instead.</p></div>
    </div>

    <div style="height:16px"></div>
    <div class="card">
      <h3 style="margin-top:0">Final score, then the auto-assign decision</h3>
      <p>The overall score is the weighted average of the three dimensions:</p>
      <div class="formula">total_score = ( {gw:.2f}·clustering + {cw:.2f}·capacity + {ww:.2f}·window ) ÷ {total_w:.2f}</div>
      <p style="margin-top:14px">The agent ranks feasible routes by <em>total_score</em> and recommends the winner.
        That same number is what gates the decision<span style="font-size:11px;color:var(--muted)"> — there's
        no separate "confidence" formula on top of it</span>. The agent <b>auto-assigns</b> when the winner's
        own total score is ≥ {thr:.0%}; otherwise it <b>escalates</b> for a specialist to review.</p>
      <p style="margin-top:12px;font-size:12.5px;color:var(--muted)">By design, a route's own score is never
        discounted just because another candidate scored nearly as well — two routes tied at a high score
        both clear the bar, and either is a safe pick. A route only gets flagged when <em>its own</em> score
        is mediocre, not because it happens to have close competition.</p>
    </div>"""


def _config_sources(config: Config, results: list[RecommendationResult]) -> str:
    """A 'where do these numbers come from' section, sourced from config + mock data."""
    gw = config.factor_weights[FACTOR_GEO_CLUSTERING]
    cw = config.factor_weights[FACTOR_CAPACITY_BUFFER]
    ww = config.factor_weights[FACTOR_WINDOW_MATCH]

    routes = {}
    for r in results:
        for e in r.candidates_considered:
            routes[(e.route.route_id, e.route.day.value)] = e.route
    caps = sorted({rt.vehicle_capacity_cases for rt in routes.values()})
    radii = sorted({rt.service_radius_miles for rt in routes.values()})
    cap_range = f"{caps[0]}–{caps[-1]}" if len(caps) > 1 else f"{caps[0]}"
    radius_range = f"{radii[0]:.0f}–{radii[-1]:.0f}" if len(radii) > 1 else f"{radii[0]:.0f}"

    def row(k: str, v: str, src: str) -> str:
        return f'<li><span class="k">{k} <span class="src">— {src}</span></span><span class="v">{v}</span></li>'

    safe_utilization = config.max_utilization_after_assignment - config.capacity_buffer_safety_margin
    if config.use_route_slot_scoring:
        scoring_rows = [
            row(
                "Route-slot weights (geo/cap/win/avail)",
                f"{config.rs_weight_geo:.2f} / {config.rs_weight_capacity:.2f} / "
                f"{config.rs_weight_window:.2f} / {config.rs_weight_availability:.2f}",
                "rs_weight_*",
            ),
            row(
                "Slot-openness harm (5·Perks/4/Other/unknown)",
                f"{config.slot_tier_harm_high:.1f} / {config.slot_tier_harm_mid:.1f} / "
                f"{config.slot_tier_harm_low:.1f} / {config.slot_tier_harm_unknown:.1f}",
                "slot_tier_harm_*",
            ),
            row(
                "Route-slot auto-assign bar",
                f"{config.route_slot_score_threshold:.0%}",
                "route_slot_score_threshold",
            ),
        ]
    else:
        scoring_rows = [
            row("No-window neutral score", f"{config.window_neutral_score:.2f}", "window_neutral_score"),
            row("Scoring weights (geo/cap/win)", f"{gw:.2f} / {cw:.2f} / {ww:.2f}", "factor_weights"),
            row(
                "Total score threshold (auto-assign bar)",
                f"{config.total_score_threshold:.0%}",
                "total_score_threshold",
            ),
        ]
    cfg_rows = "".join(
        [
            row("Route capacity ceiling", f"{config.max_utilization_after_assignment:.0%}", "max_utilization_after_assignment"),
            row(
                "Capacity safety margin (safe up to)",
                f"{config.capacity_buffer_safety_margin:.0%} pts (→ {safe_utilization:.0%})",
                "capacity_buffer_safety_margin",
            ),
            row("Clustering reference (score→0)", f"{config.cluster_reference_miles:.0f} mi", "cluster_reference_miles"),
            *scoring_rows,
            row("Serviceability hard cap", f"{config.max_service_distance_miles:.0f} mi", "max_service_distance_miles"),
            row("Candidates evaluated", f"Top-{config.top_n_candidate_routes}", "top_n_candidate_routes"),
        ]
    )
    route_rows = "".join(
        [
            row("Vehicle capacity per route", f"{cap_range} cases", "denominator of utilization %"),
            row("Service radius per route", f"{radius_range} mi", "the serviceability limit"),
            row("Committed stops + locations", "varies", "drive clustering & used capacity"),
            row("Available delivery windows", "varies", "window overlap"),
        ]
    )
    intake_rows = "".join(
        [
            row("Address → geocoded point", "per customer", "geocoding — the primary identifier"),
            row("Order quantity", "cases", "capacity math"),
            row("Preferred slot: day + time (optional)", "soft", "slot scoring only"),
            row("Sysco customer number (optional)", "placeholder", "unset for prospects; matched if on file"),
        ]
    )
    return f"""
    <span class="eyebrow">Where the numbers come from</span>
    <h2>Every threshold, and its source</h2>
    <p class="sub">The formulas above use real values from three places. Config defaults are tunable in
      <span style="font-family:var(--mono)">shared/config.py</span>; per-route numbers (capacity, radius)
      are mock data standing in for a real routing system; the rest comes from the customer's intake.</p>
    <div class="grid-3">
      <div class="card srccard"><h4>⚙️ Config defaults <span class="badge cfg">shared/config.py</span></h4>
        <ul class="srclist">{cfg_rows}</ul></div>
      <div class="card srccard"><h4>🗄️ Route data <span class="badge route">mocked TMS</span></h4>
        <ul class="srclist">{route_rows}</ul>
        <p style="margin-top:10px;font-size:12px;color:var(--muted)">From
          <span style="font-family:var(--mono)">integrations/route_capacity_client.py</span>.</p></div>
      <div class="card srccard"><h4>🧾 Customer intake <span class="badge intake">new customer</span></h4>
        <ul class="srclist">{intake_rows}</ul>
        <p style="margin-top:10px;font-size:12px;color:var(--muted)">Env vars can override any config default
          (e.g. <span style="font-family:var(--mono)">SMART_ASSIGNMENT_MAX_UTILIZATION</span>).</p></div>
    </div>"""


def _slot_rationale_factors(factor_scores, has_preference: bool, order=None) -> str:
    """Like ``_slot_factor_bars`` but annotates each bar with the concrete input
    it cited (from ``FactorScore.detail``) and its weight — the "full details"
    behind each factor's score, for the "why this slot" rationale card. ``order``
    limits/orders which factors are shown (defaults to all four)."""
    by_name = {f.name: f for f in factor_scores}
    rows = []
    for name in order or _RS_FACTOR_ORDER:
        f = by_name.get(name)
        label = FACTOR_LABEL.get(name, name)
        if f is not None:
            pct = round(f.value * 100)
            rows.append(
                '<div class="why-factor"><div class="factor">'
                f'<span class="fname">{_esc(label)}</span>'
                f'<div class="bar"><span style="width:{pct}%"></span></div>'
                f'<span class="fval">{f.value:.2f}'
                f'<small style="color:var(--muted)"> ×{f.weight:.2f}</small></span></div>'
                f'<div class="why-factor-detail">{_esc(f.detail)}</div></div>'
            )
        elif name == FACTOR_WINDOW_MATCH and not has_preference:
            rows.append(
                '<div class="why-factor"><div class="factor na">'
                f'<span class="fname">{_esc(label)}</span>'
                '<span class="na-note">not scored · no preferred slot given</span>'
                "</div></div>"
            )
    return "".join(rows)


# How this specific time WINDOW was placed on the route, in plain language.
_SLOT_BASIS_SENTENCE = {
    "preference_accommodated": (
        "This window was placed to line up with the customer's requested day and time."
    ),
    "between_adjacent_stops": (
        "This window sits between the route's nearest committed stops, so the delivery "
        "drops into the existing time cluster instead of stretching the day."
    ),
    "least_contended": (
        "This is the most open window on the route — the fewest / lowest-tier committed "
        "deliveries already overlap it."
    ),
}

# The two SLOT-level factors: they differ window-to-window on the same route, so
# they are what actually pick the time slot. (Geographic clustering and capacity
# buffer are route-level — they explain the route, not the slot.)
_SLOT_LEVEL_FACTORS = (FACTOR_WINDOW_MATCH, FACTOR_SLOT_AVAILABILITY)


def _tier_label(tier: Optional[str]) -> str:
    if not tier:
        return "unknown tier"
    return "Perks" if tier == "Perks" else f"Tier {tier}"


def _openness_breakdown_html(route, window, config: Config) -> str:
    """The exact openness calculation for the chosen window: its **contention** —
    every committed delivery whose own window overlaps it, each weighted by tier
    ("harm") — summed, then rolled up as ``openness = 1 / (1 + contention)``.
    Contention is the sole input to the slot-availability (openness) score."""
    overlapping = [
        s
        for s in route.committed_stops
        if s.delivery_time_window is not None
        and overlap_minutes(window, s.delivery_time_window) > 0
    ]
    if not overlapping:
        return (
            '<div class="why-calc"><div class="why-calc-head">Contention → openness</div>'
            '<div class="why-calc-sum">no committed delivery overlaps this window → '
            "contention = 0 → openness = 1 ÷ (1 + 0) = <b>1.00</b> (fully open)</div></div>"
        )
    items = "".join(
        f'<li><span class="why-cust">{_esc(s.customer_number)}</span>'
        f'<span class="why-meta">{_esc(_tier_label(s.customer_tier))}</span>'
        f'<span class="why-num">harm {config.tier_harm_weight(s.customer_tier):.2f}</span></li>'
        for s in sorted(
            overlapping, key=lambda s: config.tier_harm_weight(s.customer_tier), reverse=True
        )
    )
    total = sum(config.tier_harm_weight(s.customer_tier) for s in overlapping)
    openness = 1.0 / (1.0 + total)
    n = len(overlapping)
    return (
        '<div class="why-calc"><div class="why-calc-head">Contention → openness — committed '
        "deliveries already overlapping this window, weighted by tier (“harm”)</div>"
        f'<ul class="why-list">{items}</ul>'
        f'<div class="why-calc-sum">contention = Σ harm = <b>{total:.2f}</b> '
        f'({n} overlapping {"delivery" if n == 1 else "deliveries"}) → '
        f"openness = 1 ÷ (1 + {total:.2f}) = <b>{openness:.2f}</b></div></div>"
    )


def _proximity_stops_html(location, route, config: Config) -> str:
    """The nearest committed stops the window was clustered around — the
    "proximity" set: customer number, distance, tier."""
    neighbors = nearest_neighbors(
        location, route.committed_stops, config.slot_neighbor_count, config.slot_neighbor_max_miles
    )
    if not neighbors:
        return ""
    items = "".join(
        f'<li><span class="why-cust">{_esc(n.stop.customer_number)}</span>'
        f'<span class="why-meta">{_esc(_tier_label(n.stop.customer_tier))}</span>'
        f'<span class="why-num">{n.distance_miles:.1f} mi</span></li>'
        for n in neighbors
    )
    return (
        '<div class="why-calc"><div class="why-calc-head">Proximity — nearest committed '
        "stops the window is placed among</div>"
        f'<ul class="why-list">{items}</ul></div>'
    )


def _slot_rationale_html(result: RecommendationResult, config: Config) -> Optional[str]:
    """A "why this time slot" card for the recommended slot, shown under the
    delivery-window panel. It explains the SLOT choice only: how the window was
    placed on the route (its basis), the slot-level factors that pick it (slot
    match + availability), and the underlying detail — the exact openness
    calculation (which committed stops contend, by tier + harm) and the proximity
    stops the window was clustered around. Route-level factors (geography,
    capacity) are excluded. ``None`` when nothing was recommended."""
    rec = result.recommendation
    if not rec.recommended_route_id or not rec.factor_breakdown:
        return None
    has_pref = result.customer.preferred_slot is not None
    basis = _SLOT_BASIS_SENTENCE.get(rec.recommended_window_basis or "", "")
    basis_html = f'<div class="slot-why-basis">{basis}</div>' if basis else ""
    factors = _slot_rationale_factors(rec.factor_breakdown, has_pref, _SLOT_LEVEL_FACTORS)

    winner = next(
        (c for c in result.candidates_considered if c.route.route_id == rec.recommended_route_id),
        None,
    )
    detail = ""
    if winner is not None and winner.chosen_window is not None:
        detail += _openness_breakdown_html(winner.route, winner.chosen_window, config)
        if result.customer.location is not None:
            detail += _proximity_stops_html(result.customer.location, winner.route, config)

    # Collapsible (like "Routes the agent evaluated"): a triangle on the summary
    # toggles it, collapsed by default so the panel stays compact until asked.
    return (
        '<details class="slot-why">'
        '<summary class="slot-why-head"><span class="slot-why-title">Why this time slot</span>'
        f"<span class=\"slot-why-slot\"><b>{_esc(rec.recommended_day)} · "
        f'{_win(rec.recommended_window)}</b> '
        f'<small style="color:var(--muted)">on {_esc(rec.recommended_route_id)} · '
        f'{_esc(rec.recommended_route_name)}</small></span></summary>'
        '<div class="slot-why-body">'
        f"{basis_html}"
        '<div class="slot-why-sub">The slot-level factors that pick this window from the '
        "route's options (its geography &amp; capacity are covered in the evaluated routes "
        "above):</div>"
        f'<div class="factors">{factors}</div>'
        f"{detail}"
        "</div></details>"
    )


def _touchpoint_card(
    type_cls: str,
    type_label: str,
    title: str,
    where: str,
    desc: str,
    guard: str,
    flag: str,
    state_cls: str,
    state_label: str,
) -> str:
    """One 'LLM & agent touchpoint' card: what kind of model call it is (agent /
    sub-agent / LLM call / narration), what it does, its deterministic guardrail,
    and whether it's on by default. ``desc``/``guard`` are trusted literal HTML
    authored here; only the small labels are escaped."""
    flag_html = f'<span class="flag">{_esc(flag)}</span>' if flag else ""
    return (
        '<div class="card tpcard">'
        f'<span class="tptype {type_cls}">{_esc(type_label)}</span>'
        f"<h4>{_esc(title)}</h4><div class=\"where\">{_esc(where)}</div>"
        f"<p>{desc}</p>"
        f'<div class="guard"><b>Guardrail:</b> {guard}</div>'
        f'<div class="tpmeta">{flag_html}<span class="state {state_cls}">{_esc(state_label)}</span></div>'
        "</div>"
    )


def _llm_touchpoints_section(config: Config) -> str:
    """Enumerate every place a model is in the loop — the conversational agent,
    the escalation-triage sub-agent (an ADK AgentTool), the grounded address
    resolver, and the grounded decision/selection calls — so the reader can see
    exactly where an LLM acts, what it's allowed to do, and how the deterministic
    pipeline stays the floor.

    Sourced from the code (roles + flags in ``shared/config.py``), so it can't
    drift: thresholds, the sample count, and each flag's default come from the
    live ``Config``."""
    k = config.judgment_sample_count
    rs_bar = f"{config.route_slot_score_threshold:.0%}"
    ro_bar = f"{config.total_score_threshold:.0%}"
    triage_on = config.use_escalation_triage
    addr_on = config.use_address_resolution
    rse_on = config.use_grounded_route_slot_escalation

    cards = [
        _touchpoint_card(
            "agent",
            "Agent",
            "Conversational orchestrator",
            "agent.py · root_agent · ADK LlmAgent · role root_agent",
            "The one agent that drives the whole run. It talks to the user across turns and "
            "<b>decides when to call which tool</b> — Intake → Geo-Lookup → Constraint Check → "
            "Score &amp; Rank → Recommend/Decide — then narrates the grounded result in its own words.",
            "It never computes a distance, constraint, or score itself, and it never overrides the "
            "decision — every number and verdict comes back from the tools "
            "(<span class=\"where\">tools/slot_recommendation.py</span>). The conversation is LLM-driven; "
            "the numbers are not.",
            "SMART_ASSIGNMENT_MODEL_ROOT_AGENT",
            "always",
            "Always on (conversational path)",
        ),
        _touchpoint_card(
            "call",
            "LLM call · FunctionTool",
            "Grounded address resolution",
            "address_resolve/ · resolve_address FunctionTool on root_agent · role address_resolve",
            "When an address won't geocode, instead of a dead-end the agent calls this tool: an LLM "
            "<b>picks the closest of the geocoder's own real candidate matches — by index</b> — as a "
            "<em>confirmable suggestion</em> for the user. It corrects a typo or ambiguity; it never "
            "writes a new address.",
            "The valid options are fixed upstream (the geocoder's candidates, "
            "<span class=\"where\">Geocoder.suggest</span>); the LLM picks one by index and must cite the "
            "candidates' own facts, verified in code (<span class=\"where\">address_resolve/verifier.py</span>). "
            "Any failure falls back to the deterministic highest-similarity candidate, and a human confirms "
            "the pick before anything acts on it.",
            "SMART_ASSIGNMENT_USE_ADDRESS_RESOLUTION",
            "on" if addr_on else "off",
            "On by default" if addr_on else "Off by default",
        ),
        _touchpoint_card(
            "sub",
            "Sub-agent · AgentTool",
            "Escalation-triage sub-agent",
            "triage/ · an LlmAgent via google.adk.tools.AgentTool · role triage",
            "The first real multi-agent split. On an escalation, root_agent <b>consults a dedicated "
            "LlmAgent</b> (exposed as an ADK <em>AgentTool</em>, consult-and-return) to compose the "
            "specialist handoff brief: root cause, concrete remediation options, and the question to ask.",
            "Read-only — it only reads session state and runs strictly downstream of the deterministic "
            "decision, so it <b>never changes the route, score, or decision</b>. A deterministic prose scan "
            "(<span class=\"where\">triage/verifier.py</span>) appends a “⚠ Unverified” caveat to any figure "
            "not found in the evaluation trace. root_agent keeps the human-in-the-loop pause.",
            "SMART_ASSIGNMENT_USE_ESCALATION_TRIAGE",
            "on" if triage_on else "off",
            "On by default" if triage_on else "Off by default",
        ),
        _touchpoint_card(
            "call",
            "LLM call",
            "Grounded recommend-vs-escalate judgment",
            "judgment/ · role judgment",
            "When enabled, an LLM makes the <b>recommend-or-escalate call itself</b> — reasoning over a "
            f"structured evidence packet of the raw per-candidate facts — instead of the fixed weighted-sum "
            f"+ {ro_bar} threshold gate.",
            "Hard constraints still run first and are the only thing that can drop a candidate, so the LLM "
            "<b>chooses only among already-feasible routes</b>. Structured citations are verified in code "
            "(<span class=\"where\">judgment/verifier.py</span>), with one corrective retry; escalation-side "
            f"cases resample up to k={k} and require consensus. Any failure falls back to the deterministic "
            "weighted pick — never worse than today.",
            "SMART_ASSIGNMENT_USE_GROUNDED_JUDGMENT",
            "off",
            "Off in code · on in .env.example",
        ),
        _touchpoint_card(
            "call",
            "LLM call",
            "Grounded route-slot decision",
            "routeslot/ · role judgment",
            "When route-slot scoring is on, the decision unit is the <b>(route, slot) pair</b>, and an LLM "
            "<b>makes the recommend-vs-escalate call itself</b> over all feasible route-slots — not a fixed "
            "threshold. A confident recommend ships on one verified call; an escalate (or low-confidence "
            f"recommend) is resampled up to k={k} and combined by consensus before it may auto-assign. It "
            "picks the winning route-slot by index and returns a structured trade-off explanation (summary, "
            "reasons, key trade-off, runner-up, agree/diverge vs. the weighted default).",
            "The LLM chooses only among deterministically feasible, scored route-slots; its pick and every "
            "cited figure are verified in code (<span class=\"where\">routeslot/verifier.py</span>), with one "
            f"corrective retry. On any failure it falls back to the deterministic {rs_bar}-threshold decision "
            "— the reproducible floor. Whether the LLM decides escalation is itself gated by "
            "<span class=\"where\">USE_GROUNDED_ROUTE_SLOT_ESCALATION</span> "
            f"({'on' if rse_on else 'off'} by default). Absorbs the slot-pick pass.",
            "SMART_ASSIGNMENT_USE_ROUTE_SLOT_SCORING",
            "off",
            "Off in code · on in .env.example",
        ),
        _touchpoint_card(
            "call",
            "LLM call",
            "Grounded delivery-slot selection",
            "slotpick/ · role slotpick",
            "After a route is chosen, an LLM <b>picks the final delivery window</b> from that route's "
            "deterministically enumerated candidate menu — by index only, reasoning over each candidate's "
            "facts. It reasons and selects; it never generates a window. (The route-slot decision above "
            "folds this pick into its own grounded call when it runs.)",
            "Constrained to the enumerated candidates and verified against the packet "
            "(<span class=\"where\">slotpick/verifier.py</span>); it <b>only re-orders that route's slots</b>, "
            "never the route, score, or decision. The hand-tuned deterministic blend is demoted to reference "
            "+ fallback — the auditable floor.",
            "SMART_ASSIGNMENT_USE_GROUNDED_SLOT_SELECTION",
            "off",
            "Off in code · on in .env.example",
        ),
        _touchpoint_card(
            "narr",
            "LLM narration",
            "Reasoning narration",
            "reasoning.py · LLMReasoner · role reasoning",
            "An optional reasoner that rewrites the deterministic reasoning trace into more fluent prose for "
            "callers of the pipeline directly (e.g. <span class=\"where\">scripts/run_local.py</span>). The "
            "conversational agent instead narrates the recommendation in its own words.",
            "Narration only — it <b>never changes a number, a route, or the decision</b>. The deterministic "
            "trace (<span class=\"where\">DeterministicReasoner</span>) is the fallback and is exactly what "
            "generated this page — so these examples are reproducible offline.",
            "SMART_ASSIGNMENT_MODEL_REASONING",
            "off",
            "Deterministic by default",
        ),
    ]

    return f"""
    <span class="eyebrow">LLM &amp; agent touchpoints</span>
    <h2>Every place a model is in the loop — and its leash</h2>
    <p class="sub">The deterministic pipeline — geocoding, the hard-constraint filter, and the weighted
      scoring math — always runs and is the auditable floor. On top of it, several layers put an
      <b>LLM in the decision loop</b>, each gated by a <span style="font-family:var(--mono)">Config.use_*</span>
      flag. Three are <b>on by default</b> (grounded address resolution, the recommend-vs-escalate route-slot
      decision, and the escalation-triage sub-agent); the rest are turned on by the shipped
      <span style="font-family:var(--mono)">.env</span> — so with real backend credentials the
      recommend/escalate call, the route &amp; slot pick, and the address correction are <b>made by an LLM
      reasoning over the verified facts</b>, not by a fixed threshold. Here is every one, called out: what
      kind of model call it is, what it may touch, and the deterministic guardrail that keeps it honest. Each
      LLM surface resolves its model per-role via <span style="font-family:var(--mono)">Config.for_role</span>,
      so roles can run different model tiers while the backend stays global.</p>
    <div class="tp">{"".join(cards)}</div>
    <div class="guarantee">🛡️ <b>The invariant across all of them.</b> The deterministic pipeline always
      runs first and is the floor. An LLM here <b>reasons and selects; it never invents</b> an actionable
      value — it picks from a deterministically enumerated set (by index/id) and cites the facts it used,
      which are checked in code before anything acts on them. On any failure — parse, failed verification,
      missing credentials, backend error — the layer falls back to the deterministic result and logs why. So
      an LLM layer is strictly <b>additive</b>: never worse than the deterministic baseline, only — when it
      succeeds — better-reasoned and better-explained.</div>"""


def build_map_data(result: RecommendationResult, config: Optional[Config] = None) -> Optional[dict]:
    """Lat/lng data for a proximity map: the prospect's geocoded location plus,
    for every evaluated route, its service center, service radius, and existing
    committed stops -- everything needed to visually judge why a route was
    feasible/infeasible and how it scored on geographic clustering.

    Returns ``None`` if the customer was never geocoded (e.g. intake failed
    before geo-lookup ran).
    """
    config = config or DEFAULT_CONFIG
    loc = result.customer.location
    if loc is None:
        return None
    # Scored order the UI ranks by: recommended-first feasible routes, then the
    # infeasible ones -- identical to the "Routes the agent evaluated" section
    # (_routes_section), so the delivery-window panels line up with those cards.
    ranked_order = [c.route.route_id for c in result.ranked_feasible] + [
        c.route.route_id for c in result.candidates_considered if not c.feasible
    ]
    rank_by_id = {rid: i for i, rid in enumerate(ranked_order)}
    winner_id = result.recommendation.recommended_route_id
    routes = []
    for cand in result.candidates_considered:
        r = cand.route
        center = r.service_center
        # Candidate slots for the route-slot delivery panels: each scored slot,
        # highest score first, with the overall recommended one flagged. Falls
        # back to the single chosen window when route-slot scoring is off. Only
        # feasible routes offer real slots -- an infeasible route has none.
        if not cand.feasible:
            slots = []
        elif cand.scored_slots:
            slots = [
                {
                    "open": fmt_time(ss.slot.window[0]),
                    "close": fmt_time(ss.slot.window[1]),
                    "score": round(ss.total_score, 2),
                    "recommended": (
                        r.route_id == winner_id and ss.slot.window == cand.chosen_window
                    ),
                }
                for ss in sorted(
                    cand.scored_slots, key=lambda s: s.total_score, reverse=True
                )
            ]
        elif cand.chosen_window:
            slots = [
                {
                    "open": fmt_time(cand.chosen_window[0]),
                    "close": fmt_time(cand.chosen_window[1]),
                    "score": round(cand.total_score, 2),
                    "recommended": r.route_id == winner_id,
                }
            ]
        else:
            slots = []
        routes.append(
            {
                "route_id": _esc(r.route_id),
                "name": _esc(r.name),
                "day": r.day.value,
                "feasible": cand.feasible,
                "rank": rank_by_id.get(r.route_id, len(rank_by_id)),
                "distance_miles": round(cand.distance_miles, 1),
                "total_score": round(cand.total_score, 2) if cand.feasible else None,
                "service_center": {"lat": center.latitude, "lng": center.longitude},
                "service_radius_miles": r.service_radius_miles,
                # The slot we'd recommend on THIS route (location-aware, normalized
                # to the standard length) + why -- the timeline marks it with two
                # dashed guide lines. None when the route offers no windows.
                "chosen_window": (
                    {
                        "open": fmt_time(cand.chosen_window[0]),
                        "close": fmt_time(cand.chosen_window[1]),
                    }
                    if cand.chosen_window
                    else None
                ),
                "window_basis": cand.window_basis or None,
                # Route-slot candidates (each with its own score) for the
                # per-(route, slot) delivery-window panels.
                "slots": slots,
                "stops": [
                    {
                        "lat": s.location.latitude,
                        "lng": s.location.longitude,
                        "id": _esc(s.customer_number),
                        # Customer tier ("4"/"5"/"Perks"/"Other"); the timeline
                        # colours each stop's window bar by it. None when unknown.
                        "tier": _esc(s.customer_tier) if s.customer_tier else None,
                        # Committed delivery window (TW1 open/close), for the
                        # delivery-window timeline; None when unknown.
                        "window": (
                            {
                                "open": fmt_time(s.delivery_time_window[0]),
                                "close": fmt_time(s.delivery_time_window[1]),
                            }
                            if s.delivery_time_window
                            else None
                        ),
                    }
                    for s in r.committed_stops
                ],
            }
        )
    return {
        "customer": {
            "name": _esc(result.customer.name),
            "lat": loc.latitude,
            "lng": loc.longitude,
        },
        "routes": routes,
        # "Why this slot" rationale for the recommended route-slot, rendered under
        # the delivery-window panel. None when nothing was recommended.
        "rationaleHtml": _slot_rationale_html(result, config),
    }


# ---------------------------------------------------------------------------
# Frontend tab: the Salesforce-embedded "Choose a delivery slot" view a sales
# consultant confirms. Rendered per prospect from the same RecommendationResult
# the rest of the page uses, so it can never drift from the pipeline output. The
# rep only ever CONFIRMS one of the agent's enumerated options -- no free-text
# entry -- which is the frontend face of the "select from a valid set" guarantee.
# ---------------------------------------------------------------------------

_FE_FACTOR_TILE = {
    FACTOR_GEO_CLUSTERING: "Geo clustering",
    FACTOR_CAPACITY_BUFFER: "Capacity after add",
    FACTOR_WINDOW_MATCH: "Slot match",
    FACTOR_SLOT_AVAILABILITY: "Slot openness",
}
_DAY_FULL = {
    "MON": "Monday", "TUE": "Tuesday", "WED": "Wednesday", "THU": "Thursday",
    "FRI": "Friday", "SAT": "Saturday", "SUN": "Sunday",
}


def _fe_day(day_value: str) -> str:
    return _DAY_FULL.get(day_value, day_value)


def _fe_options(result: RecommendationResult):
    """Flatten the feasible candidates into ranked (route, slot) options plus the
    infeasible routes. Works on the route-slot path (per-slot scores) and the
    route-only path (one option per route from its chosen window)."""
    rec = result.recommendation
    feasible = list(result.ranked_feasible)
    options = []
    if any(e.scored_slots for e in feasible):
        for e in feasible:
            for ss in e.scored_slots:
                options.append(
                    {"cand": e, "window": ss.slot.window, "score": ss.total_score,
                     "factors": ss.factor_scores}
                )
    else:
        for e in feasible:
            if e.chosen_window:
                options.append(
                    {"cand": e, "window": e.chosen_window, "score": e.total_score,
                     "factors": e.factor_scores}
                )
    options.sort(key=lambda o: o["score"], reverse=True)
    for o in options:
        o["recommended"] = (
            o["cand"].route.route_id == rec.recommended_route_id
            and fmt_window(o["window"]) == (rec.recommended_window or "")
        )
    infeasible = [e for e in result.candidates_considered if not e.feasible]
    return options, infeasible


def _fe_tile(label: str, big: str, sub: str, score: float, miss: bool = False) -> str:
    cls = "fe-tile miss" if miss else "fe-tile"
    return (
        f'<div class="{cls}"><div class="tl">{_esc(label)}</div>'
        f'<div class="tv">{_esc(big)}</div><div class="ts">{_esc(sub)}</div>'
        f'<span class="fs">{score:.2f}</span></div>'
    )


def _fe_tiles(factors, cand: CandidateEvaluation) -> str:
    """One metric tile per scored factor, in canonical order: the concrete
    operational value the rep cares about (miles, % full, minutes matched, stops
    overlapping) plus the 0-1 factor score. Concrete numbers come from the model
    (utilization/headroom) or the factor's own grounded detail string."""
    by = {f.name: f for f in factors}
    tiles = []
    for name in _RS_FACTOR_ORDER:
        f = by.get(name)
        if f is None:
            continue
        label = _FE_FACTOR_TILE.get(name, name)
        d = f.detail or ""
        if name == FACTOR_GEO_CLUSTERING:
            m = re.search(r"avg\s+([\d.]+)\s*mi", d)
            big = f"{m.group(1)} mi" if m else f"{f.value:.2f}"
            tiles.append(_fe_tile(label, big, "avg to existing stops", f.value))
        elif name == FACTOR_CAPACITY_BUFFER:
            tiles.append(
                _fe_tile(label, f"{cand.utilization_after:.0%}",
                         f"{cand.remaining_capacity_after} cases headroom", f.value)
            )
        elif name == FACTOR_WINDOW_MATCH:
            if f.value <= 0.001:
                tiles.append(_fe_tile(label, "misses pref.", "wrong hours", f.value, miss=True))
            else:
                m = re.search(r"covers\s+(\d+)\s+of\s+the\s+(\d+)", d)
                big = f"{m.group(1)}/{m.group(2)} min" if m else f"{f.value:.2f}"
                tiles.append(_fe_tile(label, big, "of preferred window", f.value))
        else:  # slot availability (openness)
            m = re.search(r"\((\d+)\s+overlap", d)
            if m:
                n = int(m.group(1))
                sub = f"{n} committed stop overlaps" if n != 1 else "1 committed stop overlaps"
            else:
                sub = "tier-weighted openness"
            tiles.append(_fe_tile(label, f"{f.value:.2f}", sub, f.value))
    cols = max(1, len(tiles))
    return f'<div class="fe-tiles" style="--fe-cols:{cols}">{"".join(tiles)}</div>'


def _fe_why(o: dict, bar: float, recommended: bool) -> str:
    """A short, grounded 'why' line for one option, composed from the factors'
    own detail strings (never free text)."""
    by = {f.name: f for f in o["factors"]}
    parts = []
    g = by.get(FACTOR_GEO_CLUSTERING)
    if g and g.detail:
        parts.append(g.detail.rstrip("."))
    w = by.get(FACTOR_WINDOW_MATCH)
    if w is not None and w.value > 0.001 and w.detail:
        parts.append(w.detail.rstrip("."))
    elif w is not None and w.value <= 0.001:
        parts.append("but it misses the preferred hours")
    clears = "clears" if o["score"] >= bar else "is below"
    lead = "Strongest route-slot overall — " if recommended else ""
    body = "; ".join(parts)
    tail = f'Route-slot score <b>{o["score"]:.2f}</b> {clears} the {bar:.0%} auto-assign bar.'
    return f'<p class="fe-why">{lead}{_esc(body)}. {tail}</p>'


def _fe_rank(score: float, bar: float) -> tuple:
    """Map a route-slot score to a rep-facing quality rank (chip class, label),
    banded relative to the auto-assign bar. Replaces the raw numeric score as the
    headline: >= bar+0.20 is High confidence, >= bar Medium feasible, else Low."""
    if score >= bar + 0.20:
        return "hi", "High confidence"
    if score >= bar:
        return "med", "Medium feasible"
    return "lo", "Low feasible"


def _fe_option_card(o: dict, result: RecommendationResult, config: Config, bar: float) -> str:
    rec = result.recommendation
    route = o["cand"].route
    win = _win(fmt_window(o["window"]))
    recommended = o["recommended"]
    rank_cls, rank_label = _fe_rank(o["score"], bar)
    if recommended and rec.decision == Decision.RECOMMENDED:
        rank_label += " · auto-assign"
    elif recommended:  # escalated low score -> the strongest, proposed for review
        rank_label += " · needs review"
    selected = " selected" if recommended else ""
    badge = f'<span class="fe-rank {rank_cls}"><span class="d"></span>{rank_label}</span>'
    when = f"{_esc(_fe_day(route.day.value))} · {win}"
    tradeoff = ""
    if recommended and rec.key_tradeoff:
        tradeoff = (
            f'<div class="fe-tradeoff"><b>Why this over the alternative:</b> '
            f'{_esc(rec.key_tradeoff)}</div>'
        )
    # Every feasible option is a real, selectable choice (the rep confirms one).
    selrow = '<div class="fe-selrow"><span class="fe-selmark"></span></div>'
    return (
        f'<article class="fe-opt selectable{selected}" data-when="{when}" role="button" tabindex="0" '
        f'aria-pressed="{"true" if recommended else "false"}">'
        f'<div class="fe-opt-head"><div class="fe-radio" aria-hidden="true"></div>'
        f'<div class="fe-opt-title"><div class="when"><b>{_esc(_fe_day(route.day.value))}</b> · '
        f'<span class="nowrap">{win}</span></div>'
        f'<div class="fe-route">Route <b>{_esc(route.route_id)}</b> · {_esc(route.name)}</div></div>'
        f'{badge}</div>{_fe_why(o, bar, recommended)}{_fe_tiles(o["factors"], o["cand"])}{tradeoff}{selrow}</article>'
    )


def _fe_unavailable_card(cand: CandidateEvaluation) -> str:
    route = cand.route
    failed = ", ".join(CONSTRAINT_LABEL.get(c.name, c.name) for c in cand.failed_constraints)
    detail = cand.failed_constraints[0].detail if cand.failed_constraints else ""
    return (
        f'<article class="fe-opt dim"><div class="fe-opt-head">'
        f'<div class="fe-radio no" aria-hidden="true"></div>'
        f'<div class="fe-opt-title"><div class="when"><b>{_esc(_fe_day(route.day.value))}</b> · '
        f'<span class="nowrap" style="color:var(--muted);font-weight:500">no slot built</span></div>'
        f'<div class="fe-route">Route <b>{_esc(route.route_id)}</b> · {_esc(route.name)}</div></div>'
        f'<span class="fe-rank no"><span class="d"></span>Unavailable</span></div>'
        f'<div class="fe-unavail"><span class="x">✗ {_esc(failed)}</span> — {_esc(detail)}</div></article>'
    )


def _project(points, w=268, h=180, pad=28):
    lats = [p[0] for p in points]
    lngs = [p[1] for p in points]
    minlat, maxlat = min(lats), max(lats)
    minlng, maxlng = min(lngs), max(lngs)
    dlat = (maxlat - minlat) or 1e-6
    dlng = (maxlng - minlng) or 1e-6

    def proj(lat, lng):
        x = pad + (lng - minlng) / dlng * (w - 2 * pad)
        y = pad + (maxlat - lat) / dlat * (h - 2 * pad)  # invert: north is up
        return round(x, 1), round(y, 1)

    return proj


def _convex_hull(pts: list) -> list:
    """Convex hull (Andrew's monotone chain) of (lat, lng) points, returned as an
    ordered ring. <= 2 unique points are returned as-is (no polygon)."""
    pts = sorted(set(pts))
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _fe_focus(result: RecommendationResult):
    """The route to centre the map on: the recommended/proposed route, else the
    top feasible one, else None (no serviceable route)."""
    rec = result.recommendation
    focus = None
    if rec.recommended_route_id:
        focus = next(
            (c for c in result.candidates_considered if c.route.route_id == rec.recommended_route_id),
            None,
        )
    if focus is None and result.ranked_feasible:
        focus = result.ranked_feasible[0]
    return focus


# A clean, light map canvas -- a faint grid, no busy blocks -- so the cluster
# polygon and markers are the focus.
_FE_MAP_BACKDROP = (
    '<rect x="0" y="0" width="300" height="210" fill="#f7fafc"/>'
    '<g stroke="#eaeef4" stroke-width="1">'
    '<line x1="0" y1="52" x2="300" y2="52"/><line x1="0" y1="105" x2="300" y2="105"/>'
    '<line x1="0" y1="158" x2="300" y2="158"/><line x1="75" y1="0" x2="75" y2="210"/>'
    '<line x1="150" y1="0" x2="150" y2="210"/><line x1="225" y1="0" x2="225" y2="210"/></g>'
)


def _fe_cluster_svg(result: RecommendationResult) -> str:
    """A clean, grounded cluster view: the focus route's committed stops + the new
    prospect (projected from the real geocoded coordinates) inside a convex-hull
    polygon, plus a mock Depot (the OpCo the truck leaves from) with a dashed
    delivery line into the cluster."""
    focus = _fe_focus(result)
    cust = result.customer.location
    if focus is None or cust is None:
        nearest = min(result.candidates_considered, key=lambda c: c.distance_miles, default=None)
        if nearest is None:
            return '<div class="nomap">No candidate routes to display.</div>'
        return (
            f'<div class="nomap">No serviceable route near this address — the nearest, '
            f'<b>{_esc(nearest.route.route_id)}</b>, is {nearest.distance_miles:.0f} mi from its service '
            f'center (outside the serviceable radius). Routed to a specialist.</div>'
        )
    route = focus.route
    cpt = (cust.latitude, cust.longitude)
    stops = [(s.location.latitude, s.location.longitude) for s in route.committed_stops if s.location]
    # Mock Depot: place it just outside the cluster (lower-left) so a clean dashed
    # "truck leaves the OpCo" line runs into the delivery cluster. Illustrative,
    # not a real coordinate (the whole demo runs on mock data).
    lats = [la for la, _ in stops] + [cpt[0]]
    lngs = [ln for _, ln in stops] + [cpt[1]]
    span_lat = (max(lats) - min(lats)) or 0.01
    span_lng = (max(lngs) - min(lngs)) or 0.01
    depot = (min(lats) - 0.42 * span_lat, min(lngs) - 0.42 * span_lng)
    proj = _project([cpt, depot] + stops, w=300, h=210, pad=32)
    hull = _convex_hull(stops + [cpt])
    poly = " ".join("%s,%s" % proj(la, ln) for (la, ln) in hull)
    cx, cy = proj(*cpt)
    dx, dy = proj(*depot)
    dots = "".join(
        '<circle cx="%s" cy="%s" r="5" fill="#1257a6" stroke="#fff" stroke-width="1.5"/>' % proj(la, ln)
        for (la, ln) in stops
    )
    shape = (
        f'<polygon points="{poly}" fill="rgba(26,127,55,.13)" stroke="#1a7f37" '
        'stroke-width="1.8" stroke-linejoin="round"/>'
        if len(hull) >= 3
        else f'<polyline points="{poly}" fill="none" stroke="#1a7f37" stroke-width="1.8"/>'
    )
    depot_marker = (
        f'<path d="M{dx - 7},{dy - 1} L{dx},{dy - 8} L{dx + 7},{dy - 1} Z" fill="#0b2e59"/>'
        f'<rect x="{dx - 6}" y="{dy - 1}" width="12" height="9" rx="1" fill="#0b2e59"/>'
        f'<text x="{dx}" y="{dy + 20}" text-anchor="middle" font-size="8.5" font-weight="700" '
        f'fill="#0b2e59">Depot (OpCo)</text>'
    )
    return (
        f'<svg viewBox="0 0 300 210" role="img" '
        f'aria-label="Route {_esc(route.route_id)} delivery cluster with depot">'
        f'{_FE_MAP_BACKDROP}{shape}'
        f'<path d="M{dx},{dy} L{cx},{cy}" stroke="#8a93a6" stroke-width="1.3" stroke-dasharray="4 3"/>'
        f'{dots}{depot_marker}'
        f'<circle cx="{cx}" cy="{cy}" r="7" fill="#5b3fb0" stroke="#fff" stroke-width="2"/>'
        f'<text x="{cx}" y="{cy - 12}" text-anchor="middle" font-size="8.5" font-weight="700" '
        f'fill="#5b3fb0">new stop</text></svg>'
    )


def _fe_geo_miles(rec) -> Optional[str]:
    """The 'avg X mi to existing stops' figure from the winning slot's geo factor,
    for the map subtitle; None if unavailable."""
    for f in rec.factor_breakdown or []:
        if f.name == FACTOR_GEO_CLUSTERING and f.detail:
            m = re.search(r"avg\s+([\d.]+)\s*mi", f.detail)
            if m:
                return m.group(1)
    return None


def _frontend_panel_html(result: RecommendationResult, config: Config) -> str:
    """The full SC-facing 'Choose a delivery slot' view for one prospect."""
    c = result.customer
    rec = result.recommendation
    bar = (
        config.route_slot_score_threshold
        if config.use_route_slot_scoring
        else config.total_score_threshold
    )
    options, infeasible = _fe_options(result)

    # --- left: prospect ---
    sysco = _esc(c.customer_number) if c.customer_number else "— new prospect (none yet)"
    if c.preferred_slot is not None:
        pref = f"{_fe_day(c.preferred_slot.day.value)} · {_win(fmt_window(c.preferred_slot.window))}"
    else:
        pref = "No stated preference"
    side = (
        '<aside class="fe-side"><div class="kicker">New prospect · onboarding</div>'
        f'<h3>{_esc(c.name)}</h3>'
        f'<div class="fe-field"><div class="l">Sysco account #</div><div class="v">{sysco}</div></div>'
        '<div class="fe-field"><div class="l">Delivery address <span style="font-weight:400">· primary '
        f'identifier</span></div><div class="v addr"><span class="pin">📍</span>{_esc(c.address)}</div></div>'
        f'<div class="fe-field"><div class="l">Order quantity</div><div class="v">{c.order_quantity_cases} cases</div></div>'
        '<div class="fe-field"><div class="l">Preferred slot <span style="font-weight:400">· soft '
        f'preference</span></div><div class="v">{_esc(pref)}</div></div>'
        '<div class="fe-note"><div class="h">🛡️ How these options are built</div>'
        'Hard rules (serviceability, capacity) and the factor scoring run in deterministic code. The LLM '
        'ranks only the feasible options and cites the facts; on any failure it falls back to the '
        'deterministic pick. Escalation routes to a routing specialist — not a hard block.</div></aside>'
    )

    # --- center: banner + option cards + escalation + confirm ---
    banner = ""
    if rec.decision == Decision.ESCALATED_LOW_SCORE:
        banner = (
            f'<div class="fe-banner warn">⚠ The agent escalated this — no option cleared the {bar:.0%} '
            'auto-assign bar. The strongest is proposed below for a specialist to confirm.</div>'
        )
    elif rec.decision == Decision.ESCALATED_NO_FEASIBLE_SLOT:
        banner = (
            '<div class="fe-banner stop">✖ No serviceable route — every candidate failed a hard rule, so '
            'there is nothing to auto-assign. Routed to a routing specialist.</div>'
        )

    cards = "".join(_fe_option_card(o, result, config, bar) for o in options)
    cards += "".join(_fe_unavailable_card(e) for e in infeasible)

    escalate = (
        '<div class="fe-escalate"><h4>None of these work for the customer?</h4>'
        '<p>Request a specialist review. It routes to a routing specialist with the full evaluation attached '
        'as a handoff brief (composed by the escalation-triage agent) — a queue, not a hard block.</p>'
        '<button type="button">Request specialist review →</button></div>'
    )

    # The slot the rep currently has selected (the recommended one to start); the
    # JS updates <b id="fe-sel"> as the rep clicks other options.
    sel_when = ""
    chosen = next((o for o in options if o["recommended"]), options[0] if options else None)
    if chosen is not None:
        sel_when = f"{_fe_day(chosen['cand'].route.day.value)} · {_win(fmt_window(chosen['window']))}"

    if rec.decision == Decision.RECOMMENDED:
        confirm = (
            '<div class="fe-confirm"><div class="log">Selected: <b id="fe-sel">' + sel_when + '</b> · logged '
            'with the rep and the facts the agent cited — fully auditable.</div><div class="btns">'
            '<button type="button" class="fe-btn-ghost">Cancel</button>'
            '<button type="button" class="fe-btn-primary">Confirm slot</button></div></div>'
        )
    elif rec.decision == Decision.ESCALATED_LOW_SCORE:
        confirm = (
            '<div class="fe-confirm"><div class="log">Selected: <b id="fe-sel">' + sel_when + '</b> — below '
            'the auto-assign bar; confirming logs the rep’s override, or send it to a specialist.</div>'
            '<div class="btns"><button type="button" class="fe-btn-ghost">Confirm proposed slot</button>'
            '<button type="button" class="fe-btn-primary">Send to specialist</button></div></div>'
        )
    else:  # no feasible slot
        confirm = (
            '<div class="fe-confirm"><div class="log">No assignable slot — this must go to a routing '
            'specialist.</div><div class="btns">'
            '<button type="button" class="fe-btn-primary" disabled>Confirm slot</button></div></div>'
        )

    main = f'<main class="fe-main">{banner}{cards}{escalate}{confirm}</main>'

    # --- right: clean cluster map (SVG: polygon + stops + new prospect + depot) ---
    focus = _fe_focus(result)
    svg = _fe_cluster_svg(result)
    if focus is not None:
        route = focus.route
        miles = _fe_geo_miles(rec)
        near = f"new stop {miles} mi from the cluster" if miles else "new stop near the existing cluster"
        map_card = (
            f'<aside class="fe-map"><div class="mt">{_esc(route.route_id)} · cluster view</div>'
            f'<div class="ms">{_esc(route.name)} · {near}</div>{svg}'
            '<div class="fe-legend">'
            '<span><i class="fe-dot" style="background:#1257a6"></i>Existing stops</span>'
            '<span><i class="fe-dot" style="background:#5b3fb0"></i>New prospect</span>'
            '<span><i class="fe-dot" style="background:#0b2e59;border-radius:2px"></i>Depot (OpCo)</span>'
            '<span><i class="fe-dot" style="background:rgba(26,127,55,.35);'
            'border:1px solid #1a7f37;border-radius:2px"></i>Route cluster</span>'
            '</div><div class="fe-gain">▲ Adds density without extending route time</div></aside>'
        )
    else:
        map_card = f'<aside class="fe-map"><div class="mt">Proximity</div>{svg}</aside>'

    return f'<div class="fe-grid">{side}{main}{map_card}</div>'


def build_workflow_payload(
    result: RecommendationResult,
    config: Config,
    reasoning_override: Optional[str] = None,
) -> dict:
    """The visualization payload for one workflow run: the animated step cards,
    the final result card, and the proximity-map data.

    This is the single source of truth for the Simulator's data, shared by the
    static page generator (``build_page``) and the live web app
    (``smart_assignment.webapp``) so the interactive UI can never drift from the
    published examples — both render the exact same structure.

    ``reasoning_override`` lets the live LLM chat pass the agent's own
    recommendation narration, so the result card's "Why the agent chose this"
    shows the same text the chat box did instead of a separately-rendered version.
    """
    return {
        "name": _esc(result.customer.name),
        "address": _esc(result.customer.address),
        "steps": _sim_steps(result, config),
        # The recommended-route card, without its embedded routes list -- the full
        # evaluated-routes breakdown is `routesHtml`, rendered separately (below
        # the map in the web app) so it can be a rich, default-open section.
        "resultHtml": _example_card(result, include_routes=False, reasoning_override=reasoning_override),
        "routesHtml": _route_cards(result, config),
        # The SC-facing "Choose a delivery slot" view for the Frontend tab.
        "frontendHtml": _frontend_panel_html(result, config),
        "map": build_map_data(result, config),
        # UI banners (e.g. grounded reasoning fell back to deterministic). Empty
        # for the normal deterministic/weighted path, so the published static
        # page never shows one.
        "notices": _payload_notices(result),
    }


def _payload_notices(result: RecommendationResult) -> list[dict]:
    """Structured UI notices derived from the recommendation (rendered as
    banners by the web app). Currently just the grounded-judgment fallback."""
    rec = result.recommendation
    notices: list[dict] = []
    if getattr(rec, "grounded_fallback", False):
        notices.append(
            {
                "kind": "warning",
                "text": _esc(
                    rec.grounded_fallback_reason
                    or "Grounded reasoning was unavailable; showing the deterministic result."
                ),
            }
        )
    return notices


def build_page(results: list[RecommendationResult], config: Config) -> str:
    """Render the full three-tab overview HTML from live workflow results."""
    threshold = f"{config.total_score_threshold:.0%}"
    fe_bar = f"{(config.route_slot_score_threshold if config.use_route_slot_scoring else config.total_score_threshold):.0%}"
    top_n = config.top_n_candidate_routes
    cards = "".join(_example_card(r) for r in results)
    payload = {r.customer.lookup_key: build_workflow_payload(r, config) for r in results}
    data_block = (
        '<script type="application/json" id="workflow-data">'
        + json.dumps(payload, ensure_ascii=False)
        + "</script>"
    )
    js_block = "<script>" + _SIM_JS + _TABS_JS + _FRONTEND_JS + "</script>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Smart Assignment — AI Agent for Delivery Slot Recommendation</title>
<meta name="description" content="An AI agent that autonomously assigns delivery slots for new Sysco customers. Explore the agentic workflow, architecture, and run it live on mock data." />
<!-- GENERATED by scripts/generate_page.py from live workflow output. Do not edit by hand. -->
<style>{_STYLE}{_FE_STYLE}</style>
</head>
<body>

<header class="hero">
  <div class="wrap">
    <span class="chip">🤖 Agentic workflow · Sysco</span>
    <h1>Smart Assignment</h1>
    <p class="lead">An <strong>AI agent</strong> that autonomously recommends the best delivery day &amp;
      time slot for a <strong>new customer</strong> — running every step end-to-end, enforcing hard
      operational rules in code and scoring options with deterministic math, then having an
      <strong>LLM reason over those verified facts</strong> to make the recommend-or-escalate call and pick
      the slot — handing off to a human (with a triage sub-agent drafting the brief) whenever the best
      option falls short.</p>
    <div class="meta">
      <span class="chip">🤖 Agent-orchestrated end to end</span>
      <span class="chip">🧠 LLM-grounded, verified decisions</span>
      <span class="chip">✅ Deterministic rules, scoring &amp; fallback</span>
      <span class="chip">🧪 Running on mock data</span>
    </div>
  </div>
</header>

<nav class="tabbar" role="tablist" aria-label="Sections">
  <div class="wrap">
    <button class="tabbtn active" data-tab="overview" role="tab" aria-selected="true">Overview</button>
    <button class="tabbtn" data-tab="architecture" role="tab" aria-selected="false">Architecture</button>
    <button class="tabbtn" data-tab="simulator" role="tab" aria-selected="false">Simulator</button>
    <button class="tabbtn" data-tab="frontend" role="tab" aria-selected="false">Frontend</button>
  </div>
</nav>

<main>

<div class="tabpanel active" id="tab-overview" role="tabpanel">
  <section>
    <div class="wrap">
      <span class="eyebrow">The problem</span>
      <h2>Onboarding a new account, every time, the same way</h2>
      <p class="sub">When a new customer signs on, someone has to decide which delivery route and time
        window they should join. That decision balances truck capacity, geography, and the customer's
        preference — and it needs to be consistent, explainable, and fast. The Smart Assignment agent
        makes that call automatically, and flags the tricky ones for a specialist instead of guessing.</p>
    </div>
  </section>

  <section style="background:#fff; border-top:1px solid var(--line); border-bottom:1px solid var(--line);">
    <div class="wrap">
      <span class="eyebrow">How the agent works</span>
      <h2>Five steps, executed autonomously by the agent</h2>
      <div class="agent-banner"><span class="big">🤖</span>
        <div><b>A single AI agent performs all five stages below in one end-to-end run</b> — these are
        phases of the same agent's workflow, not five separate agents. The hard rules and scoring are
        deterministic code, but the stage-5 recommend-or-escalate <em>decision</em> is an LLM reasoning over
        those verified facts. On an escalation it consults one <em>sub-agent</em> (the escalation-triage
        AgentTool) to write the handoff brief; otherwise a person is involved only if the agent decides to
        escalate. See the <b>Architecture</b> tab for every agent, sub-agent, and LLM call.</div></div>
      <div class="flow">
        <div class="step"><div class="num">1</div><h3>Intake</h3><p>Capture the customer's address, order quantity (cases), and preferred slot (day + time).</p><p class="action">Validate the intake profile &amp; build it.</p></div>
        <div class="step"><div class="num">2</div><h3>Geo-Lookup</h3><p>Geocode the address and find the nearest candidate routes.</p><p class="action">Geocode &amp; pick the Top-{top_n} nearest.</p></div>
        <div class="step"><div class="num">3</div><h3>Constraint Check</h3><p>Drop any route that fails a hard rule.</p><p class="action">Enforce serviceability &amp; capacity.</p></div>
        <div class="step"><div class="num">4</div><h3>Score &amp; Rank</h3><p>Rank survivors on weighted business factors.</p><p class="action">Score &amp; order the options.</p></div>
        <div class="step"><div class="num">5</div><h3>Recommend</h3><p>An LLM reasons over the verified facts to recommend the top slot — or escalate.</p><p class="action">Grounded recommend-or-escalate call.</p></div>
      </div>
    </div>
  </section>

  <section>
    <div class="wrap">
      <span class="eyebrow">The rules the agent enforces</span>
      <h2>Hard constraints — non-negotiable, checked in code</h2>
      <p class="sub">Just <b>two</b> objective facts gate feasibility — not judgment calls. The agent removes
        any route that fails one before ranking; it can never "reason" a customer onto a full truck or
        outside the serviceable area. (See the <b>Simulator</b> tab for exactly how the surviving routes
        are scored.)</p>
      <div class="grid-2">
        <div class="card"><div class="icon">\U0001f4cd</div><h3>Geographic serviceability</h3><p>The customer must fall within the route's serviceable radius.</p></div>
        <div class="card"><div class="icon">\U0001f4e6</div><h3>Route capacity</h3><p>The truck stays at or below {config.max_utilization_after_assignment:.0%} capacity after adding this order.</p></div>
      </div>
      <p class="sub" style="margin-top:18px">🕑 <b>The preferred delivery slot — a day of week plus a time
        window — is a soft preference, not a hard constraint</b> — it never eliminates a route. Instead it
        feeds the <em>slot match</em> scoring factor, so a route on a different day or time can still win if
        it's the best overall fit.</p>
    </div>
  </section>

  <section style="background:#fff; border-top:1px solid var(--line); border-bottom:1px solid var(--line);">
    <div class="wrap">
      <span class="eyebrow">The outcome</span>
      <h2>Three possible agent decisions</h2>
      <p class="sub">Every run ends in one of three states. When the agent judges the best option too weak
        to auto-assign (the {threshold} bar is its reference and deterministic fallback), or there's no valid
        slot at all, it hands the case to a human — with full context attached.</p>
      <div class="legend">
        <div class="card"><span class="pill rec">✔ Recommended</span><p style="margin-top:12px">A clear winner whose own total score clears the bar — the agent auto-assigns.</p></div>
        <div class="card"><span class="pill low">⚠ Low score</span><p style="margin-top:12px">A slot is proposed, but its own total score falls short — the agent asks a specialist to confirm.</p></div>
        <div class="card"><span class="pill no">✖ No feasible slot</span><p style="margin-top:12px">Every candidate failed a hard rule — the agent routes it to a specialist.</p></div>
      </div>
    </div>
  </section>
</div>

<div class="tabpanel" id="tab-architecture" role="tabpanel">
  <section>
    <div class="wrap">
      <span class="eyebrow">Architecture</span>
      <h2>The agentic workflow, end to end</h2>
      <p class="sub">The agent is a Google ADK <em>LlmAgent</em> that talks to the user and calls a tool for
        each of the five steps — it decides <em>when</em> to call which tool and narrates the result. Every
        distance, constraint check, and score comes back from deterministic code, but the recommend/escalate
        <em>decision</em> and the route &amp; slot <em>pick</em> are made by an LLM reasoning over those
        verified facts — constrained to the feasible options, cited, checked in code, with a deterministic
        fallback. At <b>Geo-Lookup</b>, when an address won't geocode, a grounded resolver LLM picks the
        closest of the geocoder's <em>own</em> suggested matches for the user to confirm — it never invents an
        address. On an escalation it consults a dedicated LlmAgent — the escalation-triage <em>AgentTool</em>
        — to write the handoff brief. The <b>LLM &amp; agent touchpoints</b> below list every model in the loop.</p>
      <div class="arch">
        {_ARCH_SVG}
        <div class="arch-legend">
          <div class="card"><div class="icon">🤖</div><h4>Agent orchestrator</h4><p>An ADK LlmAgent drives all five steps via tool calls, conversationally.</p></div>
          <div class="card"><div class="icon">🧭</div><h4>Deterministic tools</h4><p>Geo, hard constraints, and weighted scoring — plain code the agent calls.</p></div>
          <div class="card"><div class="icon">🤝</div><h4>A triage sub-agent</h4><p>On escalation, a dedicated LlmAgent (an ADK AgentTool) composes the specialist brief — read-only, it never changes the decision.</p></div>
          <div class="card"><div class="icon">🎯</div><h4>Selects, never invents</h4><p>The recommend/escalate call, the route &amp; slot pick, and the on-miss address correction are an LLM's — but only from a deterministically enumerated set (feasible routes; the geocoder's own suggestions), with every cited fact verified in code and a deterministic fallback.</p></div>
        </div>
      </div>
    </div>
  </section>

  <section style="background:#fff; border-top:1px solid var(--line);">
    <div class="wrap">
      {_llm_touchpoints_section(config)}
    </div>
  </section>

  <section style="border-top:1px solid var(--line);">
    <div class="wrap">
      <span class="eyebrow">Key mechanics</span>
      <h2>What makes it trustworthy</h2>
      <div class="grid-2">
        <div class="card"><h4>🤖 One agent, five steps</h4><p>A single ADK LlmAgent calls one tool per step — Intake → Geo-Lookup → Constraint Check → Score &amp; Rank → Decide — and pauses to escalate when needed.</p></div>
        <div class="card"><h4>🧭 Hard rules &amp; scoring are code, not vibes</h4><p>The feasibility filter (serviceability, capacity) and the factor scoring are deterministic Python — identical inputs always yield identical facts, and that deterministic result is always the floor and the fallback.</p></div>
        <div class="card"><h4>🎯 The decision is an LLM's — but grounded &amp; verified</h4><p>With the backend configured, an LLM makes the recommend/escalate call and picks the route &amp; slot. It only ever chooses from a deterministically enumerated, feasible set and cites facts that are checked in code — it never free-generates a route, window, or score, and falls back to the deterministic result on any failure.</p></div>
        <div class="card"><h4>🙋 Human-in-the-loop on escalation</h4><p>The agent pauses and waits for a specialist's reply (via ADK's request_input tool) when it finds no feasible slot, or the best option's own total score falls short of the auto-assign bar — after the triage sub-agent has drafted the brief.</p></div>
      </div>
    </div>
  </section>
</div>

<div class="tabpanel" id="tab-simulator" role="tabpanel">
  <section id="try">
    <div class="wrap">
      <span class="eyebrow">Try it yourself</span>
      <h2>Run the agent on a customer</h2>
      <p class="sub">Enter a mock customer's address (or pick one below) and watch the agent execute each step —
        including the scoring math for every feasible route — then render its recommendation. Everything
        runs in your browser; no data leaves the page.</p>
      <div class="sim">
        <div class="sim-controls">
          <input id="cust-input" placeholder="e.g. 5085 Westheimer Rd, Houston, TX 77056" aria-label="Customer address" autocomplete="off" />
          <button id="run-btn" class="run">▶ Run agent workflow</button>
        </div>
        <div class="sim-hint" id="sim-hint"></div>
        <div class="sim-error" id="sim-error"></div>
        <div class="picker">
          <span class="picker-label">Sample prospects — click one to load their address into the box above, then press <b>Run agent workflow</b>:</span>
          <div class="chips" id="chips"></div>
        </div>
        <div id="sim-steps"></div>
        <div class="sim-output" id="sim-output"></div>
      </div>
    </div>
  </section>

  <section style="border-top:1px solid var(--line);">
    <div class="wrap">
      {_scoring_section(config)}
    </div>
  </section>

  <section style="background:#fff; border-top:1px solid var(--line);">
    <div class="wrap">
      {_config_sources(config, results)}
    </div>
  </section>

  <section style="border-top:1px solid var(--line);">
    <div class="wrap">
      <span class="eyebrow">All outcomes at a glance</span>
      <h2>What the agent decided for each mock customer</h2>
      <p class="sub">These cards are generated straight from the agent's output. Each customer lands on a
        different outcome; expand <em>“Routes the agent evaluated”</em> to audit the full decision.</p>
      <div class="examples">{cards}
      </div>
    </div>
  </section>

  <section style="padding-top:0;">
    <div class="wrap">
      <div class="note">
        <strong>About this data.</strong> The agent runs on <strong>mock</strong> Houston-area routes and
        geocoding so the workflow can be demonstrated end-to-end. Capacities, service radii, scoring
        weights, and thresholds are illustrative starting points — not validated Sysco policy. The route
        source and geocoder are designed to be swapped for real systems without changing the agent's logic.
      </div>
    </div>
  </section>
</div>

<div class="tabpanel" id="tab-frontend" role="tabpanel">
  <section>
    <div class="fe-wrap">
      <div class="fe-eyebrow">Salesforce Lightning component · preview</div>
      <h2>Choose a delivery slot</h2>
      <p class="sub">The Salesforce-embedded view a sales consultant sees. Agent-ranked delivery slots for a new
        prospect, grounded in route capacity, geographic clustering, slot match, and openness. The rep
        <b>selects one of the agent's options</b> — no free-text entry, so no slot is ever invented.</p>
      <div class="fe-status">🛰️ <span><b>Grounded</b> on Houston route data · hard rules &amp; scoring are
        <b>deterministic</b> · options ranked by the <b>LLM</b> and verified in code · auto-assign bar
        <b>{fe_bar}</b></span></div>
      <div class="fe-picker">
        <span class="picker-label">Sample prospects — click one to load their delivery-slot view:</span>
        <div class="chips" id="fe-chips"></div>
      </div>
      <div id="fe-view"></div>
    </div>
  </section>
</div>

</main>

<footer>
  <div class="wrap">
    Smart Assignment · an AI agent for delivery slot recommendation ·
    built on Google's Agent Development Kit (ADK).<br />
    Source &amp; docs on <a href="https://github.com/MuhammadVT/smart-assignment">GitHub</a>.
    Hard rules and scoring are deterministic and auditable; the recommend/escalate call, the route &amp; slot
    pick, and the address correction are an LLM's — grounded in verified facts, constrained to feasible
    options, with a deterministic fallback. See the Architecture tab for every agent, sub-agent, and LLM call.
  </div>
</footer>

{data_block}
{js_block}
</body>
</html>
"""


def generate(output_path: Optional[Path] = None, config: Optional[Config] = None) -> Path:
    """Run the workflow over the sample customers and write ``docs/index.html``."""
    config = config or DEFAULT_CONFIG
    reasoner = DeterministicReasoner()  # reproducible, no API key/network
    results = [
        run_slot_recommendation(customer, config=config, reasoner=reasoner)
        for customer in SAMPLE_CUSTOMERS
    ]
    out = output_path or DEFAULT_OUTPUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_page(results, config), encoding="utf-8")
    return out
