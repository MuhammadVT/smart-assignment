"""
Generate the GitHub Pages overview site (``docs/index.html``) directly from
live workflow output, so the content can never drift from the code.

The page is a three-tab single-page site:
  1. Overview     — what the agent does, the steps, the rules, the outcomes.
  2. Architecture — the agentic workflow diagram + how the pieces fit.
  3. Simulator    — how scoring is computed (with the real formulas), plus an
                    interactive runner and the per-customer results.

All example/interactive data is precomputed here from the real pipeline
(``mock_customers.SAMPLE_CUSTOMERS``) and embedded as JSON, so the static site
needs no backend. Reasoning uses the DeterministicReasoner so the page is
reproducible offline and regenerating with no code change produces no diff.

CLI: ``python3 scripts/generate_page.py``.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Optional

from smart_assignment.mock_customers import SAMPLE_CUSTOMERS
from smart_assignment.pipeline import run_slot_recommendation
from smart_assignment.reasoning import DeterministicReasoner
from smart_assignment.shared.config import (
    DEFAULT_CONFIG,
    FACTOR_CAPACITY_BUFFER,
    FACTOR_GEO_CLUSTERING,
    FACTOR_WINDOW_MATCH,
    Config,
)
from smart_assignment.shared.constraints import CONSTRAINT_LABEL, build_context
from smart_assignment.shared.models import (
    CandidateEvaluation,
    Decision,
    RecommendationResult,
)
from smart_assignment.shared.timeutils import duration_minutes, fmt_window

DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "docs" / "index.html"

FACTOR_LABEL = {
    FACTOR_GEO_CLUSTERING: "Geographic clustering",
    FACTOR_CAPACITY_BUFFER: "Capacity buffer",
    FACTOR_WINDOW_MATCH: "Slot match (day + time)",
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
  details.routes { margin-top: 14px; border-top: 1px dashed var(--line); padding-top: 10px; }
  details.routes summary { cursor: pointer; font-size: 13px; font-weight: 600; color: var(--blue); }
  .routelist { margin-top: 10px; display: grid; gap: 8px; }
  .route { border: 1px solid var(--line); border-radius: 9px; padding: 9px 11px; font-size: 12.5px; }
  .route .rtitle { display: flex; justify-content: space-between; gap: 8px; font-weight: 600; }
  .route .status { font-size: 11px; font-weight: 700; padding: 1px 7px; border-radius: 5px; }
  .route .status.ok { background: var(--green-soft); color: var(--green); }
  .route .status.bad { background: var(--red-soft); color: var(--red); }
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
    .grid-2, .grid-3, .legend, .arch-legend { grid-template-columns: 1fr; }
    .examples { grid-template-columns: 1fr; }
    .hero h1 { font-size: 32px; }
    .factor { grid-template-columns: 120px 1fr 40px; }
    .tabbar .wrap { overflow-x: auto; }
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

  <rect x="350" y="12" width="280" height="40" rx="20" fill="#0b2e59"/>
  <text x="490" y="38" text-anchor="middle" fill="#fff" font-size="14" font-weight="600">New customer · 067-123456</text>
  <line x1="490" y1="52" x2="490" y2="86" stroke="#1257a6" stroke-width="2" marker-end="url(#arw)"/>

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

  <text x="334" y="288" text-anchor="middle" font-size="11" font-weight="700" fill="#0b2e59">5 · Decide</text>
  <polygon points="334,300 412,342 334,384 256,342" fill="#fff" stroke="#9a6700" stroke-width="2"/>
  <text x="334" y="338" text-anchor="middle" font-size="11.5" font-weight="700" fill="#9a6700">feasible &amp;</text>
  <text x="334" y="353" text-anchor="middle" font-size="11.5" font-weight="700" fill="#9a6700">high score?</text>
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
  <text x="823" y="332" text-anchor="middle" font-size="12.5" font-weight="700" fill="#5b3fb0">🧠 LLM reasoner (Gemini)</text>
  <text x="823" y="351" text-anchor="middle" font-size="10.5" fill="#6b5aa0">writes the plain-English rationale</text>
  <line x1="690" y1="340" x2="648" y2="340" stroke="#7c8aa0" stroke-width="1.6" stroke-dasharray="5 4" marker-end="url(#arwd)"/>

  <text x="24" y="656" font-size="11.5" fill="#5b6675">→ agent flow &#160;&#160;·&#160;&#160; ⇢ agent calls a service &#160;&#160;·&#160;&#160; ◇ agent's own decision point</text>
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
    outEl.innerHTML = d.resultHtml;
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
  var valid = { overview: 1, architecture: 1, simulator: 1 };
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


def _factor_rows(rec) -> str:
    rows = []
    for f in rec.factor_breakdown:
        label = FACTOR_LABEL.get(f.name, f.name)
        pct = round(f.value * 100)
        rows.append(
            f'<div class="factor"><span class="fname">{_esc(label)}</span>'
            f'<div class="bar"><span style="width:{pct}%"></span></div>'
            f'<span class="fval">{f.value:.2f}</span></div>'
        )
    return "".join(rows)


def _route_rows(candidates: list[CandidateEvaluation]) -> str:
    rows = []
    for cand in candidates:
        r = cand.route
        if cand.feasible:
            status_cls, status = "ok", f"FEASIBLE · {cand.total_score:.2f}"
        else:
            status_cls, status = "bad", "INFEASIBLE"
        checks = []
        for oc in cand.constraint_outcomes:
            sym = '<span class="ok">✓</span>' if oc.passed else '<span class="no">✗</span>'
            label = CONSTRAINT_LABEL.get(oc.name, oc.name)
            checks.append(f"<li>{sym} {_esc(label)}: {_esc(oc.detail)}</li>")
        title = f"{_esc(r.route_id)} · {_esc(r.name)} · {r.day.value} · {cand.distance_miles:.1f} mi"
        rows.append(
            f'<div class="route"><div class="rtitle"><span>{title}</span>'
            f'<span class="status {status_cls}">{status}</span></div>'
            f'<ul class="checks">{"".join(checks)}</ul></div>'
        )
    return "".join(rows)


def _example_card(result: RecommendationResult) -> str:
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

    factors_html = (
        f'<div class="factors">{_factor_rows(rec)}</div>' if rec.factor_breakdown else ""
    )
    reason_label = "Why the agent escalated" if rec.requires_human_review else "Why the agent chose this"

    cnum_text = _esc(c.customer_number) if c.customer_number else "new prospect — no Sysco number yet"
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
          <div class="reason"><span class="lbl">{reason_label}</span>{_esc(rec.reasoning)}</div>
          <details class="routes"><summary>Routes the agent evaluated ({n_routes})</summary>
            <div class="routelist">{_route_rows(result.candidates_considered)}</div>
          </details>
        </div>
      </article>"""


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

    if result.ranked_feasible:
        score = ["Each dimension is normalized to 0–1, then combined by weight:"]
        for e in result.ranked_feasible:
            ctx = build_context(c, e.route)
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

    decide = [
        f"Decision: <b>{DECISION_SHORT[rec.decision]}</b>",
        f"Total score for the winning route: <b>{rec.total_score:.0%}</b> "
        f"(auto-assign bar {config.total_score_threshold:.0%})",
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
            "action": "The agent scores each feasible route on the weighted factors (with the math) and ranks them.",
            "lines": score,
        },
        {
            "title": "Recommend / Decide",
            "action": "The agent picks the best slot, checks its total score against the auto-assign bar, and decides.",
            "lines": decide,
        },
    ]


def _scoring_section(config: Config) -> str:
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
    cfg_rows = "".join(
        [
            row("Route capacity ceiling", f"{config.max_utilization_after_assignment:.0%}", "max_utilization_after_assignment"),
            row(
                "Capacity safety margin (safe up to)",
                f"{config.capacity_buffer_safety_margin:.0%} pts (→ {safe_utilization:.0%})",
                "capacity_buffer_safety_margin",
            ),
            row("Clustering reference (score→0)", f"{config.cluster_reference_miles:.0f} mi", "cluster_reference_miles"),
            row("No-window neutral score", f"{config.window_neutral_score:.2f}", "window_neutral_score"),
            row("Scoring weights (geo/cap/win)", f"{gw:.2f} / {cw:.2f} / {ww:.2f}", "factor_weights"),
            row(
                "Total score threshold (auto-assign bar)",
                f"{config.total_score_threshold:.0%}",
                "total_score_threshold",
            ),
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


def build_page(results: list[RecommendationResult], config: Config) -> str:
    """Render the full three-tab overview HTML from live workflow results."""
    threshold = f"{config.total_score_threshold:.0%}"
    top_n = config.top_n_candidate_routes
    cards = "".join(_example_card(r) for r in results)
    payload = {
        r.customer.lookup_key: {
            "name": _esc(r.customer.name),
            "address": _esc(r.customer.address),
            "steps": _sim_steps(r, config),
            "resultHtml": _example_card(r),
        }
        for r in results
    }
    data_block = (
        '<script type="application/json" id="workflow-data">'
        + json.dumps(payload, ensure_ascii=False)
        + "</script>"
    )
    js_block = "<script>" + _SIM_JS + _TABS_JS + "</script>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Smart Assignment — AI Agent for Delivery Slot Recommendation</title>
<meta name="description" content="An AI agent that autonomously assigns delivery slots for new Sysco customers. Explore the agentic workflow, architecture, and run it live on mock data." />
<!-- GENERATED by scripts/generate_page.py from live workflow output. Do not edit by hand. -->
<style>{_STYLE}</style>
</head>
<body>

<header class="hero">
  <div class="wrap">
    <span class="chip">🤖 Agentic workflow · Sysco</span>
    <h1>Smart Assignment</h1>
    <p class="lead">An <strong>AI agent</strong> that autonomously recommends the best delivery day &amp;
      time slot for a <strong>new customer</strong> — running every step end-to-end, enforcing hard
      operational rules in code, ranking options on weighted business factors, and escalating to a
      human whenever the best option's own score falls short.</p>
    <div class="meta">
      <span class="chip">🤖 Fully agent-automated</span>
      <span class="chip">✅ Deterministic, auditable decisions</span>
      <span class="chip">🧠 LLM-written reasoning</span>
      <span class="chip">🧪 Running on mock data</span>
    </div>
  </div>
</header>

<nav class="tabbar" role="tablist" aria-label="Sections">
  <div class="wrap">
    <button class="tabbtn active" data-tab="overview" role="tab" aria-selected="true">Overview</button>
    <button class="tabbtn" data-tab="architecture" role="tab" aria-selected="false">Architecture</button>
    <button class="tabbtn" data-tab="simulator" role="tab" aria-selected="false">Simulator</button>
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
        phases of the same agent's workflow, not five separate agents. A person is involved only if the
        agent decides to escalate at the final stage.</div></div>
      <div class="flow">
        <div class="step"><div class="num">1</div><h3>Intake</h3><p>Capture the customer's address, order quantity (cases), and preferred slot (day + time).</p><p class="action">Validate the intake profile &amp; build it.</p></div>
        <div class="step"><div class="num">2</div><h3>Geo-Lookup</h3><p>Geocode the address and find the nearest candidate routes.</p><p class="action">Geocode &amp; pick the Top-{top_n} nearest.</p></div>
        <div class="step"><div class="num">3</div><h3>Constraint Check</h3><p>Drop any route that fails a hard rule.</p><p class="action">Enforce serviceability &amp; capacity.</p></div>
        <div class="step"><div class="num">4</div><h3>Score &amp; Rank</h3><p>Rank survivors on weighted business factors.</p><p class="action">Score &amp; order the options.</p></div>
        <div class="step"><div class="num">5</div><h3>Recommend</h3><p>Return the top slot with a reasoning trace — or escalate.</p><p class="action">Check the total score &amp; decide.</p></div>
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
      <p class="sub">Every run ends in one of three states. Anything whose own total score falls below
        {threshold}, or with no valid slot at all, the agent hands to a human — with full context attached.</p>
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
        each of the five steps — it decides <em>when</em> to call which tool and narrates the result, but
        every distance, constraint check, and score comes back from the tool itself, so the decision stays
        reproducible and auditable.</p>
      <div class="arch">
        {_ARCH_SVG}
        <div class="arch-legend">
          <div class="card"><div class="icon">🤖</div><h4>Agent orchestrator</h4><p>An ADK LlmAgent drives all five steps via tool calls, conversationally.</p></div>
          <div class="card"><div class="icon">🧭</div><h4>Deterministic tools</h4><p>Geo, hard constraints, and weighted scoring — plain code the agent calls.</p></div>
          <div class="card"><div class="icon">🗣️</div><h4>Narrates, doesn't decide</h4><p>The same agent explains the already-decided result in its own words — it never computes a number itself.</p></div>
          <div class="card"><div class="icon">🙋</div><h4>Human-in-the-loop</h4><p>On escalation, the agent pauses the conversation and waits for a specialist's reply.</p></div>
        </div>
      </div>
    </div>
  </section>

  <section style="background:#fff; border-top:1px solid var(--line);">
    <div class="wrap">
      <span class="eyebrow">Key mechanics</span>
      <h2>What makes it trustworthy</h2>
      <div class="grid-2">
        <div class="card"><h4>🤖 One agent, five steps</h4><p>A single ADK LlmAgent calls one tool per step — Intake → Geo-Lookup → Constraint Check → Score &amp; Rank → Decide — and pauses to escalate when needed.</p></div>
        <div class="card"><h4>✅ Decisions are code, not vibes</h4><p>Constraints and scoring are deterministic Python, so identical inputs always yield the identical recommendation.</p></div>
        <div class="card"><h4>🧠 The agent narrates, never computes</h4><p>It presents the deterministic reasoning trace in its own words; it never changes a number, a route, or the decision itself.</p></div>
        <div class="card"><h4>🙋 Human-in-the-loop on escalation</h4><p>The agent pauses and waits for a specialist's reply (via ADK's request_input tool) when it finds no feasible slot, or the best option's own total score falls short of the auto-assign bar.</p></div>
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

</main>

<footer>
  <div class="wrap">
    Smart Assignment · an AI agent for delivery slot recommendation ·
    built on Google's Agent Development Kit (ADK).<br />
    Source &amp; docs on <a href="https://github.com/MuhammadVT/smart-assignment">GitHub</a>.
    Decisions are deterministic and auditable; reasoning narration is LLM-written with a deterministic fallback.
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
