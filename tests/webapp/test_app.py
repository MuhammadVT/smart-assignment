"""
Tests for the live web app (smart_assignment/webapp/app.py).

These exercise the HTTP surface end-to-end with FastAPI's TestClient: the
recommendation endpoint returns the same Simulator payload the static page uses,
and numbers match the deterministic pipeline. Skipped cleanly if the optional
`web` extra (fastapi) is not installed.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi", reason="install the web extra: pip install -e '.[web]'")

from fastapi.testclient import TestClient  # noqa: E402

from smart_assignment.webapp import app as app_module  # noqa: E402
from smart_assignment.webapp.app import app  # noqa: E402

client = TestClient(app)


def test_index_serves_html_with_shared_style():
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.text
    assert "<!DOCTYPE html>" in body
    # The shared Simulator CSS was injected (placeholder replaced).
    assert "/*__SHARED_STYLE__*/" not in body
    assert ".sim-step" in body


def test_index_cache_busts_assets():
    """The page must revalidate (no-cache) and reference app.css/app.js with a
    content-hash ?v= query, so a new build isn't hidden by a stale browser cache."""
    resp = client.get("/")
    assert resp.headers.get("cache-control") == "no-cache"
    assert "/static/app.js?v=" in resp.text
    assert "/static/app.css?v=" in resp.text
    # No un-versioned references left behind.
    assert 'src="/static/app.js"' not in resp.text
    assert 'href="/static/app.css"' not in resp.text


def test_asset_version_reflects_content():
    from smart_assignment.webapp.app import _asset_version

    assert _asset_version("app.js") not in ("", "0")
    assert _asset_version("does-not-exist.xyz") == "0"


def test_samples_endpoint_lists_prospects():
    resp = client.get("/api/samples")
    assert resp.status_code == 200
    samples = resp.json()
    assert len(samples) == 4
    assert all("name" in s and "message" in s for s in samples)
    assert any("cases" in s["message"] for s in samples)


def test_recommend_returns_five_step_payload_and_result():
    resp = client.post(
        "/api/recommend",
        json={"message": "1200 McKinney St, Houston, TX 77010, 90 cases, TUE 07:00-10:00"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    payload = data["payload"]
    assert [s["title"] for s in payload["steps"]] == [
        "Intake",
        "Geo-Lookup",
        "Constraint Check",
        "Score & Rank",
        "Recommend / Decide",
    ]
    # A clean downtown recommend should render a recommend pill in the result card.
    assert "Recommended" in payload["resultHtml"]
    # The evaluated-routes breakdown is a separate default-open section (rendered
    # below the map in the UI), not embedded in the result card.
    assert "Routes the agent evaluated" not in payload["resultHtml"]
    assert payload["routesHtml"].startswith('<details class="routes" open>')
    assert "★ recommended" in payload["routesHtml"]
    # Proximity-map data rides along in the same payload -- customer + every
    # evaluated route's service center and stops, for the frontend map.
    m = payload["map"]
    assert m is not None
    assert isinstance(m["customer"]["lat"], float) and isinstance(m["customer"]["lng"], float)
    assert len(m["routes"]) == 3  # DEFAULT_CONFIG.top_n_candidate_routes
    assert any(r["feasible"] for r in m["routes"])
    assert all("service_center" in r and "stops" in r for r in m["routes"])


def test_recommend_escalates_large_catering_order():
    resp = client.post(
        "/api/recommend",
        json={"message": "5085 Westheimer Rd, Houston, TX 77056, 400 cases"},
    )
    data = resp.json()
    assert data["ok"] is True
    # Matches the published Simulator / run_local.py: low-score escalation.
    assert "human review" in data["payload"]["resultHtml"].lower()


def test_recommend_asks_for_missing_fields():
    resp = client.post("/api/recommend", json={"message": "how does this work?"})
    data = resp.json()
    assert data["ok"] is False
    assert data["needs_input"] is True
    assert data["reply"]


def test_sample_message_round_trips_through_recommend():
    samples = client.get("/api/samples").json()
    for s in samples:
        resp = client.post("/api/recommend", json={"message": s["message"]})
        data = resp.json()
        assert data["ok"] is True, f"sample {s['name']!r} failed: {data}"
        assert len(data["payload"]["steps"]) == 5


def test_frontend_page_serves_html_with_fe_styles():
    resp = client.get("/frontend")
    assert resp.status_code == 200
    body = resp.text
    assert "<!DOCTYPE html>" in body
    # The shared Simulator CSS + the Frontend tab's own styles were injected.
    assert "/*__SHARED_STYLE__*/" not in body
    assert ".fe-opt" in body  # _FE_STYLE present
    assert "Choose a delivery slot" in body
    # Assets are content-hash versioned and the page revalidates (like index).
    assert "/static/frontend.js?v=" in body
    assert "/static/frontend.css?v=" in body
    assert resp.headers.get("cache-control") == "no-cache"


def test_api_frontend_empty_before_any_run():
    """Until a session has produced a result, the customer view has nothing to
    show -- the page falls back to its empty state."""
    data = client.get("/api/frontend", params={"session": "never-ran"}).json()
    assert data == {"ok": False}


def test_api_frontend_shows_slots_after_a_chat_run():
    """A completed chat turn caches its slots so the read-only customer view can
    render exactly what the chat produced -- no re-run, no new decision."""
    sid = "fe-sess-1"
    resp = client.post(
        "/api/chat",
        json={
            "session_id": sid,
            "message": "1200 McKinney St, Houston, TX 77010, 90 cases, TUE 07:00-10:00",
        },
    )
    assert resp.status_code == 200
    assert any(f["type"] == "visualization" for f in _sse_frames(resp.text))

    data = client.get("/api/frontend", params={"session": sid}).json()
    assert data["ok"] is True
    assert data["name"]
    assert 'class="fe-grid"' in data["frontendHtml"]
    assert "fe-opt" in data["frontendHtml"]

    # A session that never ran anything stays empty -- results are per-session.
    assert client.get("/api/frontend", params={"session": "other-sess"}).json() == {"ok": False}


def test_mode_endpoint_reports_a_mode():
    resp = client.get("/api/mode")
    assert resp.status_code == 200
    data = resp.json()
    assert data["mode"] in ("llm", "deterministic")
    assert "configured" in data


def _sse_frames(text: str) -> list[dict]:
    frames = []
    for block in text.split("\n\n"):
        block = block.strip()
        if block.startswith("data:"):
            frames.append(json.loads(block[len("data:") :].strip()))
    return frames


class _BrokenService:
    async def stream_turn(self, session_id, message):
        raise RuntimeError("no credentials")
        yield  # pragma: no cover - makes this an async generator


def test_chat_falls_back_to_deterministic_on_llm_error(monkeypatch):
    """When the LLM path raises (no creds / model error), /api/chat degrades to
    the deterministic brain so the chat never dead-ends."""
    monkeypatch.setattr(app_module, "resolve_mode", lambda *a, **k: {"mode": "llm"})
    monkeypatch.setattr(app_module, "_chat_service", _BrokenService())
    resp = client.post(
        "/api/chat",
        json={
            "session_id": "t1",
            "message": "1200 McKinney St, Houston, TX 77010, 90 cases, TUE 07:00-10:00",
        },
    )
    assert resp.status_code == 200
    frames = _sse_frames(resp.text)
    viz = [f for f in frames if f["type"] == "visualization"]
    assert viz and len(viz[0]["payload"]["steps"]) == 5
    assert frames[-1] == {"type": "done"}
    # The agent error is surfaced (not silently swallowed) so a downgrade never
    # masquerades as agent parity: a notice naming the failure precedes the
    # deterministic frames.
    messages = [f["text"] for f in frames if f["type"] == "message"]
    assert any("agent errored" in m and "RuntimeError" in m for m in messages)


def test_chat_fallback_asks_for_missing_fields(monkeypatch):
    monkeypatch.setattr(app_module, "resolve_mode", lambda *a, **k: {"mode": "llm"})
    monkeypatch.setattr(app_module, "_chat_service", _BrokenService())
    resp = client.post("/api/chat", json={"session_id": "t2", "message": "how does this work?"})
    frames = _sse_frames(resp.text)
    assert any(f["type"] == "message" for f in frames)
    assert frames[-1] == {"type": "done"}


def test_chat_deterministic_conversation_remembers_context():
    """The deterministic chat is session-aware: a follow-up that only gives the
    order size still runs, because the address from an earlier turn is remembered
    -- and a later revision ("try 20 cases") re-runs without re-stating the
    address. This is the multi-turn behaviour that matches ``adk web``."""
    sid = "conv-remember"

    def turn(msg):
        return _sse_frames(client.post("/api/chat", json={"session_id": sid, "message": msg}).text)

    # Turn 1: address only -> asks for the order size, no run yet.
    f1 = turn("1200 McKinney St, Houston, TX 77010")
    assert not [f for f in f1 if f["type"] == "visualization"]
    assert any(f["type"] == "message" and "cases" in f["text"].lower() for f in f1)

    # Turn 2: just the cases -> runs, because the address is remembered.
    f2 = turn("90 cases")
    assert [f for f in f2 if f["type"] == "visualization"]

    # Turn 3: a revision that never repeats the address -> still runs.
    f3 = turn("try 20 cases")
    viz3 = [f for f in f3 if f["type"] == "visualization"]
    assert viz3
    # The run reflects the revised order size (20 cases), proving the merge.
    assert "20 cases" in viz3[0]["payload"]["resultHtml"]
