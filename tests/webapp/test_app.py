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


def test_chat_falls_back_to_deterministic_on_llm_error(monkeypatch):
    """When the LLM path raises (no creds / model error), /api/chat degrades to a
    deterministic run so the chat never dead-ends."""

    class _BrokenService:
        async def stream_turn(self, session_id, message):
            raise RuntimeError("no credentials")
            yield  # pragma: no cover - makes this an async generator

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


def test_chat_fallback_asks_for_missing_fields(monkeypatch):
    class _BrokenService:
        async def stream_turn(self, session_id, message):
            raise RuntimeError("no credentials")
            yield  # pragma: no cover

    monkeypatch.setattr(app_module, "_chat_service", _BrokenService())
    resp = client.post("/api/chat", json={"session_id": "t2", "message": "how does this work?"})
    frames = _sse_frames(resp.text)
    assert any(f["type"] == "message" for f in frames)
    assert frames[-1] == {"type": "done"}
