"""
Tests for the live web app (smart_assignment/webapp/app.py).

These exercise the HTTP surface end-to-end with FastAPI's TestClient: the
recommendation endpoint returns the same Simulator payload the static page uses,
and numbers match the deterministic pipeline. Skipped cleanly if the optional
`web` extra (fastapi) is not installed.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi", reason="install the web extra: pip install -e '.[web]'")

from fastapi.testclient import TestClient  # noqa: E402

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
