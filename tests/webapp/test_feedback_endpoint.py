"""
Tests for the /api/feedback HTTP surface and the payload feedback stamp.

Exercised end-to-end with FastAPI's TestClient. The feature is flag-gated, so
these flip ``DEFAULT_CONFIG.use_human_feedback`` for the duration of a test and
point the log at a tmp path, asserting: the capability shows up in /api/mode, a
result payload is stamped with a decision_id only when enabled, and a posted
annotation is persisted (and validated).
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi", reason="install the web extra: pip install -e '.[web]'")

from fastapi.testclient import TestClient  # noqa: E402

from smart_assignment.webapp import app as app_module  # noqa: E402
from smart_assignment.webapp.app import app  # noqa: E402
from smart_assignment.feedback.store import read_records  # noqa: E402

client = TestClient(app)


@pytest.fixture
def feedback_on(tmp_path, monkeypatch):
    """Turn the feedback loop on for the app's DEFAULT_CONFIG and isolate its log
    + in-memory context map, restoring both afterwards."""
    cfg = app_module.DEFAULT_CONFIG
    log = str(tmp_path / "annotations.jsonl")
    monkeypatch.setattr(cfg, "use_human_feedback", True, raising=False)
    monkeypatch.setattr(cfg, "feedback_scrub_pii", True, raising=False)
    monkeypatch.setattr(cfg, "feedback_log_path", log, raising=False)
    monkeypatch.setattr(cfg, "use_tracing", False, raising=False)
    app_module._feedback_context.clear()
    yield log
    app_module._feedback_context.clear()


def test_mode_advertises_feedback_capability(feedback_on):
    assert client.get("/api/mode").json().get("feedback") is True


def test_mode_feedback_off_by_default():
    # Without the fixture the DEFAULT_CONFIG flag is off.
    if app_module.DEFAULT_CONFIG.use_human_feedback:
        pytest.skip("environment enabled feedback")
    assert client.get("/api/mode").json().get("feedback") is False


def test_feedback_disabled_returns_disabled(monkeypatch):
    monkeypatch.setattr(app_module.DEFAULT_CONFIG, "use_human_feedback", False, raising=False)
    resp = client.post("/api/feedback", json={"decision_id": "d1", "label": "thumbs_up"})
    assert resp.status_code == 200
    assert resp.json() == {"ok": False, "disabled": True}


def test_recommend_payload_is_stamped_when_enabled(feedback_on):
    resp = client.post(
        "/api/recommend",
        json={"message": "1200 McKinney St, Houston, TX 77010, 90 cases, TUE 07:00-10:00"},
    )
    body = resp.json()
    assert body["ok"] is True
    fb = body["payload"].get("feedback")
    assert fb and fb["enabled"] is True and fb["decision_id"]


def test_post_feedback_persists(feedback_on):
    # First produce a stamped decision so its context is remembered server-side.
    rec = client.post(
        "/api/recommend",
        json={"message": "1200 McKinney St, Houston, TX 77010, 90 cases, TUE 07:00-10:00"},
    ).json()
    decision_id = rec["payload"]["feedback"]["decision_id"]

    resp = client.post(
        "/api/feedback",
        json={
            "decision_id": decision_id,
            "label": "thumbs_down",
            "score": 0,
            "note": "wrong slot",
            "session_id": "s1",
        },
    )
    assert resp.status_code == 200
    assert resp.json()["ok"] is True

    stored = read_records(feedback_on)
    assert len(stored) == 1
    assert stored[0].target.decision_id == decision_id
    assert stored[0].label == "thumbs_down"
    # The remembered decision context rode along for curation.
    assert stored[0].context.get("outcome") in ("recommend", "escalate", None)


def test_post_feedback_rejects_bad_label(feedback_on):
    resp = client.post("/api/feedback", json={"decision_id": "d1", "label": ""})
    assert resp.status_code == 400


# --- Increments A + B: outcome context + real trace linkage plumbing ------------


def test_attach_feedback_consumes_decision_and_trace_hints(feedback_on):
    # Simulate what the chat services / recommend stamp on a payload.
    payload = {
        "name": "Galleria Grill",
        "address": "5085 Westheimer Rd",
        "_decision": {"outcome": "escalate", "recommended_route_id": "RTE-1"},
        "_trace": {"trace_id": "a" * 32, "span_id": "b" * 16},
    }
    app_module._attach_feedback(payload, "s9")

    # The private hints are stripped (never reach the browser)...
    assert "_decision" not in payload and "_trace" not in payload
    # ...and the feedback stamp carries the REAL trace link from the hint.
    fb = payload["feedback"]
    assert fb["enabled"] is True
    assert fb["trace_id"] == "a" * 32 and fb["span_id"] == "b" * 16
    # The remembered context carries the recommend/escalate outcome for curation.
    remembered = app_module._feedback_context[fb["decision_id"]]["context"]
    assert remembered["outcome"] == "escalate"
    assert remembered["recommended_route_id"] == "RTE-1"


def test_attach_feedback_strips_hints_even_when_disabled(monkeypatch):
    monkeypatch.setattr(app_module.DEFAULT_CONFIG, "use_human_feedback", False, raising=False)
    payload = {"name": "X", "_decision": {"outcome": "recommend"}, "_trace": {"trace_id": "z"}}
    app_module._attach_feedback(payload, "s1")
    # No feedback block when off, but the private hints must not leak either.
    assert "feedback" not in payload
    assert "_decision" not in payload and "_trace" not in payload


def test_recommend_persists_outcome_context(feedback_on):
    rec = client.post(
        "/api/recommend",
        json={"message": "1200 McKinney St, Houston, TX 77010, 90 cases, TUE 07:00-10:00"},
    ).json()
    decision_id = rec["payload"]["feedback"]["decision_id"]
    client.post("/api/feedback", json={"decision_id": decision_id, "label": "thumbs_down"})

    stored = read_records(feedback_on)[0]
    # Increment A: the recommend/escalate outcome now rides along on every path.
    assert stored.context.get("outcome") in ("recommend", "escalate")
    assert stored.context.get("order_quantity_cases") == 90
