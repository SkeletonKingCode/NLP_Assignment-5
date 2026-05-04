"""
tests/test_api.py

Integration tests for the FastAPI REST endpoints.
Uses FastAPI's TestClient (synchronous) so no running server is needed.
Voice and LLM dependencies are mocked via conftest.py.
"""

import sys
from pathlib import Path

import pytest

# Ensure backend is importable
_BACKEND = Path(__file__).resolve().parent.parent / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

# conftest.py already stubs Voice.asr / Voice.tts before this import
from api.main import app  # noqa: E402

from fastapi.testclient import TestClient

client = TestClient(app)


# ── Health Check ─────────────────────────────────────────────────────────────

class TestHealthEndpoint:

    def test_health_returns_200(self):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_body_structure(self):
        data = client.get("/health").json()
        assert data["status"] == "ok"
        assert "active_connections" in data
        assert "timestamp" in data

    def test_health_active_connections_is_integer(self):
        data = client.get("/health").json()
        assert isinstance(data["active_connections"], int)

    def test_health_timestamp_is_positive(self):
        data = client.get("/health").json()
        assert data["timestamp"] > 0


# ── Session Lifecycle ────────────────────────────────────────────────────────

class TestSessionEndpoints:

    def test_create_session(self):
        resp = client.post("/session")
        assert resp.status_code == 201
        data = resp.json()
        assert "session_id" in data
        assert len(data["session_id"]) == 36  # UUID

    def test_create_session_message(self):
        resp = client.post("/session")
        data = resp.json()
        assert "message" in data
        assert "created" in data["message"].lower() or "success" in data["message"].lower()

    def test_get_session(self):
        sid = client.post("/session").json()["session_id"]
        resp = client.get(f"/session/{sid}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == sid
        assert data["stage"] == "greeting"

    def test_get_session_not_found(self):
        resp = client.get("/session/does-not-exist")
        assert resp.status_code == 404

    def test_delete_session(self):
        sid = client.post("/session").json()["session_id"]
        resp = client.delete(f"/session/{sid}")
        assert resp.status_code == 200
        # Verify it's actually gone
        assert client.get(f"/session/{sid}").status_code == 404

    def test_delete_session_not_found(self):
        resp = client.delete("/session/does-not-exist")
        assert resp.status_code == 404

    def test_session_initial_state(self):
        sid = client.post("/session").json()["session_id"]
        data = client.get(f"/session/{sid}").json()
        assert data["stage"] == "greeting"
        assert data["selected_category"] is None
        assert data["selected_subtype"] is None
        assert data["selected_price"] is None
        assert data["turn_count"] == 0


# ── Root / Frontend ──────────────────────────────────────────────────────────

class TestRootEndpoint:

    def test_root_serves_page(self):
        resp = client.get("/")
        assert resp.status_code == 200
        # Should serve the HTML file or a JSON fallback
        content_type = resp.headers.get("content-type", "")
        assert "html" in content_type or "json" in content_type

    def test_docs_endpoint(self):
        resp = client.get("/docs")
        assert resp.status_code == 200


# ── Synth Endpoint (mocked TTS) ─────────────────────────────────────────────

class TestSynthEndpoint:

    def test_synth_empty_text_returns_error(self):
        resp = client.post("/synth", json={"text": ""})
        assert resp.status_code in [400, 503]  # 400 if validated, 503 if TTS unavailable

    def test_synth_tts_unavailable(self):
        """TTS is mocked as unavailable in conftest."""
        resp = client.post("/synth", json={"text": "Hello world"})
        # Should return 503 since TTS is stubbed as unavailable
        assert resp.status_code == 503
