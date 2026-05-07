"""
tests/conftest.py

Shared fixtures for the Ali Real Estate Chatbot evaluation suite.
No global mocks are applied – all external dependencies are used as installed.
If a dependency is missing, the corresponding tests will be skipped naturally
(e.g., via pytest.skip in the test itself).
"""

import os
import asyncio
import tempfile
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Ensure backend modules are importable
# ---------------------------------------------------------------------------

_BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

for p in (_BACKEND_DIR, _PROJECT_ROOT):
    if str(p) not in __import__("sys").path:
        __import__("sys").path.insert(0, str(p))

# ---------------------------------------------------------------------------
# Configuration – Base URL for live server tests
# ---------------------------------------------------------------------------

CHATBOT_BASE_URL = os.environ.get("CHATBOT_BASE_URL", "http://localhost:8000")
CHATBOT_WS_URL = os.environ.get("CHATBOT_WS_URL", "ws://localhost:8000/ws/chat")


@pytest.fixture()
def base_url():
    """Return the base URL of the chatbot server."""
    return CHATBOT_BASE_URL


@pytest.fixture()
def ws_url():
    """Return the WebSocket URL of the chatbot server."""
    return CHATBOT_WS_URL


@pytest.fixture()
def clear_sessions():
    """Wipe the in-memory session store before each test."""
    from backend.Conversation.conversation import _sessions
    _sessions.clear()
    yield
    _sessions.clear()


@pytest.fixture()
def temp_db():
    """Provide a temporary SQLite database for CRM tests."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    yield db_path
    os.unlink(db_path)


@pytest.fixture()
def temp_calendar_db():
    """Provide a temporary SQLite database for calendar tests."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    yield db_path
    os.unlink(db_path)


@pytest.fixture()
def event_loop():
    """Create a new event loop for each test function."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()