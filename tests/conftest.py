"""
tests/conftest.py

Shared fixtures for the Ali Real Estate Chatbot evaluation suite.
Patches heavy dependencies (Ollama, ASR, TTS) so unit tests run
without any external services. Integration/performance tests that
need the live server use a separate configuration.
"""

import sys
import os
import json
import asyncio
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Ensure backend modules are importable
# ---------------------------------------------------------------------------

_BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Stub out ALL heavy external dependencies BEFORE anything imports them.
# Order matters: ollama must be in sys.modules before conversation.py loads.
# ---------------------------------------------------------------------------

# 1) ollama — used by Conversation.conversation
_ollama_stub = MagicMock()
_ollama_stub.AsyncClient = MagicMock
_ollama_stub.ResponseError = type("ResponseError", (Exception,), {"error": ""})
sys.modules.setdefault("ollama", _ollama_stub)

# 2) faster_whisper — used by Voice.asr
sys.modules.setdefault("faster_whisper", MagicMock())

# 3) piper / piper.voice — used by Voice.tts
sys.modules.setdefault("piper", MagicMock())
sys.modules.setdefault("piper.voice", MagicMock())

# 4) Voice module stubs — so api.main can do `from Voice import asr, tts`
_asr_stub = MagicMock()
_asr_stub.preload = MagicMock()
_asr_stub.transcribe = MagicMock(return_value="hello world")

_tts_stub = MagicMock()
_tts_stub.preload = MagicMock()
_tts_stub.is_available = MagicMock(return_value=False)
_tts_stub.synthesize = MagicMock(return_value=b"RIFF----WAVEfmt ")

sys.modules["Voice"] = MagicMock(asr=_asr_stub, tts=_tts_stub)
sys.modules["Voice.asr"] = _asr_stub
sys.modules["Voice.tts"] = _tts_stub


# ---------------------------------------------------------------------------
# Configuration — Base URL for live server tests
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
    from Conversation.conversation import _sessions
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
