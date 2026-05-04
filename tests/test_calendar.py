"""
tests/test_calendar.py

Unit tests for the Calendar tool — CRUD operations, date filtering,
persistence, and edge cases.
"""

import sys
import os
import asyncio
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

_BACKEND = Path(__file__).resolve().parent.parent / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))


# ---------------------------------------------------------------------------
# Fixture: isolated calendar DB for each test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolated_calendar_db(tmp_path):
    """Redirect the calendar module to use a temporary database."""
    temp_db = tmp_path / "test_calendar.db"
    with patch("Tools.calendar.DB_PATH", temp_db):
        # Re-initialize the DB schema in the temp location
        from Tools.calendar import _init_db
        with patch("Tools.calendar.DB_PATH", temp_db):
            conn = sqlite3.connect(str(temp_db))
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    description TEXT NOT NULL
                )
            ''')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_date ON events(date)')
            conn.commit()
            conn.close()
        yield temp_db


# ---------------------------------------------------------------------------
# Functional Correctness — CRUD
# ---------------------------------------------------------------------------

class TestCalendarCRUD:
    """Tests for adding and retrieving events."""

    @pytest.mark.asyncio
    async def test_add_event_returns_success(self, isolated_calendar_db):
        with patch("Tools.calendar.DB_PATH", isolated_calendar_db):
            from Tools.calendar import add_event
            result = await add_event("2026-05-15", "Visit DHA Phase 6")
            assert "Successfully" in result
            assert "2026-05-15" in result

    @pytest.mark.asyncio
    async def test_add_and_retrieve_event(self, isolated_calendar_db):
        with patch("Tools.calendar.DB_PATH", isolated_calendar_db):
            from Tools.calendar import add_event, get_events
            await add_event("2026-05-15", "Visit DHA Phase 6")
            events = await get_events("2026-05-15")
            assert len(events) == 1
            assert events[0]["date"] == "2026-05-15"
            assert "DHA Phase 6" in events[0]["description"]

    @pytest.mark.asyncio
    async def test_add_multiple_events_same_date(self, isolated_calendar_db):
        with patch("Tools.calendar.DB_PATH", isolated_calendar_db):
            from Tools.calendar import add_event, get_events
            await add_event("2026-05-15", "Morning visit")
            await add_event("2026-05-15", "Afternoon meeting")
            events = await get_events("2026-05-15")
            assert len(events) == 2

    @pytest.mark.asyncio
    async def test_get_events_filters_by_date(self, isolated_calendar_db):
        with patch("Tools.calendar.DB_PATH", isolated_calendar_db):
            from Tools.calendar import add_event, get_events
            await add_event("2026-05-15", "Event A")
            await add_event("2026-05-16", "Event B")
            events_15 = await get_events("2026-05-15")
            events_16 = await get_events("2026-05-16")
            assert len(events_15) == 1
            assert len(events_16) == 1
            assert events_15[0]["description"] == "Event A"

    @pytest.mark.asyncio
    async def test_get_all_events(self, isolated_calendar_db):
        with patch("Tools.calendar.DB_PATH", isolated_calendar_db):
            from Tools.calendar import add_event, get_events
            await add_event("2026-05-15", "Event A")
            await add_event("2026-05-16", "Event B")
            await add_event("2026-05-17", "Event C")
            all_events = await get_events()
            assert len(all_events) == 3

    @pytest.mark.asyncio
    async def test_get_events_empty_date(self, isolated_calendar_db):
        with patch("Tools.calendar.DB_PATH", isolated_calendar_db):
            from Tools.calendar import get_events
            events = await get_events("2099-12-31")
            assert len(events) == 0

    @pytest.mark.asyncio
    async def test_get_events_no_filter_empty(self, isolated_calendar_db):
        with patch("Tools.calendar.DB_PATH", isolated_calendar_db):
            from Tools.calendar import get_events
            events = await get_events()
            assert len(events) == 0


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------

class TestCalendarEdgeCases:
    """Tests for edge cases and missing arguments."""

    @pytest.mark.asyncio
    async def test_add_event_no_description(self, isolated_calendar_db):
        """When description is None, the tool builds one from kwargs or uses default."""
        with patch("Tools.calendar.DB_PATH", isolated_calendar_db):
            from Tools.calendar import add_event, get_events
            result = await add_event("2026-06-01")
            assert "Successfully" in result
            events = await get_events("2026-06-01")
            assert len(events) == 1
            assert events[0]["description"] == "No description provided."

    @pytest.mark.asyncio
    async def test_add_event_with_kwargs(self, isolated_calendar_db):
        """When description is None but kwargs are present, description is built from them."""
        with patch("Tools.calendar.DB_PATH", isolated_calendar_db):
            from Tools.calendar import add_event, get_events
            result = await add_event("2026-06-01", title="Property Visit", location="DHA")
            events = await get_events("2026-06-01")
            assert len(events) == 1
            assert "Title" in events[0]["description"] or "Property Visit" in events[0]["description"]

    @pytest.mark.asyncio
    async def test_event_ordering_by_date(self, isolated_calendar_db):
        with patch("Tools.calendar.DB_PATH", isolated_calendar_db):
            from Tools.calendar import add_event, get_events
            await add_event("2026-05-20", "Late event")
            await add_event("2026-05-10", "Early event")
            all_events = await get_events()
            assert all_events[0]["date"] <= all_events[1]["date"]

    @pytest.mark.asyncio
    async def test_event_structure(self, isolated_calendar_db):
        with patch("Tools.calendar.DB_PATH", isolated_calendar_db):
            from Tools.calendar import add_event, get_events
            await add_event("2026-07-01", "Test structure")
            events = await get_events("2026-07-01")
            event = events[0]
            assert "id" in event
            assert "date" in event
            assert "description" in event
            assert isinstance(event["id"], int)
