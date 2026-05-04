"""
tests/test_crm.py

Unit and integration tests for the CRM module — CRUD correctness,
semantic field matching, persistence, and error handling.
Tests call the CRM tool directly (not through the LLM).
"""

import sys
import os
import asyncio
import sqlite3
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock

import pytest

_BACKEND = Path(__file__).resolve().parent.parent / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))


# ---------------------------------------------------------------------------
# Fixture: isolated CRM DB for each test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolated_crm_db(tmp_path):
    """Redirect the CRM module to use a temporary database."""
    temp_db = tmp_path / "test_crm.db"
    with patch("CRM.crm.DB_PATH", temp_db):
        # Initialize schema
        conn = sqlite3.connect(str(temp_db))
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                data TEXT
            )
        ''')
        conn.commit()
        conn.close()
        yield temp_db


# ---------------------------------------------------------------------------
# CRUD Correctness — Direct calls
# ---------------------------------------------------------------------------

class TestCRMCrud:
    """Tests for Create, Read, Update operations on CRM."""

    @pytest.mark.asyncio
    async def test_create_user(self, isolated_crm_db):
        with patch("CRM.crm.DB_PATH", isolated_crm_db):
            from CRM.crm import _create_user_sync, _get_user_info_sync
            _create_user_sync("user_001", {"name": "Ali", "budget": "1 Crore"})
            data = _get_user_info_sync("user_001")
            assert data["name"] == "Ali"
            assert data["budget"] == "1 Crore"

    @pytest.mark.asyncio
    async def test_read_nonexistent_user(self, isolated_crm_db):
        with patch("CRM.crm.DB_PATH", isolated_crm_db):
            from CRM.crm import _get_user_info_sync
            data = _get_user_info_sync("nonexistent_user")
            assert data == {}

    @pytest.mark.asyncio
    async def test_update_existing_field(self, isolated_crm_db):
        with patch("CRM.crm.DB_PATH", isolated_crm_db):
            from CRM.crm import _create_user_sync, _update_user_info_sync, _get_user_info_sync
            _create_user_sync("user_002", {"budget": "1 Crore"})
            _update_user_info_sync("user_002", "budget", "2 Crore")
            data = _get_user_info_sync("user_002")
            assert data["budget"] == "2 Crore"

    @pytest.mark.asyncio
    async def test_update_adds_new_field(self, isolated_crm_db):
        with patch("CRM.crm.DB_PATH", isolated_crm_db):
            from CRM.crm import _create_user_sync, _update_user_info_sync, _get_user_info_sync
            _create_user_sync("user_003", {"name": "Ahmed"})
            _update_user_info_sync("user_003", "phone", "03001234567")
            data = _get_user_info_sync("user_003")
            assert data["name"] == "Ahmed"
            assert data["phone"] == "03001234567"

    @pytest.mark.asyncio
    async def test_overwrite_user(self, isolated_crm_db):
        """create_user with same ID should overwrite."""
        with patch("CRM.crm.DB_PATH", isolated_crm_db):
            from CRM.crm import _create_user_sync, _get_user_info_sync
            _create_user_sync("user_004", {"name": "Old"})
            _create_user_sync("user_004", {"name": "New", "city": "Lahore"})
            data = _get_user_info_sync("user_004")
            assert data["name"] == "New"
            assert data["city"] == "Lahore"

    @pytest.mark.asyncio
    async def test_update_nonexistent_user_creates_entry(self, isolated_crm_db):
        """Updating a user that does not exist should create the record."""
        with patch("CRM.crm.DB_PATH", isolated_crm_db):
            from CRM.crm import _update_user_info_sync, _get_user_info_sync
            _update_user_info_sync("user_005", "budget", "5 Crore")
            data = _get_user_info_sync("user_005")
            assert data["budget"] == "5 Crore"


# ---------------------------------------------------------------------------
# Data Persistence
# ---------------------------------------------------------------------------

class TestCRMPersistence:
    """Tests that data persists across separate read operations."""

    @pytest.mark.asyncio
    async def test_data_persists(self, isolated_crm_db):
        with patch("CRM.crm.DB_PATH", isolated_crm_db):
            from CRM.crm import _create_user_sync, _get_user_info_sync
            _create_user_sync("persist_user", {"key": "value"})
            # Read again (simulates different request)
            data = _get_user_info_sync("persist_user")
            assert data["key"] == "value"

    @pytest.mark.asyncio
    async def test_multiple_fields_persist(self, isolated_crm_db):
        with patch("CRM.crm.DB_PATH", isolated_crm_db):
            from CRM.crm import _create_user_sync, _update_user_info_sync, _get_user_info_sync
            _create_user_sync("multi_user", {"name": "Test"})
            _update_user_info_sync("multi_user", "budget", "3 Crore")
            _update_user_info_sync("multi_user", "phone", "0300")
            _update_user_info_sync("multi_user", "city", "Karachi")
            data = _get_user_info_sync("multi_user")
            assert len(data) == 4
            assert data["name"] == "Test"
            assert data["budget"] == "3 Crore"
            assert data["phone"] == "0300"
            assert data["city"] == "Karachi"


# ---------------------------------------------------------------------------
# Async Wrapper Tests
# ---------------------------------------------------------------------------

class TestCRMAsync:
    """Tests for the async wrapper functions."""

    @pytest.mark.asyncio
    async def test_async_get_user_info(self, isolated_crm_db):
        with patch("CRM.crm.DB_PATH", isolated_crm_db):
            from CRM.crm import _create_user_sync
            _create_user_sync("async_user", {"name": "AsyncTest"})
            
            from CRM.crm import get_user_info
            # Mock the semantic memory parts to avoid ChromaDB dependency
            with patch("CRM.crm._find_semantic_field", new_callable=AsyncMock, return_value="name"):
                with patch("CRM.crm._sync_memory_entry", new_callable=AsyncMock):
                    data = await get_user_info("async_user")
                    assert data["name"] == "AsyncTest"

    @pytest.mark.asyncio
    async def test_async_update_user_info(self, isolated_crm_db):
        with patch("CRM.crm.DB_PATH", isolated_crm_db):
            from CRM.crm import _create_user_sync
            _create_user_sync("async_update", {"budget": "1 Crore"})
            
            from CRM.crm import update_user_info
            with patch("CRM.crm._find_semantic_field", new_callable=AsyncMock, return_value="budget"):
                with patch("CRM.crm._sync_memory_entry", new_callable=AsyncMock):
                    data = await update_user_info("async_update", "budget", "3 Crore")
                    assert data["budget"] == "3 Crore"

    @pytest.mark.asyncio
    async def test_async_create_user(self, isolated_crm_db):
        with patch("CRM.crm.DB_PATH", isolated_crm_db):
            from CRM.crm import create_user, get_user_info
            with patch("CRM.crm._sync_memory_entry", new_callable=AsyncMock):
                result = await create_user("new_async", {"name": "NewUser", "budget": "5 Lac"})
                assert result is True
            
            with patch("CRM.crm._find_semantic_field", new_callable=AsyncMock, return_value="name"):
                with patch("CRM.crm._sync_memory_entry", new_callable=AsyncMock):
                    data = await get_user_info("new_async")
                    assert data["name"] == "NewUser"


# ---------------------------------------------------------------------------
# Data Type Edge Cases
# ---------------------------------------------------------------------------

class TestCRMDataTypes:
    """Tests for various data types stored in CRM."""

    @pytest.mark.asyncio
    async def test_numeric_value(self, isolated_crm_db):
        with patch("CRM.crm.DB_PATH", isolated_crm_db):
            from CRM.crm import _create_user_sync, _get_user_info_sync
            _create_user_sync("type_user", {"marla": 10})
            data = _get_user_info_sync("type_user")
            assert data["marla"] == 10

    @pytest.mark.asyncio
    async def test_boolean_value(self, isolated_crm_db):
        with patch("CRM.crm.DB_PATH", isolated_crm_db):
            from CRM.crm import _create_user_sync, _get_user_info_sync
            _create_user_sync("bool_user", {"interested": True})
            data = _get_user_info_sync("bool_user")
            assert data["interested"] is True

    @pytest.mark.asyncio
    async def test_nested_dict_value(self, isolated_crm_db):
        with patch("CRM.crm.DB_PATH", isolated_crm_db):
            from CRM.crm import _create_user_sync, _get_user_info_sync
            _create_user_sync("nested_user", {"preferences": {"area": "DHA", "size": "10 marla"}})
            data = _get_user_info_sync("nested_user")
            assert data["preferences"]["area"] == "DHA"

    @pytest.mark.asyncio
    async def test_empty_data(self, isolated_crm_db):
        with patch("CRM.crm.DB_PATH", isolated_crm_db):
            from CRM.crm import _create_user_sync, _get_user_info_sync
            _create_user_sync("empty_user", {})
            data = _get_user_info_sync("empty_user")
            assert data == {}
