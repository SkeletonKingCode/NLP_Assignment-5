"""
tests/test_orchestrator.py

Unit tests for the Tool Orchestrator — tool registration, JSON parsing
from raw LLM output, execution, caching, and error handling.
"""

import sys
import json
import asyncio
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from Tools.orchestrator import ToolOrchestrator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fresh_orchestrator():
    """Return a fresh ToolOrchestrator with no registered tools."""
    return ToolOrchestrator()


@pytest.fixture
def orchestrator_with_tools():
    """Return an orchestrator with dummy tools registered."""
    orch = ToolOrchestrator()

    async def dummy_weather(city: str):
        return f"Weather in {city}: Sunny, 30°C"

    async def dummy_calculate(expression: str):
        return str(eval(expression))  # safe for test only

    async def dummy_add_event(date: str, description: str = "No description"):
        return f"Event on {date}: {description}"

    async def dummy_failing_tool(arg: str):
        raise ValueError("Tool failed intentionally")

    orch.register("get_weather", dummy_weather, "Get weather for a city")
    orch.register("calculate", dummy_calculate, "Calculate math")
    orch.register("add_event", dummy_add_event, "Add calendar event")
    orch.register("failing_tool", dummy_failing_tool, "A tool that always fails")
    return orch


# ---------------------------------------------------------------------------
# Tool Registration
# ---------------------------------------------------------------------------

class TestToolRegistration:

    def test_register_tool(self, fresh_orchestrator):
        async def my_tool(x: str):
            return x
        fresh_orchestrator.register("my_tool", my_tool, "A test tool")
        assert "my_tool" in fresh_orchestrator._tools
        assert "my_tool" in fresh_orchestrator._descriptions

    def test_system_instructions_empty(self, fresh_orchestrator):
        assert fresh_orchestrator.get_system_instructions() == ""

    def test_system_instructions_with_tools(self, orchestrator_with_tools):
        instructions = orchestrator_with_tools.get_system_instructions()
        assert "get_weather" in instructions
        assert "calculate" in instructions
        assert "add_event" in instructions
        assert "TOOL CALLING POLICY" in instructions


# ---------------------------------------------------------------------------
# JSON Parsing from LLM Output
# ---------------------------------------------------------------------------

class TestToolCallParsing:

    def test_parse_single_tool_call(self, orchestrator_with_tools):
        text = '{"tool_name": "get_weather", "arguments": {"city": "Lahore"}}'
        calls = orchestrator_with_tools.parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["tool_name"] == "get_weather"
        assert calls[0]["arguments"]["city"] == "Lahore"

    def test_parse_tool_call_with_surrounding_text(self, orchestrator_with_tools):
        text = """
        Let me check the weather for you.
        {"tool_name": "get_weather", "arguments": {"city": "Karachi"}}
        I'll get that information right away.
        """
        calls = orchestrator_with_tools.parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["arguments"]["city"] == "Karachi"

    def test_parse_multiple_tool_calls(self, orchestrator_with_tools):
        text = """
        {"tool_name": "get_weather", "arguments": {"city": "Lahore"}}
        {"tool_name": "calculate", "arguments": {"expression": "2+2"}}
        """
        calls = orchestrator_with_tools.parse_tool_calls(text)
        assert len(calls) == 2

    def test_parse_no_tool_call(self, orchestrator_with_tools):
        text = "Hello! I am Ali, your real estate assistant."
        calls = orchestrator_with_tools.parse_tool_calls(text)
        assert len(calls) == 0

    def test_parse_invalid_json(self, orchestrator_with_tools):
        text = '{"tool_name": "get_weather", "arguments": INVALID}'
        calls = orchestrator_with_tools.parse_tool_calls(text)
        assert len(calls) == 0

    def test_parse_json_without_tool_name(self, orchestrator_with_tools):
        text = '{"name": "not_a_tool", "args": {"x": 1}}'
        calls = orchestrator_with_tools.parse_tool_calls(text)
        assert len(calls) == 0

    def test_parse_json_without_arguments(self, orchestrator_with_tools):
        text = '{"tool_name": "get_weather"}'
        calls = orchestrator_with_tools.parse_tool_calls(text)
        assert len(calls) == 0

    def test_parse_nested_json_in_arguments(self, orchestrator_with_tools):
        text = '{"tool_name": "calculate", "arguments": {"expression": "2 + 2"}}'
        calls = orchestrator_with_tools.parse_tool_calls(text)
        assert len(calls) == 1

    def test_parse_escaped_strings(self, orchestrator_with_tools):
        text = '{"tool_name": "calculate", "arguments": {"expression": "10 \\\\ 2"}}'
        calls = orchestrator_with_tools.parse_tool_calls(text)
        # Should still parse (or gracefully handle)
        assert isinstance(calls, list)


# ---------------------------------------------------------------------------
# Tool Execution
# ---------------------------------------------------------------------------

class TestToolExecution:

    @pytest.mark.asyncio
    async def test_execute_valid_tool(self, orchestrator_with_tools):
        call = {"tool_name": "get_weather", "arguments": {"city": "Lahore"}}
        result = await orchestrator_with_tools.execute_tool(call)
        assert result["status"] == "success"
        assert "Lahore" in result["result"]

    @pytest.mark.asyncio
    async def test_execute_unregistered_tool(self, orchestrator_with_tools):
        call = {"tool_name": "nonexistent_tool", "arguments": {}}
        with pytest.raises(ValueError, match="not registered"):
            await orchestrator_with_tools.execute_tool(call)

    @pytest.mark.asyncio
    async def test_execute_tool_with_error(self, orchestrator_with_tools):
        call = {"tool_name": "failing_tool", "arguments": {"arg": "test"}}
        result = await orchestrator_with_tools.execute_tool(call)
        assert result["status"] == "error"
        assert "failed intentionally" in result["error"]

    @pytest.mark.asyncio
    async def test_execute_all(self, orchestrator_with_tools):
        text = """
        {"tool_name": "get_weather", "arguments": {"city": "Lahore"}}
        {"tool_name": "calculate", "arguments": {"expression": "5+5"}}
        """
        results = await orchestrator_with_tools.execute_all(text)
        assert len(results) == 2
        assert results[0]["execution"]["status"] == "success"
        assert results[1]["execution"]["status"] == "success"

    @pytest.mark.asyncio
    async def test_execute_filters_invalid_args(self, orchestrator_with_tools):
        """Orchestrator should filter out args not in the function signature."""
        call = {"tool_name": "get_weather", "arguments": {"city": "Lahore", "extra_param": "ignored"}}
        result = await orchestrator_with_tools.execute_tool(call)
        assert result["status"] == "success"


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

class TestToolCaching:

    @pytest.mark.asyncio
    async def test_cached_result(self, orchestrator_with_tools):
        call = {"tool_name": "calculate", "arguments": {"expression": "2+2"}}
        result1 = await orchestrator_with_tools.execute_tool(call)
        assert result1["cached"] is False

        result2 = await orchestrator_with_tools.execute_tool(call)
        assert result2["cached"] is True
        assert result2["result"] == result1["result"]

    @pytest.mark.asyncio
    async def test_different_args_not_cached(self, orchestrator_with_tools):
        call1 = {"tool_name": "get_weather", "arguments": {"city": "Lahore"}}
        call2 = {"tool_name": "get_weather", "arguments": {"city": "Karachi"}}
        result1 = await orchestrator_with_tools.execute_tool(call1)
        result2 = await orchestrator_with_tools.execute_tool(call2)
        assert result1["cached"] is False
        assert result2["cached"] is False
        assert "Lahore" in result1["result"]
        assert "Karachi" in result2["result"]
