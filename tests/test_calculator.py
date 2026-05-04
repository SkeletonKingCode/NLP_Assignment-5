"""
tests/test_calculator.py

Unit tests for the Calculator tool — functional correctness with valid
and invalid inputs, error handling, and edge cases.
"""

import sys
import asyncio
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from Tools.calculator import calculate, _evaluate_expression


# ---------------------------------------------------------------------------
# Functional Correctness — Direct Invocation
# ---------------------------------------------------------------------------

class TestCalculatorFunctional:
    """Unit tests that call the calculator tool directly."""

    @pytest.mark.asyncio
    async def test_basic_addition(self):
        result = await calculate("2 + 3")
        assert result == "5"

    @pytest.mark.asyncio
    async def test_basic_subtraction(self):
        result = await calculate("10 - 4")
        assert result == "6"

    @pytest.mark.asyncio
    async def test_multiplication(self):
        result = await calculate("6 * 7")
        assert result == "42"

    @pytest.mark.asyncio
    async def test_division(self):
        result = await calculate("20 / 4")
        assert result == "5.0"

    @pytest.mark.asyncio
    async def test_complex_expression(self):
        result = await calculate("10 + 5 * 2")
        assert result == "20"

    @pytest.mark.asyncio
    async def test_parenthetical_expression(self):
        result = await calculate("(100 / 2) ** 2")
        assert result == "2500.0"

    @pytest.mark.asyncio
    async def test_nested_parentheses(self):
        result = await calculate("((2 + 3) * (4 - 1))")
        assert result == "15"

    @pytest.mark.asyncio
    async def test_negative_numbers(self):
        result = await calculate("-5 + 3")
        assert result == "-2"

    @pytest.mark.asyncio
    async def test_float_arithmetic(self):
        result = await calculate("50 * 1.2 / 2")
        assert result == "30.0"

    @pytest.mark.asyncio
    async def test_large_number(self):
        result = await calculate("1000000 * 42")
        assert result == "42000000"


# ---------------------------------------------------------------------------
# Error Handling
# ---------------------------------------------------------------------------

class TestCalculatorErrors:
    """Tests for invalid inputs and error conditions."""

    @pytest.mark.asyncio
    async def test_division_by_zero(self):
        result = await calculate("10 / 0")
        assert "Error" in result or "zero" in result.lower()

    @pytest.mark.asyncio
    async def test_invalid_syntax(self):
        result = await calculate("10 + * 2")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_code_injection_import(self):
        result = await calculate("import os")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_code_injection_exec(self):
        result = await calculate("__import__('os').system('ls')")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_exponent_too_large(self):
        result = await calculate("2 ** 1000")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_empty_expression(self):
        result = await calculate("")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_string_input(self):
        result = await calculate("hello world")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_special_characters(self):
        result = await calculate("@#$%")
        assert "Error" in result


# ---------------------------------------------------------------------------
# Synchronous evaluator tests
# ---------------------------------------------------------------------------

class TestEvaluateExpression:
    """Direct tests on the internal synchronous evaluator."""

    def test_integer_result(self):
        assert _evaluate_expression("2 + 2") == 4

    def test_float_result(self):
        assert _evaluate_expression("7 / 2") == 3.5

    def test_power(self):
        assert _evaluate_expression("3 ** 3") == 27

    def test_unary_negative(self):
        assert _evaluate_expression("-10") == -10

    def test_unary_positive(self):
        assert _evaluate_expression("+5") == 5
