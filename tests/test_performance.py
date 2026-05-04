"""
tests/test_performance.py

Performance evaluation tests — measures latency metrics for the chatbot.

Metrics:
  - Time to First Token (TTFT)
  - Inter-Token Latency (ITL)
  - End-to-End Response Time (E2E)

Scenarios:
  (a) Simple dialogue (no RAG, no tool)
  (b) RAG-only (retrieval required)
  (c) Tool-only (single tool call)
  (d) Mixed (RAG + tool)

Each scenario runs 30+ trials. Reports mean, median, p90, p99.

IMPORTANT: These tests require the chatbot server to be running on localhost:8000
and Ollama to be available. They are skipped if the server is not reachable.
"""

import sys
import os
import json
import time
import asyncio
import statistics
from pathlib import Path
from typing import Optional

import pytest

try:
    import websockets
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WS_URL = os.environ.get("CHATBOT_WS_URL", "ws://localhost:8000/ws/chat")
BASE_URL = os.environ.get("CHATBOT_BASE_URL", "http://localhost:8000")
NUM_TRIALS = int(os.environ.get("PERF_NUM_TRIALS", "30"))
RESULTS_DIR = Path(__file__).resolve().parent.parent / "eval_results"


# ---------------------------------------------------------------------------
# Helper: check server availability
# ---------------------------------------------------------------------------

def _server_available() -> bool:
    """Check if the chatbot server is reachable."""
    try:
        import urllib.request
        req = urllib.request.Request(f"{BASE_URL}/health", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


def _create_session() -> Optional[str]:
    """Create a session via REST API."""
    try:
        import urllib.request
        req = urllib.request.Request(
            f"{BASE_URL}/session",
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            return data.get("session_id")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Latency measurement via WebSocket
# ---------------------------------------------------------------------------

async def _measure_single_turn(ws_url: str, session_id: str, message: str) -> dict:
    """
    Send a single message over WebSocket and measure latency metrics.
    
    Returns dict with:
      - ttft: Time to First Token (seconds)
      - e2e: End-to-End response time (seconds)
      - itl_avg: Average inter-token latency (seconds)
      - itl_values: List of all inter-token latencies
      - token_count: Number of tokens received
      - response: Full response text
    """
    async with websockets.connect(ws_url, ping_interval=None, close_timeout=60) as ws:
        payload = json.dumps({
            "session_id": session_id,
            "message": message,
            "voice": False,
        })

        send_time = time.perf_counter()
        await ws.send(payload)

        first_token_time = None
        last_token_time = None
        token_times = []
        tokens = []
        done = False

        while not done:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=120)
                recv_time = time.perf_counter()
                msg = json.loads(raw)

                if msg["type"] == "token":
                    if first_token_time is None:
                        first_token_time = recv_time
                    token_times.append(recv_time)
                    tokens.append(msg["data"])
                    last_token_time = recv_time

                elif msg["type"] == "done":
                    done = True

                elif msg["type"] == "error":
                    done = True

                elif msg["type"] == "session_created":
                    session_id = msg["data"]

            except asyncio.TimeoutError:
                done = True

        # Compute metrics
        ttft = (first_token_time - send_time) if first_token_time else None
        e2e = (last_token_time - send_time) if last_token_time else None

        # Inter-token latency
        itl_values = []
        if len(token_times) > 1:
            for i in range(1, len(token_times)):
                itl_values.append(token_times[i] - token_times[i - 1])

        itl_avg = statistics.mean(itl_values) if itl_values else None

        return {
            "ttft": ttft,
            "e2e": e2e,
            "itl_avg": itl_avg,
            "itl_values": itl_values,
            "token_count": len(tokens),
            "response": "".join(tokens),
        }


def _compute_stats(values: list) -> dict:
    """Compute mean, median, p90, p99 from a list of values."""
    if not values:
        return {"mean": None, "median": None, "p90": None, "p99": None}
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    return {
        "mean": statistics.mean(sorted_vals),
        "median": statistics.median(sorted_vals),
        "p90": sorted_vals[int(n * 0.90)] if n > 1 else sorted_vals[0],
        "p99": sorted_vals[int(n * 0.99)] if n > 1 else sorted_vals[0],
    }


# ---------------------------------------------------------------------------
# Scenario Definitions
# ---------------------------------------------------------------------------

SCENARIOS = {
    "simple_dialogue": {
        "description": "Simple dialogue — no RAG, no tool",
        "messages": [
            "Hello!",
            "How are you?",
            "Thank you!",
        ],
    },
    "rag_only": {
        "description": "RAG-only — retrieval required",
        "messages": [
            "Tell me about DHA Phase 6 residential features.",
            "What taxes apply when buying property in Pakistan?",
            "What amenities does Bahria Town Karachi offer?",
        ],
    },
    "tool_only": {
        "description": "Tool-only — single tool call",
        "messages": [
            "What is the weather in Lahore right now?",
            "Calculate (150 * 5) / 2 for me.",
            "What is 2 percent of 4.2 crore?",
        ],
    },
    "mixed": {
        "description": "Mixed — RAG + tool",
        "messages": [
            "Tell me about DHA Phase 1 plots, and what's the weather in Lahore?",
            "What taxes apply to property? Calculate 3% of 8.5 crore.",
            "Tell me about Bahria Town and schedule a visit for 2026-06-01.",
        ],
    },
}


# ---------------------------------------------------------------------------
# Performance Test Class
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_WEBSOCKETS, reason="websockets library not installed")
class TestPerformanceLatency:
    """
    Measures TTFT, ITL, and E2E latency across 4 scenarios.
    Requires the chatbot server to be running.
    """

    @pytest.fixture(autouse=True)
    def check_server(self):
        if not _server_available():
            pytest.skip("Chatbot server not available at " + BASE_URL)

    @pytest.mark.asyncio
    async def test_latency_simple_dialogue(self):
        """Scenario (a): Simple dialogue — no RAG, no tool."""
        results = await self._run_scenario("simple_dialogue")
        self._print_report("Simple Dialogue", results)
        assert len(results["ttft_values"]) > 0

    @pytest.mark.asyncio
    async def test_latency_rag_only(self):
        """Scenario (b): RAG-only — retrieval required."""
        results = await self._run_scenario("rag_only")
        self._print_report("RAG Only", results)
        assert len(results["ttft_values"]) > 0

    @pytest.mark.asyncio
    async def test_latency_tool_only(self):
        """Scenario (c): Tool-only — single tool call."""
        results = await self._run_scenario("tool_only")
        self._print_report("Tool Only", results)
        assert len(results["ttft_values"]) > 0

    @pytest.mark.asyncio
    async def test_latency_mixed(self):
        """Scenario (d): Mixed — RAG + tool."""
        results = await self._run_scenario("mixed")
        self._print_report("Mixed (RAG + Tool)", results)
        assert len(results["ttft_values"]) > 0

    async def _run_scenario(self, scenario_name: str) -> dict:
        """Run a scenario for NUM_TRIALS and collect metrics."""
        scenario = SCENARIOS[scenario_name]
        messages = scenario["messages"]

        ttft_values = []
        e2e_values = []
        itl_all = []

        trials_per_message = max(1, NUM_TRIALS // len(messages))

        for msg in messages:
            for _ in range(trials_per_message):
                session_id = _create_session()
                if not session_id:
                    continue

                try:
                    metrics = await _measure_single_turn(WS_URL, session_id, msg)
                    if metrics["ttft"] is not None:
                        ttft_values.append(metrics["ttft"])
                    if metrics["e2e"] is not None:
                        e2e_values.append(metrics["e2e"])
                    if metrics["itl_values"]:
                        itl_all.extend(metrics["itl_values"])
                except Exception as e:
                    print(f"[WARN] Trial failed: {e}")

        return {
            "ttft_values": ttft_values,
            "e2e_values": e2e_values,
            "itl_values": itl_all,
            "ttft_stats": _compute_stats(ttft_values),
            "e2e_stats": _compute_stats(e2e_values),
            "itl_stats": _compute_stats(itl_all),
        }

    def _print_report(self, name: str, results: dict):
        """Print a formatted latency report."""
        print(f"\n{'='*60}")
        print(f"LATENCY REPORT: {name}")
        print(f"{'='*60}")
        for metric_name, key in [
            ("Time to First Token (TTFT)", "ttft_stats"),
            ("End-to-End Response Time", "e2e_stats"),
            ("Inter-Token Latency", "itl_stats"),
        ]:
            stats = results[key]
            if stats["mean"] is not None:
                print(f"\n  {metric_name}:")
                print(f"    Mean:   {stats['mean']:.4f}s")
                print(f"    Median: {stats['median']:.4f}s")
                print(f"    P90:    {stats['p90']:.4f}s")
                print(f"    P99:    {stats['p99']:.4f}s")
            else:
                print(f"\n  {metric_name}: No data")
        print(f"  Trials: {len(results['ttft_values'])}")
        print(f"{'='*60}")

        # Save results to file
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        out_path = RESULTS_DIR / f"latency_{safe_name}.json"
        with open(out_path, "w") as f:
            serializable = {
                "scenario": name,
                "trials": len(results["ttft_values"]),
                "ttft": results["ttft_stats"],
                "e2e": results["e2e_stats"],
                "itl": results["itl_stats"],
            }
            json.dump(serializable, f, indent=2)
