"""
tests/test_throughput.py

Throughput / Concurrency evaluation for the chatbot.

Simulates increasing numbers of concurrent WebSocket sessions,
each sending a fixed set of messages. Determines:
  - Maximum sustainable concurrency
  - Breakpoint where latency degrades sharply
  - Turns per second at sustainable concurrency

IMPORTANT: Requires the chatbot server to be running on localhost:8000.
"""

import sys
import os
import json
import time
import asyncio
import statistics
from pathlib import Path
from typing import List, Dict

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
RESULTS_DIR = Path(__file__).resolve().parent.parent / "eval_results"

# Acceptable latency thresholds
MAX_TTFT_SECONDS = float(os.environ.get("MAX_TTFT", "2.0"))
MAX_E2E_SECONDS = float(os.environ.get("MAX_E2E", "10.0"))

# Concurrency levels to test
CONCURRENCY_LEVELS = [1, 2, 3, 5, 8, 10]

# Messages each simulated user sends
USER_MESSAGES = [
    "Hello!",
    "I want to see houses.",
    "Tell me about the 10 marla option.",
]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _server_available() -> bool:
    try:
        import urllib.request
        req = urllib.request.Request(f"{BASE_URL}/health", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


def _create_session():
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


async def _simulate_user(ws_url: str, user_id: int, messages: List[str]) -> Dict:
    """
    Simulate a single user sending multiple messages over WebSocket.
    Returns metrics for all turns.
    """
    session_id = await asyncio.get_event_loop().run_in_executor(None, _create_session)
    if not session_id:
        return {"user_id": user_id, "error": "Could not create session", "turns": []}

    turn_metrics = []
    
    try:
        async with websockets.connect(ws_url, ping_interval=None, close_timeout=120) as ws:
            for msg in messages:
                payload = json.dumps({
                    "session_id": session_id,
                    "message": msg,
                    "voice": False,
                })

                send_time = time.perf_counter()
                await ws.send(payload)

                first_token_time = None
                last_token_time = None
                token_count = 0
                done = False

                while not done:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=120)
                        recv_time = time.perf_counter()
                        data = json.loads(raw)

                        if data["type"] == "token":
                            if first_token_time is None:
                                first_token_time = recv_time
                            last_token_time = recv_time
                            token_count += 1
                        elif data["type"] in ("done", "error"):
                            done = True
                        elif data["type"] == "session_created":
                            session_id = data["data"]
                    except asyncio.TimeoutError:
                        done = True

                ttft = (first_token_time - send_time) if first_token_time else None
                e2e = (last_token_time - send_time) if last_token_time else None

                turn_metrics.append({
                    "message": msg,
                    "ttft": ttft,
                    "e2e": e2e,
                    "token_count": token_count,
                })

    except Exception as e:
        return {"user_id": user_id, "error": str(e), "turns": turn_metrics}

    return {"user_id": user_id, "error": None, "turns": turn_metrics}


async def _run_concurrency_level(ws_url: str, num_users: int, messages: List[str]) -> Dict:
    """
    Run num_users concurrent WebSocket sessions and collect aggregate metrics.
    """
    start_time = time.perf_counter()

    tasks = [
        _simulate_user(ws_url, i, messages)
        for i in range(num_users)
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    total_time = time.perf_counter() - start_time

    # Aggregate metrics
    all_ttft = []
    all_e2e = []
    total_turns = 0
    errors = 0

    for r in results:
        if isinstance(r, Exception):
            errors += 1
            continue
        if r.get("error"):
            errors += 1
        for turn in r.get("turns", []):
            total_turns += 1
            if turn["ttft"] is not None:
                all_ttft.append(turn["ttft"])
            if turn["e2e"] is not None:
                all_e2e.append(turn["e2e"])

    ttft_stats = _compute_stats(all_ttft)
    e2e_stats = _compute_stats(all_e2e)

    turns_per_second = total_turns / total_time if total_time > 0 else 0

    return {
        "num_users": num_users,
        "total_turns": total_turns,
        "total_time": total_time,
        "turns_per_second": turns_per_second,
        "errors": errors,
        "ttft": ttft_stats,
        "e2e": e2e_stats,
        "within_threshold": (
            (ttft_stats["median"] or 0) <= MAX_TTFT_SECONDS and
            (e2e_stats["median"] or 0) <= MAX_E2E_SECONDS
        ),
    }


def _compute_stats(values: list) -> dict:
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
# Throughput Test Class
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_WEBSOCKETS, reason="websockets library not installed")
class TestThroughputConcurrency:
    """
    Determines maximum sustainable concurrency and breakpoint.
    """

    @pytest.fixture(autouse=True)
    def check_server(self):
        if not _server_available():
            pytest.skip("Chatbot server not available at " + BASE_URL)

    @pytest.mark.asyncio
    async def test_concurrency_sweep(self):
        """
        Run increasing concurrency levels and determine:
        - Maximum sustainable concurrency
        - Breakpoint where latency degrades
        """
        all_results = []
        max_sustainable = 0
        breakpoint_level = None

        print(f"\n{'='*70}")
        print("THROUGHPUT / CONCURRENCY EVALUATION")
        print(f"Thresholds: TTFT < {MAX_TTFT_SECONDS}s, E2E < {MAX_E2E_SECONDS}s")
        print(f"{'='*70}")

        for level in CONCURRENCY_LEVELS:
            print(f"\n--- Testing {level} concurrent users ---")
            result = await _run_concurrency_level(WS_URL, level, USER_MESSAGES)
            all_results.append(result)

            within = result["within_threshold"]
            ttft_med = result["ttft"]["median"]
            e2e_med = result["e2e"]["median"]

            print(f"  Users:           {level}")
            print(f"  Total turns:     {result['total_turns']}")
            print(f"  Total time:      {result['total_time']:.2f}s")
            print(f"  Turns/sec:       {result['turns_per_second']:.2f}")
            print(f"  Errors:          {result['errors']}")
            print(f"  Median TTFT:     {ttft_med:.4f}s" if ttft_med else "  Median TTFT:     N/A")
            print(f"  Median E2E:      {e2e_med:.4f}s" if e2e_med else "  Median E2E:      N/A")
            print(f"  Within threshold: {'✓' if within else '✗'}")

            if within:
                max_sustainable = level
            elif breakpoint_level is None:
                breakpoint_level = level

        # Summary
        print(f"\n{'='*70}")
        print("CONCURRENCY SUMMARY")
        print(f"{'='*70}")
        print(f"  Max sustainable concurrency: {max_sustainable} users")
        print(f"  Breakpoint:                  {breakpoint_level or 'Not reached'} users")
        if max_sustainable > 0:
            sustainable_result = next(
                r for r in all_results if r["num_users"] == max_sustainable
            )
            print(f"  Turns/sec at sustainable:    {sustainable_result['turns_per_second']:.2f}")
        print(f"{'='*70}")

        # Save results
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RESULTS_DIR / "throughput_results.json"
        with open(out_path, "w") as f:
            json.dump({
                "thresholds": {
                    "max_ttft_seconds": MAX_TTFT_SECONDS,
                    "max_e2e_seconds": MAX_E2E_SECONDS,
                },
                "max_sustainable_concurrency": max_sustainable,
                "breakpoint": breakpoint_level,
                "levels": all_results,
            }, f, indent=2)

        # The test always passes; it's a measurement
        assert len(all_results) > 0
