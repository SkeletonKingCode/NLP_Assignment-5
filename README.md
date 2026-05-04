# Ali Real Estate — Evaluation Suite for Autonomous Agentic Chatbot

## Group Members
*   **Name:** Wajeeha Mahmood | **ID:** 23i-0105
*   **Name:** Muhammad Alyun Shah | **ID:** 23i-0022
*   **Name:** Awais Ali | **ID:** 23i-0080

---

## Overview

This repository contains a comprehensive, automated evaluation suite for the **Ali Real Estate Chatbot** (Assignment 4). The suite systematically measures correctness, performance, and scalability across all system components.

### What is Evaluated

| Dimension | Components | Metrics |
|-----------|-----------|---------|
| **Conversational Correctness** | Full dialogues | Task completion, policy adherence, coherence |
| **RAG Component** | Retrieval pipeline | Precision@k, Recall@k, context relevance, faithfulness |
| **CRM Tool** | CRUD operations | Data correctness, persistence, semantic matching |
| **Calculator Tool** | Math evaluation | Functional correctness, error handling, security |
| **Weather Tool** | API integration | Valid/invalid inputs, timeout handling |
| **Calendar Tool** | Event management | CRUD, date filtering, ordering |
| **Tool Orchestrator** | JSON parsing | Tool call extraction, execution, caching |
| **Latency** | WebSocket streaming | TTFT, inter-token latency, E2E |
| **Throughput** | Concurrency | Max users, breakpoint, turns/sec |

---

## Quick Start

### Prerequisites

1. Python 3.10+
2. The chatbot backend from Assignment 4
3. Ollama with the `ali-realestate` model (for live tests)

### Installation

```bash
# Clone and set up
cd NLP_Assignment-5
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Running the Evaluation Suite

```bash
# Run all unit/component tests (NO server needed)
python run_evals.py --unit

# Run full suite (server + Ollama REQUIRED)
python run_evals.py --all

# Run only performance benchmarks
python run_evals.py --perf

# Run LLM-as-judge evaluation
python run_evals.py --judge

# Generate report from existing results
python run_evals.py --report-only

# Or use pytest directly
pytest tests/ -v                          # All tests
pytest tests/test_calculator.py -v        # Single component
pytest tests/ -k "not TestPerformance" -v # Skip perf tests
```

### Configuration (Environment Variables)

| Variable | Default | Description |
|----------|---------|-------------|
| `CHATBOT_BASE_URL` | `http://localhost:8000` | Chatbot REST API base URL |
| `CHATBOT_WS_URL` | `ws://localhost:8000/ws/chat` | WebSocket endpoint URL |
| `JUDGE_MODEL` | `ali-realestate` | Ollama model for LLM-as-judge |
| `PERF_NUM_TRIALS` | `30` | Number of trials per latency scenario |
| `MAX_TTFT` | `2.0` | Acceptable TTFT threshold (seconds) |
| `MAX_E2E` | `10.0` | Acceptable E2E threshold (seconds) |

---

## Test Suite Structure

```text
tests/
├── conftest.py                        # Shared fixtures, dependency stubs
├── test_data/
│   ├── test_conversations.json        # 13 multi-turn dialogue test cases
│   ├── rag_ground_truth.json          # 30 annotated RAG queries
│   └── tool_invocation_test_set.json  # Tool call accuracy test sets
├── test_calculator.py                 # Calculator: 18 tests
├── test_weather.py                    # Weather: 11 tests
├── test_calendar.py                   # Calendar: 11 tests
├── test_crm.py                        # CRM: 13 tests
├── test_orchestrator.py               # Orchestrator: 18 tests
├── test_conversation.py               # Conversation manager: 22 tests
├── test_api.py                        # FastAPI endpoints: 13 tests
├── test_rag.py                        # RAG metrics: 7 tests
├── test_conversational_correctness.py # E2E dialogue validation: 5 tests
├── test_performance.py                # Latency benchmarks: 4 scenarios
├── test_throughput.py                 # Concurrency sweep: 6 levels
└── test_llm_judge.py                  # LLM judge: 4 evaluation dimensions

run_evals.py                           # Master runner + report generator
pyproject.toml                         # Pytest configuration
eval_results/                          # Generated results directory
└── evaluation_report.md               # Auto-generated report
```

---

## How Metrics Are Computed

### RAG Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| **Precision@k** | `relevant_in_top_k / k` | Fraction of retrieved chunks that are relevant |
| **Recall@k** | `relevant_in_top_k / total_relevant` | Fraction of relevant docs successfully retrieved |
| **Context Relevance** | `useful_chunks / total_chunks` | Proportion of chunks containing expected keywords |
| **Faithfulness** | `matched_keywords / expected_keywords` | Keyword overlap between context and ground truth |

Relevance is determined by checking if retrieved chunks contain content from annotated source documents or expected keywords from `rag_ground_truth.json`.

### Latency Metrics

| Metric | Description |
|--------|-------------|
| **TTFT** | Time from sending message to receiving first token (via WebSocket) |
| **ITL** | Average time between consecutive tokens |
| **E2E** | Time from sending message to receiving last token |

Each metric reports **mean, median, P90, P99** across 30+ trials per scenario.

### Throughput Metrics

| Metric | Description |
|--------|-------------|
| **Max Sustainable Concurrency** | Highest concurrent users where median TTFT < 2s AND median E2E < 10s |
| **Breakpoint** | Concurrency level where latency exceeds thresholds |
| **Turns/sec** | Total turns completed per second at sustainable concurrency |

### LLM-as-Judge Metrics

The judge LLM evaluates using structured rubrics and scores on a 0–1 scale:

| Metric | What It Measures |
|--------|-----------------|
| **Task Completion** | Did the bot fulfil the user's request? |
| **Policy Adherence** | Did the bot refuse off-topic requests appropriately? |
| **Coherence** | Does the bot remember context and avoid contradictions? |
| **Faithfulness** | Is the answer grounded in retrieved documents? |

---

## Interpreting Results

### What is "Good" Performance?

| Metric | Good | Acceptable | Poor |
|--------|------|------------|------|
| Precision@3 | > 0.7 | 0.4–0.7 | < 0.4 |
| Recall@3 | > 0.6 | 0.3–0.6 | < 0.3 |
| Context Relevance | > 0.7 | 0.4–0.7 | < 0.4 |
| Faithfulness | > 0.8 | 0.5–0.8 | < 0.5 |
| Task Completion (judge) | > 0.8 | 0.5–0.8 | < 0.5 |
| TTFT (simple) | < 1.0s | 1.0–2.0s | > 2.0s |
| E2E (simple) | < 5.0s | 5.0–10.0s | > 10.0s |
| Sustainable concurrency | > 5 users | 2–5 users | < 2 users |

---

## Assumptions and Limitations

1. **Model limitations:** The `ali-realestate` model (2B parameters) may hallucinate JSON syntax in long prompts, affecting tool call accuracy.
2. **Hardware dependency:** All performance metrics are hardware-specific. Results will vary on different machines.
3. **External APIs:** Weather tool tests depend on `wttr.in` availability. Mocked in unit tests.
4. **LLM Judge bias:** The judge uses the same model family; ideally a larger/different model should be used. Validation against human judgment recommended.
5. **RAG index required:** RAG evaluation tests require the index to be pre-built via `python backend/RAG/indexer.py`.
6. **Single LLM thread:** Throughput is bottlenecked by Ollama's sequential inference. Concurrent tests measure queueing behavior, not true parallel inference.

---

## Changes from Assignment 4

No changes were made to the core chatbot system. The evaluation suite is a pure addition:
- All test files are in `tests/`
- Runner script at `run_evals.py`
- Results output to `eval_results/`
- Dependencies added: `pytest`, `pytest-asyncio`, `websockets`

---

## Video Demo

**[YouTube Link]** — [Youtube](https://youtu.be)

The video demonstrates:
1. Running the automated evaluation suite
2. Key results from component tests
3. Performance latency measurements
4. One interesting finding: [describe your finding]
