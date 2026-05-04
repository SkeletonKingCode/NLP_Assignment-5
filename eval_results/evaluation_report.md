# Ali Real Estate Chatbot — Evaluation Report

**Generated:** 2026-05-04 23:51:44
**Platform:** Linux-6.17.0-23-generic-x86_64-with-glibc2.39

## 1. Hardware Configuration

| Spec | Value |
|------|-------|
| CPU | Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz |
| CPU Cores | 8 |
| RAM | 31.2 GB |
| GPU | NVIDIA GeForce GTX 1050 (4096 MiB, driver 580.142) |
| Disk (Total/Free) | 467.3/386.8 GB |
| Python | 3.11.14 |

## 2. Dependency Versions

| Package | Version |
|---------|---------|
| fastapi | 0.136.1 |
| uvicorn | 0.46.0 |
| pydantic | 2.13.3 |
| chromadb | 1.5.8 |
| sentence_transformers | 5.4.1 |
| pytest | 9.0.3 |
| websockets | 16.0 |
| ollama | installed |

## 3. Component-Level Correctness

### 3.1 Calculator Tool
- **Functional tests:** Arithmetic operations, error handling, code injection prevention
- **Edge cases:** Division by zero, large exponents, empty input, special characters

### 3.2 Weather Tool
- **Functional tests:** Valid/invalid cities, mocked HTTP responses
- **Error handling:** Timeouts, network errors, unknown locations

### 3.3 Calendar Tool
- **CRUD tests:** Add/retrieve events, date filtering, ordering
- **Edge cases:** Missing descriptions, kwargs fallback, empty results

### 3.4 CRM Tool
- **CRUD tests:** Create, read, update user records
- **Persistence:** Data survives across operations
- **Data types:** Numeric, boolean, nested dict, empty data

### 3.5 Tool Orchestrator
- **Registration:** Tool registration and system instructions
- **JSON Parsing:** Single/multiple tool calls, surrounding text, invalid JSON
- **Execution:** Valid/invalid tools, argument filtering, caching

## 4. RAG Evaluation

| Metric | Score | Queries |
|--------|-------|---------|
| Precision@3 | 0.6333 | 30 |
| Recall@3 | 0.8333 | 30 |
| Context Relevance | 0.6333 | 30 |
| Faithfulness (keyword) | 0.7222 | 30 |

## 5. Overall Conversational Correctness

### Stage Transition Accuracy
- **Passed:** 11/13 dialogues
- **Accuracy:** 84.6%

### LLM-as-Judge Evaluation

| Metric | Average Score | Dialogues |
|--------|---------------|-----------|
| Task Completion | 0.0000 | 0 |
| Policy Adherence | 0.0000 | 0 |
| Coherence | 0.0000 | 0 |
| Faithfulness | 0.0000 | 0 |

## 6. Performance — Latency

| Scenario | Trials | TTFT Mean | TTFT Median | TTFT P90 | E2E Mean | E2E Median | E2E P90 |
|----------|--------|-----------|-------------|----------|----------|------------|---------|
| Simple Dialogue | 30 | 2.642s | 2.082s | 2.516s | 7.241s | 6.237s | 13.001s |
| RAG Only | — | — | — | — | — | — | — |
| Tool Only | — | — | — | — | — | — | — |
| Mixed (RAG + Tool) | — | — | — | — | — | — | — |

## 7. Performance — Throughput

*(Run throughput tests to populate)*

## 8. Test Failures and Errors

See detailed results in `eval_results/junit_results.xml`

## 9. Analysis and Insights

### Key Findings

1. **Stage Machine Reliability:** The deterministic stage machine reliably
   advances through greeting → category → subtype → closing based on keyword
   matching with semantic fallback for typo tolerance.

2. **Tool Orchestration:** The JSON parser successfully extracts tool calls
   from mixed LLM output. The argument safety filter prevents unexpected
   parameters from reaching tool functions.

3. **RAG Retrieval:** With 52 documents indexed, retrieval relevance depends
   on query specificity. Queries mentioning specific project names (DHA, Bahria)
   achieve higher precision than generic queries.

4. **CRM Persistence:** SQLite-backed CRM correctly persists data across
   operations. Semantic field matching via ChromaDB enables typo tolerance.

5. **Calculator Security:** AST-based evaluation prevents code injection
   while supporting all standard mathematical operations.

### Known Limitations

1. The 2B LLM model may occasionally hallucinate JSON syntax in long prompts.
2. Throughput is limited by single-threaded LLM inference (Ollama).
3. RAG faithfulness drops when queries are ambiguous or span multiple documents.
4. Weather tool depends on external wttr.in API availability.
