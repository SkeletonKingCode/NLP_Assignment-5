# Ali Real Estate Chatbot — Evaluation Report

**Generated:** 2026-05-07 21:40:27
**Platform:** Linux-6.17.0-23-generic-x86_64-with-glibc2.39

## 1. Hardware Configuration

| Spec | Value |
|------|-------|
| CPU | Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz |
| CPU Cores | 8 |
| RAM | 31.2 GB |
| GPU | NVIDIA GeForce GTX 1050 (4096 MiB, driver 580.142) |
| Disk (Used/Total) | 81.30000000000001/467.3 GB |
| Disk (Free/Total) | 386.0/467.3 GB |
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

## 3. Unit & Component Test Results

| Test Module / Class | Total | Passed | Failed | Errors | Skipped |
|---------------------|-------|--------|--------|--------|---------|
| tests.test_api::TestHealthEndpoint | 4 | 4 | 0 | 0 | 0 |
| tests.test_api::TestRootEndpoint | 2 | 2 | 0 | 0 | 0 |
| tests.test_api::TestSessionEndpoints | 7 | 7 | 0 | 0 | 0 |
| tests.test_api::TestSynthEndpoint | 2 | 1 | 1 | 0 | 0 |
| tests.test_calculator::TestCalculatorErrors | 8 | 8 | 0 | 0 | 0 |
| tests.test_calculator::TestCalculatorFunctional | 10 | 10 | 0 | 0 | 0 |
| tests.test_calculator::TestEvaluateExpression | 5 | 5 | 0 | 0 | 0 |
| tests.test_calendar::TestCalendarCRUD | 7 | 7 | 0 | 0 | 0 |
| tests.test_calendar::TestCalendarEdgeCases | 4 | 4 | 0 | 0 | 0 |
| tests.test_conversation::TestContextWindow | 5 | 5 | 0 | 0 | 0 |
| tests.test_conversation::TestInventory | 6 | 6 | 0 | 0 | 0 |
| tests.test_conversation::TestSessionCRUD | 8 | 8 | 0 | 0 | 0 |
| tests.test_conversation::TestStageAdvancement | 13 | 13 | 0 | 0 | 0 |
| tests.test_conversation::TestSystemPrompt | 8 | 8 | 0 | 0 | 0 |
| tests.test_conversational_correctness::TestDeterministicStageTransitions | 4 | 4 | 0 | 0 | 0 |
| tests.test_crm::TestCRMAsync | 3 | 3 | 0 | 0 | 0 |
| tests.test_crm::TestCRMCrud | 6 | 6 | 0 | 0 | 0 |
| tests.test_crm::TestCRMDataTypes | 4 | 4 | 0 | 0 | 0 |
| tests.test_crm::TestCRMPersistence | 2 | 2 | 0 | 0 | 0 |
| tests.test_orchestrator::TestToolCaching | 2 | 2 | 0 | 0 | 0 |
| tests.test_orchestrator::TestToolCallParsing | 9 | 9 | 0 | 0 | 0 |
| tests.test_orchestrator::TestToolExecution | 5 | 5 | 0 | 0 | 0 |
| tests.test_orchestrator::TestToolRegistration | 3 | 3 | 0 | 0 | 0 |
| tests.test_rag::TestRAGContextRelevance | 1 | 1 | 0 | 0 | 0 |
| tests.test_rag::TestRAGFaithfulness | 1 | 1 | 0 | 0 | 0 |
| tests.test_rag::TestRAGRetrievalMetrics | 3 | 3 | 0 | 0 | 0 |
| tests.test_rag::TestSemanticMatch | 3 | 3 | 0 | 0 | 0 |
| tests.test_weather::TestWeatherFunctional | 4 | 4 | 0 | 0 | 0 |
| tests.test_weather::TestWeatherReturnType | 2 | 2 | 0 | 0 | 0 |
| tests.test_weather::TestWeatherSyncErrors | 4 | 4 | 0 | 0 | 0 |

**Summary:** 144 passed, 1 failed, 0 errors, 0 skipped out of 145 total tests.

### Test Failures

- `tests.test_api.TestSynthEndpoint.test_synth_tts_unavailable`: assert 500 == 503
 +  where 500 = <Response [500 Internal Server Error]>.status_code

## 4. Component-Level Correctness (Narrative)

### 4.1 Calculator Tool
- **Functional tests:** Arithmetic operations, error handling, code injection prevention
- **Edge cases:** Division by zero, large exponents, empty input, special characters

### 4.2 Weather Tool
- **Functional tests:** Valid/invalid cities, mocked HTTP responses
- **Error handling:** Timeouts, network errors, unknown locations

### 4.3 Calendar Tool
- **CRUD tests:** Add/retrieve events, date filtering, ordering
- **Edge cases:** Missing descriptions, kwargs fallback, empty results

### 4.4 CRM Tool
- **CRUD tests:** Create, read, update user records
- **Persistence:** Data survives across operations
- **Data types:** Numeric, boolean, nested dict, empty data

### 4.5 Tool Orchestrator
- **Registration:** Tool registration and system instructions
- **JSON Parsing:** Single/multiple tool calls, surrounding text, invalid JSON
- **Execution:** Valid/invalid tools, argument filtering, caching

## 5. RAG Evaluation

| Metric | Score | Queries |
|--------|-------|---------|
| Precision@3 | 0.6333 | 30 |
| Recall@3 | 0.8333 | 30 |
| Context Relevance | 0.6333 | 30 |
| Faithfulness (keyword) | 0.7222 | 30 |

## 6. Overall Conversational Correctness

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

## 7. Performance — Latency

| Scenario | Trials | TTFT Mean | TTFT Median | TTFT P90 | E2E Mean | E2E Median | E2E P90 |
|----------|--------|-----------|-------------|----------|----------|------------|---------|
| Simple Dialogue | 9 | 4.831s | 2.066s | 26.222s | 8.342s | 5.757s | 33.069s |
| RAG Only | 9 | 2.380s | 2.508s | 2.603s | 17.079s | 18.763s | 25.021s |
| Tool Only | 9 | 4.052s | 2.188s | 11.728s | 14.071s | 13.234s | 22.120s |
| Mixed (RAG + Tool) | 30 | 2.315s | 2.418s | 2.448s | 17.051s | 16.502s | 29.646s |

## 8. Performance — Throughput

- **Max Sustainable Concurrency:** 0 users
- **Breakpoint:** 1 users
- **Thresholds:** TTFT < 2.0s, E2E < 10.0s

| Users | Turns | Time (s) | Turns/sec | Errors | Med TTFT | Med E2E | Within |
|-------|-------|----------|-----------|--------|----------|---------|--------|
| 1 | 3 | 30.5 | 0.10 | 0 || 3.136s | 11.657s |
| 2 | 6 | 71.4 | 0.08 | 0 || 11.968s | 19.125s |
| 3 | 9 | 125.1 | 0.07 | 0 || 19.242s | 27.125s |
| 5 | 15 | 160.8 | 0.09 | 0 || 41.413s | 48.373s |
| 8 | 24 | 260.0 | 0.09 | 0 || 68.626s | 74.021s |
| 10 | 30 | 325.9 | 0.09 | 0 || 79.290s | 83.929s |

## 9. Detailed Test Failures

See the JUnit XML report for full details: [`junit_results.xml`](./junit_results.xml)

## 10. Analysis and Insights

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
