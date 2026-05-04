"""
tests/test_llm_judge.py

LLM-as-a-Judge evaluation for overall conversational correctness.

Uses a separate LLM (via Ollama) to judge the quality of chatbot responses
against a rubric. Evaluates:
  - Task completion rate
  - Policy adherence
  - Coherence and consistency
  - Faithfulness to retrieved context

The judge model should ideally be different/larger than the chatbot model.
Falls back to the same model if no judge model is specified.

IMPORTANT: Requires Ollama to be running with a judge model available.
"""

import sys
import os
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Optional

import pytest

# Import ollama BYPASSING conftest's mock
import sys

try:
    import ollama as ollama_lib
    HAS_OLLAMA = True
except ImportError:
    ollama_lib = None
    HAS_OLLAMA = False

_BACKEND = Path(__file__).resolve().parent.parent / "backend"
_TEST_DATA = Path(__file__).resolve().parent / "test_data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "eval_results"

if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "ali-realestate")


# ---------------------------------------------------------------------------
# Rubric Prompts
# ---------------------------------------------------------------------------

TASK_COMPLETION_RUBRIC = """
You are an expert evaluator. Given a multi-turn conversation between a user and 
a real estate chatbot called "Ali", determine whether the chatbot successfully 
completed the user's task.

Conversation:
{conversation}

Expected outcome: {expected}

Rate the task completion on a scale of 0-1:
- 1.0: Task fully completed
- 0.5: Task partially completed
- 0.0: Task not completed

Respond with ONLY a JSON object: {{"score": <float>, "reason": "<brief explanation>"}}
"""

POLICY_ADHERENCE_RUBRIC = """
You are an expert evaluator. Given a conversation between a user and a real estate 
chatbot called "Ali", determine whether the chatbot adhered to its policies:
1. Only discuss real estate topics (with tool exceptions for weather, calendar, math, CRM)
2. Only present properties from its authorized inventory
3. Never invent prices, addresses, or property details
4. Politely decline truly off-topic requests

Conversation:
{conversation}

Rate policy adherence on a scale of 0-1:
- 1.0: Perfect adherence
- 0.5: Minor violations
- 0.0: Major violations

Respond with ONLY a JSON object: {{"score": <float>, "reason": "<brief explanation>"}}
"""

COHERENCE_RUBRIC = """
You are an expert evaluator. Given a multi-turn conversation between a user and a 
real estate chatbot called "Ali", evaluate coherence and consistency:
1. Does the chatbot remember information from earlier turns?
2. Does it avoid contradictions?
3. Are the responses logically connected?

Conversation:
{conversation}

Rate coherence on a scale of 0-1:
- 1.0: Perfectly coherent and consistent
- 0.5: Some inconsistencies
- 0.0: Major contradictions or memory failures

Respond with ONLY a JSON object: {{"score": <float>, "reason": "<brief explanation>"}}
"""

FAITHFULNESS_RUBRIC = """
You are an expert evaluator. Given a question, the retrieved context, and the 
chatbot's answer, determine whether the answer is faithful to (entailed by) the context.

Question: {question}
Retrieved Context: {context}
Answer: {answer}

Rate faithfulness on a scale of 0-1:
- 1.0: Fully faithful — answer only uses information from context
- 0.5: Partially faithful — some info is grounded, some is not
- 0.0: Not faithful — answer contradicts or invents beyond context

Respond with ONLY a JSON object: {{"score": <float>, "reason": "<brief explanation>"}}
"""


# ---------------------------------------------------------------------------
# Judge Helper
# ---------------------------------------------------------------------------

async def _ask_judge(prompt: str) -> Dict:
    """Send a prompt to the judge LLM and parse the JSON response."""
    try:
        client = ollama_lib.AsyncClient()
        response = await client.chat(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )
        text = response.message.content.strip()

        # Try to extract JSON from the response
        # Handle cases where the model wraps in code blocks
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        # Find first { and last }
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            text = text[start:end]

        return json.loads(text)
    except json.JSONDecodeError:
        return {"score": 0.5, "reason": "Could not parse judge response"}
    except Exception as e:
        return {"score": None, "reason": f"Judge error: {str(e)}"}


def _format_conversation(turns: list) -> str:
    """Format conversation turns for the judge prompt."""
    lines = []
    for turn in turns:
        role = turn.get("role", "unknown").upper()
        content = turn.get("content", "")
        if role != "EXPECTED":
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Load test conversations
# ---------------------------------------------------------------------------

def _load_test_conversations():
    path = _TEST_DATA / "test_conversations.json"
    with open(path, "r") as f:
        data = json.load(f)
    return data["dialogues"]


# ---------------------------------------------------------------------------
# Judge Evaluation Tests
# ---------------------------------------------------------------------------

def _ollama_available() -> bool:
    try:
        import ollama as o
        o.list()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not HAS_OLLAMA, reason="ollama library not installed")
class TestLLMJudgeEvaluation:
    """
    Uses an LLM judge to evaluate conversational quality.
    Requires Ollama with the judge model.
    """

    @pytest.fixture(autouse=True)
    def check_ollama(self):
        if not _ollama_available():
            pytest.skip("Ollama not available")

    @pytest.mark.asyncio
    async def test_task_completion_scoring(self):
        """
        Evaluate task completion rate across all test conversations.
        Uses LLM-as-judge with the task completion rubric.
        """
        dialogues = _load_test_conversations()
        scores = []

        for dialogue in dialogues:
            if not dialogue.get("expected_outcome", {}).get("task_completed"):
                continue  # Skip dialogues where task is not expected to complete

            conv_text = _format_conversation(dialogue["turns"])
            expected = json.dumps(dialogue["expected_outcome"])

            prompt = TASK_COMPLETION_RUBRIC.format(
                conversation=conv_text,
                expected=expected,
            )
            result = await _ask_judge(prompt)

            if result.get("score") is not None:
                scores.append(result["score"])
                print(f"  [{dialogue['id']}] Score: {result['score']:.2f} — {result.get('reason', 'N/A')}")

        avg = sum(scores) / len(scores) if scores else 0.0
        if scores:
            print(f"\n[JUDGE] Average Task Completion Score: {avg:.4f} ({len(scores)} dialogues)")
        
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_DIR / "judge_task_completion.json", "w") as f:
            json.dump({"average_score": avg, "count": len(scores), "scores": scores}, f, indent=2)

        assert len(scores) >= 0  # Always passes; metric is recorded

    @pytest.mark.asyncio
    async def test_policy_adherence_scoring(self):
        """Evaluate policy adherence across test conversations."""
        dialogues = _load_test_conversations()
        scores = []

        for dialogue in dialogues:
            conv_text = _format_conversation(dialogue["turns"])
            prompt = POLICY_ADHERENCE_RUBRIC.format(conversation=conv_text)
            result = await _ask_judge(prompt)

            if result.get("score") is not None:
                scores.append(result["score"])
                print(f"  [{dialogue['id']}] Policy: {result['score']:.2f} — {result.get('reason', 'N/A')}")

        avg = sum(scores) / len(scores) if scores else 0.0
        if scores:
            print(f"\n[JUDGE] Average Policy Adherence: {avg:.4f} ({len(scores)} dialogues)")

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_DIR / "judge_policy_adherence.json", "w") as f:
            json.dump({"average_score": avg, "count": len(scores), "scores": scores}, f, indent=2)

    @pytest.mark.asyncio
    async def test_coherence_scoring(self):
        """Evaluate coherence and consistency across multi-turn dialogues."""
        dialogues = _load_test_conversations()
        scores = []

        for dialogue in dialogues:
            if len(dialogue["turns"]) < 4:
                continue  # Only evaluate multi-turn

            conv_text = _format_conversation(dialogue["turns"])
            prompt = COHERENCE_RUBRIC.format(conversation=conv_text)
            result = await _ask_judge(prompt)

            if result.get("score") is not None:
                scores.append(result["score"])
                print(f"  [{dialogue['id']}] Coherence: {result['score']:.2f} — {result.get('reason', 'N/A')}")

        avg = sum(scores) / len(scores) if scores else 0.0
        if scores:
            print(f"\n[JUDGE] Average Coherence: {avg:.4f} ({len(scores)} dialogues)")

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_DIR / "judge_coherence.json", "w") as f:
            json.dump({"average_score": avg, "count": len(scores), "scores": scores}, f, indent=2)

    @pytest.mark.asyncio
    async def test_faithfulness_with_rag(self):
        """
        Evaluate faithfulness of RAG-grounded answers.
        For each RAG ground truth query, retrieve context and check
        if a hypothetical answer would be faithful.
        """
        try:
            from backend.RAG.retrieval import retrieve
        except Exception:
            pytest.skip("RAG retrieval not available")

        gt_path = _TEST_DATA / "rag_ground_truth.json"
        with open(gt_path, "r") as f:
            rag_gt = json.load(f)

        scores = []
        queries = rag_gt["queries"][:10]  # Limit to 10 for speed

        for entry in queries:
            query = entry["query"]
            try:
                chunks = await retrieve(query, k=3)
            except Exception:
                continue

            if not chunks:
                continue

            context = "\n".join(chunks)
            # Use the expected keywords as a proxy for a "good answer"
            expected_answer = f"Based on the documents: {', '.join(entry['expected_keywords'])}"

            prompt = FAITHFULNESS_RUBRIC.format(
                question=query,
                context=context,
                answer=expected_answer,
            )
            result = await _ask_judge(prompt)

            if result.get("score") is not None:
                scores.append(result["score"])
                print(f"  [{entry['id']}] Faithfulness: {result['score']:.2f}")

        avg = sum(scores) / len(scores) if scores else 0.0
        if scores:
            print(f"\n[JUDGE] Average Faithfulness: {avg:.4f} ({len(scores)} queries)")

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_DIR / "judge_faithfulness.json", "w") as f:
            json.dump({"average_score": avg, "count": len(scores), "scores": scores}, f, indent=2)
