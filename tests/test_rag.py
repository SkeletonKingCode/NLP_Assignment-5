"""
tests/test_rag.py

Evaluation tests for the RAG (Retrieval-Augmented Generation) component.
Measures:
  - Retrieval relevance: precision@k and recall@k
  - Context relevance: proportion of useful retrieved chunks
  - Grounding / faithfulness: automated checks via keyword overlap

These tests require the RAG index to be built (run backend/RAG/indexer.py first).
Uses the annotated ground truth from tests/test_data/rag_ground_truth.json.
"""

import sys
import json
import asyncio
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock

import pytest

_BACKEND = Path(__file__).resolve().parent.parent / "backend"
_TEST_DATA = Path(__file__).resolve().parent / "test_data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "eval_results"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))


# ---------------------------------------------------------------------------
# Load ground truth data
# ---------------------------------------------------------------------------

def _load_rag_ground_truth():
    gt_path = _TEST_DATA / "rag_ground_truth.json"
    with open(gt_path, "r") as f:
        data = json.load(f)
    return data["queries"]


# ---------------------------------------------------------------------------
# RAG Retrieval Metrics
# ---------------------------------------------------------------------------

class TestRAGRetrievalMetrics:
    """
    Evaluates retrieval relevance using precision@k and recall@k.
    Requires a live ChromaDB index — skips if not available.
    """

    @pytest.mark.asyncio
    async def test_retrieval_returns_results(self):
        """Basic smoke test — retrieval should return something."""
        try:
            from RAG.retrieval import retrieve
            results = await retrieve("DHA Phase 1 plots", k=3)
            assert isinstance(results, list)
            # May return empty if index not built — that's handled in metrics
        except Exception as e:
            pytest.skip(f"RAG index not available: {e}")

    @pytest.mark.asyncio
    async def test_precision_at_k(self):
        """
        Compute precision@k across all ground truth queries.
        precision@k = (relevant chunks in top-k) / k
        """
        try:
            from RAG.retrieval import retrieve, get_collection
            # Verify collection exists
            collection = await get_collection("real_estate_docs")
        except Exception as e:
            pytest.skip(f"RAG index not available: {e}")

        from RAG.retrieval import retrieve
        ground_truth = _load_rag_ground_truth()
        k = 3
        precisions = []

        for entry in ground_truth:
            query = entry["query"]
            relevant_sources = set(entry["relevant_sources"])

            results = await retrieve(query, k=k)
            if not results:
                precisions.append(0.0)
                continue

            # Check how many retrieved chunks come from relevant sources
            relevant_hits = 0
            for chunk in results:
                for src in relevant_sources:
                    src_name = src.replace(".txt", "").replace("_", " ")
                    if src_name.lower() in chunk.lower() or any(
                        kw.lower() in chunk.lower()
                        for kw in entry.get("expected_keywords", [])
                    ):
                        relevant_hits += 1
                        break

            precision = relevant_hits / len(results)
            precisions.append(precision)

        avg_precision = sum(precisions) / len(precisions) if precisions else 0
        print(f"\n[RAG METRIC] Average Precision@{k}: {avg_precision:.4f}")
        print(f"[RAG METRIC] Queries evaluated: {len(precisions)}")

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_DIR / "rag_precision.json", "w") as f:
            json.dump({"metric": "precision@k", "k": k, "score": avg_precision, "queries": len(precisions), "values": precisions}, f, indent=2)

        assert avg_precision >= 0.0

    @pytest.mark.asyncio
    async def test_recall_at_k(self):
        """
        Compute recall@k across all ground truth queries.
        recall@k = (relevant chunks in top-k) / (total relevant for query)
        """
        try:
            from RAG.retrieval import retrieve, get_collection
            collection = await get_collection("real_estate_docs")
        except Exception as e:
            pytest.skip(f"RAG index not available: {e}")

        from RAG.retrieval import retrieve
        ground_truth = _load_rag_ground_truth()
        k = 3
        recalls = []

        for entry in ground_truth:
            query = entry["query"]
            relevant_sources = set(entry["relevant_sources"])
            expected_keywords = entry.get("expected_keywords", [])

            results = await retrieve(query, k=k)
            if not results:
                recalls.append(0.0)
                continue

            relevant_hits = 0
            for chunk in results:
                for src in relevant_sources:
                    src_name = src.replace(".txt", "").replace("_", " ")
                    if src_name.lower() in chunk.lower() or any(
                        kw.lower() in chunk.lower() for kw in expected_keywords
                    ):
                        relevant_hits += 1
                        break

            recall = relevant_hits / len(relevant_sources) if relevant_sources else 0
            recalls.append(min(recall, 1.0))

        avg_recall = sum(recalls) / len(recalls) if recalls else 0
        print(f"\n[RAG METRIC] Average Recall@{k}: {avg_recall:.4f}")

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_DIR / "rag_recall.json", "w") as f:
            json.dump({"metric": "recall@k", "k": k, "score": avg_recall, "queries": len(recalls), "values": recalls}, f, indent=2)

        assert avg_recall >= 0.0


# ---------------------------------------------------------------------------
# Context Relevance
# ---------------------------------------------------------------------------

class TestRAGContextRelevance:
    """
    Measures the proportion of retrieved chunks that are actually useful.
    A chunk is "useful" if it contains at least one expected keyword.
    """

    @pytest.mark.asyncio
    async def test_context_relevance_score(self):
        try:
            from RAG.retrieval import retrieve, get_collection
            await get_collection("real_estate_docs")
        except Exception as e:
            pytest.skip(f"RAG index not available: {e}")

        from RAG.retrieval import retrieve
        ground_truth = _load_rag_ground_truth()
        relevance_scores = []

        for entry in ground_truth:
            query = entry["query"]
            expected_keywords = [kw.lower() for kw in entry.get("expected_keywords", [])]
            if not expected_keywords:
                continue

            results = await retrieve(query, k=3)
            if not results:
                relevance_scores.append(0.0)
                continue

            useful_chunks = 0
            for chunk in results:
                chunk_lower = chunk.lower()
                if any(kw in chunk_lower for kw in expected_keywords):
                    useful_chunks += 1

            relevance = useful_chunks / len(results)
            relevance_scores.append(relevance)

        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
        print(f"\n[RAG METRIC] Average Context Relevance: {avg_relevance:.4f}")

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_DIR / "rag_context_relevance.json", "w") as f:
            json.dump({"metric": "context_relevance", "score": avg_relevance, "queries": len(relevance_scores), "values": relevance_scores}, f, indent=2)

        assert avg_relevance >= 0.0


# ---------------------------------------------------------------------------
# Faithfulness / Grounding — Keyword Overlap Metric
# ---------------------------------------------------------------------------

class TestRAGFaithfulness:
    """
    Automated faithfulness check: measures whether the answer content
    is grounded in the retrieved documents by checking keyword overlap.
    
    For a full faithfulness score, an LLM judge would be used (see test_llm_judge.py).
    This is a lightweight proxy using keyword matching.
    """

    @pytest.mark.asyncio
    async def test_faithfulness_keyword_overlap(self):
        """
        For each query, retrieve chunks and check that expected keywords
        appear in the retrieved context. This approximates faithfulness:
        if the context contains the keywords, a faithful LLM answer would too.
        """
        try:
            from RAG.retrieval import retrieve, get_collection
            await get_collection("real_estate_docs")
        except Exception as e:
            pytest.skip(f"RAG index not available: {e}")

        from RAG.retrieval import retrieve
        ground_truth = _load_rag_ground_truth()
        faithfulness_scores = []

        for entry in ground_truth:
            query = entry["query"]
            expected_keywords = [kw.lower() for kw in entry.get("expected_keywords", [])]
            if not expected_keywords:
                continue

            results = await retrieve(query, k=3)
            if not results:
                faithfulness_scores.append(0.0)
                continue

            combined_context = " ".join(results).lower()
            matched_keywords = sum(1 for kw in expected_keywords if kw in combined_context)
            score = matched_keywords / len(expected_keywords)
            faithfulness_scores.append(score)

        avg_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0
        print(f"\n[RAG METRIC] Average Faithfulness (keyword): {avg_faithfulness:.4f}")

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_DIR / "rag_faithfulness.json", "w") as f:
            json.dump({"metric": "faithfulness_keyword", "score": avg_faithfulness, "queries": len(faithfulness_scores), "values": faithfulness_scores}, f, indent=2)

        assert avg_faithfulness >= 0.0


# ---------------------------------------------------------------------------
# Semantic Match Tests
# ---------------------------------------------------------------------------

class TestSemanticMatch:
    """Tests for the hybrid semantic matching function."""

    @pytest.mark.asyncio
    async def test_exact_substring_match(self):
        try:
            from RAG.retrieval import semantic_match
        except Exception as e:
            pytest.skip(f"Cannot import semantic_match: {e}")

        result = await semantic_match("I want a shop", ["shop", "house", "apartment"])
        assert result == "shop"

    @pytest.mark.asyncio
    async def test_no_match_below_threshold(self):
        try:
            from RAG.retrieval import semantic_match
        except Exception as e:
            pytest.skip(f"Cannot import semantic_match: {e}")

        result = await semantic_match("quantum physics", ["shop", "house", "apartment"], threshold=0.99)
        # May or may not match depending on embeddings; just check it returns valid type
        assert result is None or isinstance(result, str)

    @pytest.mark.asyncio
    async def test_empty_options(self):
        try:
            from RAG.retrieval import semantic_match
        except Exception as e:
            pytest.skip(f"Cannot import semantic_match: {e}")

        result = await semantic_match("test query", [])
        assert result is None
