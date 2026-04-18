"""
Tests for diary_session safe search query generation and ArXiv safety.
"""

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime

from diary.diary_session import DiarySessionManager, DiarySession, DiaryEntry
from services.web_search_service import _safe_query


# ?��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��?
# Helpers
# ?��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��?

PSYCH_TERMS = [
    "psychology", "therapy", "coping", "emotion", "emotional",
    "neuroscience", "brain", "stress", "mindfulness",
    "cognitive", "behavioral", "well-being", "regulation",
    "flexibility", "awareness", "meditation", "processing",
    "disgust", "moral", "immune", "default mode",
    "dopamine", "serotonin", "neurochemistry", "happiness",
    "anger", "fear", "anxiety", "depression", "sadness",
    "mental health", "resilience", "intervention",
    "positive", "negative", "prefrontal", "amygdala",
    "novelty", "uncertainty", "startle", "adaptation",
    "sensitivity", "contamination", "avoidance",
    "intelligence", "resting state",
]


def _has_psych_term(text: str) -> bool:
    text_lower = text.lower()
    return any(term in text_lower for term in PSYCH_TERMS)


def _make_entry(text="I feel sad today", emotion="sad", confidence=0.8):
    return DiaryEntry(
        timestamp="12:00:00",
        text=text,
        face_emotion=emotion,
        face_confidence=confidence,
        voice_sentiment={"polarity": -0.3, "subjectivity": 0.6, "emotion": emotion, "confidence": 0.7},
        audio_emotion={"energy": 0.4, "pitch_var": 0.3, "estimated_emotion": emotion, "confidence": 0.6},
        fused_emotion=emotion,
        fused_confidence=confidence,
    )


def _make_session(entries=None, emotion="sad"):
    session = DiarySession(
        session_id="test-123",
        start_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )
    if entries:
        session.entries = entries
    else:
        session.entries = [_make_entry(emotion=emotion)]
    return session


# ?��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��?
# Research query safety
# ?��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��?


class TestResearchQuerySafety:
    """Ensure research queries from diary sessions are always psychology-scoped."""

    def test_llm_queries_get_safe_query_applied(self):
        """When LLM generates queries, _safe_query is applied to each."""
        mgr = DiarySessionManager()
        session = _make_session(emotion="sad")

        with patch.object(mgr._llm, "_generate", return_value="how to feel better\ndealing with problems\novercoming challenges"):
            queries = mgr.get_research_queries(session)

        for q in queries:
            assert _has_psych_term(q), f"LLM query not safety-scoped: {q}"

    def test_llm_queries_with_existing_psych_terms_unchanged(self):
        """Queries already containing psychology terms shouldn't be double-appended."""
        mgr = DiarySessionManager()
        session = _make_session(emotion="sad")

        with patch.object(mgr._llm, "_generate", return_value="cognitive behavioral therapy effectiveness\nmindfulness meditation research"):
            queries = mgr.get_research_queries(session)

        assert any("cognitive behavioral therapy" in q.lower() for q in queries)
        assert any("mindfulness" in q.lower() for q in queries)

    def test_empty_session_returns_empty(self):
        mgr = DiarySessionManager()
        session = DiarySession(session_id="empty", start_time="2026-01-01 00:00:00")
        assert mgr.get_research_queries(session) == []

    def test_llm_failure_returns_empty_queries(self):
        """When LLM fails, get_research_queries should return an empty list."""
        mgr = DiarySessionManager()
        session = _make_session(emotion="angry")

        with patch.object(mgr._llm, "_generate", return_value=""):
            queries = mgr.get_research_queries(session)

        assert queries == []


# ?��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��?
# ArXiv search safety
# ?��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��?


class TestArxivSearchSafety:
    """Ensure ArXiv searches use safe queries."""

    def test_arxiv_search_applies_safe_query(self):
        """_search_arxiv should run queries through _safe_query."""
        queries = ["surprise response brain", "novel stimulus reaction"]
        with patch("mcp_client.search_papers") as mock_search:
            mock_search.return_value = []
            DiarySessionManager._search_arxiv(queries, max_per_query=3)

            for call in mock_search.call_args_list:
                q = call[0][0]
                assert _has_psych_term(q), f"ArXiv query not safe: {q}"

    def test_arxiv_search_already_safe_query_unchanged(self):
        """Queries with psychology terms should pass through without extra appending."""
        queries = ["emotion regulation neuroscience"]
        with patch("mcp_client.search_papers") as mock_search:
            mock_search.return_value = []
            DiarySessionManager._search_arxiv(queries, max_per_query=3)

            call_q = mock_search.call_args_list[0][0][0]
            assert call_q == "emotion regulation neuroscience"

    def test_arxiv_search_handles_search_exception(self):
        """Individual query failures should be caught, not crash the whole search."""
        with patch("mcp_client.search_papers", side_effect=Exception("API error")):
            results = DiarySessionManager._search_arxiv(["test query psychology"], max_per_query=3)
            assert results == []

    def test_arxiv_limits_to_3_queries(self):
        """At most 3 queries should be searched."""
        queries = ["q1 emotion", "q2 psychology", "q3 therapy", "q4 coping", "q5 mindfulness"]
        with patch("mcp_client.search_papers") as mock_search:
            mock_search.return_value = []
            DiarySessionManager._search_arxiv(queries, max_per_query=3)
            assert mock_search.call_count == 3


# ?��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��?
# end_session web search safety
# ?��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��?


class TestEndSessionWebSearchSafety:
    """Ensure end_session produces safe web search results."""

    def test_end_session_uses_search_emotion_articles(self):
        """end_session should call search_emotion_articles (which uses safe search)."""
        mgr = DiarySessionManager()
        session = _make_session(emotion="surprise")

        with patch("diary_session.search_emotion_articles") as mock_articles, \
             patch("diary_session.search_general") as mock_general, \
             patch.object(mgr._llm, "_generate", return_value="Session about surprise."), \
             patch("mcp_client.search_papers", return_value=[]):

            mock_articles.return_value = [
                {"title": "Surprise Psychology", "url": "https://apa.org/surprise", "snippet": "Research on surprise emotion."},
            ]
            mock_general.return_value = []

            result = mgr.end_session(session)
            mock_articles.assert_called_once()
            assert mock_articles.call_args[0][0] == "surprise"

    def test_end_session_web_results_deduplicated(self):
        """Web results in end_session should be deduplicated by URL."""
        mgr = DiarySessionManager()
        session = _make_session(emotion="sad")

        with patch("diary_session.search_emotion_articles") as mock_articles, \
             patch("diary_session.search_general") as mock_general, \
             patch.object(mgr._llm, "_generate", return_value="Session about sadness."), \
             patch("mcp_client.search_papers", return_value=[]):

            dup_result = {"title": "Coping with Sadness", "url": "https://apa.org/coping", "snippet": "Strategies for sadness."}
            mock_articles.return_value = [dup_result]
            mock_general.return_value = [dup_result]

            result = mgr.end_session(session)
            urls = [r["url"] for r in result.web_results]
            assert len(urls) == len(set(urls)), "Web results should be deduplicated by URL"


# ?��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��?
# _safe_query integration
# ?��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��?


class TestSafeQueryIntegration:
    """Tests that _safe_query is importable from diary_session context."""

    def test_safe_query_importable(self):
        from services.web_search_service import _safe_query
        assert callable(_safe_query)

    def test_safe_query_idempotent_for_psych_queries(self):
        q = "cognitive behavioral therapy for depression"
        assert _safe_query(q) == q

    def test_safe_query_adds_context_to_generic(self):
        q = "how to deal with problems"
        result = _safe_query(q)
        # Scoping appended when no safe keyword present
        assert q in result
        assert len(result) > len(q)
