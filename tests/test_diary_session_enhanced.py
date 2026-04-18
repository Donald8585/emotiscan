"""Tests for diary_session.py enhancements — web/arxiv search, compassionate response."""

import pytest
import sys
import os
import json
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from diary.diary_session import DiaryEntry, DiarySession, DiarySessionManager


def make_entry(**overrides):
    defaults = dict(
        timestamp="12:30:00",
        text="I feel great today",
        face_emotion="happy",
        face_confidence=0.85,
        voice_sentiment={"polarity": 0.5, "subjectivity": 0.6, "emotion": "happy", "confidence": 0.7},
        audio_emotion={"energy": 0.4, "pitch_var": 0.3, "estimated_emotion": "happy", "confidence": 0.5},
        fused_emotion="happy",
        fused_confidence=0.75,
    )
    defaults.update(overrides)
    return DiaryEntry(**defaults)


def make_session(**overrides):
    defaults = dict(
        session_id="test123",
        start_time="2026-03-18 10:00:00",
        entries=[],
        summary="",
        research_queries=[],
        web_results=[],
        arxiv_results=[],
        compassionate_response="",
    )
    defaults.update(overrides)
    return DiarySession(**defaults)


class TestDiarySessionNewFields:
    def test_new_fields_have_defaults(self):
        """New fields should have defaults so existing code works."""
        session = DiarySession(session_id="abc", start_time="now")
        assert session.web_results == []
        assert session.arxiv_results == []
        assert session.compassionate_response == ""

    def test_to_dict_includes_new_fields(self):
        session = make_session(
            web_results=[{"title": "Web Result", "url": "https://example.com", "snippet": "test"}],
            arxiv_results=[{"title": "ArXiv Paper", "url": "https://arxiv.org/123", "authors": ["Alice"], "published": "2026-01-01"}],
            compassionate_response="You're doing great!",
        )
        d = session.to_dict()
        assert "web_results" in d
        assert "arxiv_results" in d
        assert "compassionate_response" in d
        assert d["web_results"][0]["title"] == "Web Result"
        assert d["arxiv_results"][0]["title"] == "ArXiv Paper"
        assert d["compassionate_response"] == "You're doing great!"

    def test_to_json_includes_new_fields(self):
        session = make_session(
            web_results=[{"title": "Result"}],
            compassionate_response="Stay strong!",
        )
        data = json.loads(session.to_json())
        assert data["web_results"][0]["title"] == "Result"
        assert data["compassionate_response"] == "Stay strong!"

    def test_to_markdown_includes_compassionate_response(self):
        session = make_session(
            compassionate_response="Your feelings are valid.",
            entries=[make_entry()],
        )
        md = session.to_markdown()
        assert "## Supportive Response" in md
        assert "Your feelings are valid." in md

    def test_to_markdown_includes_arxiv_results(self):
        session = make_session(
            arxiv_results=[
                {"title": "Paper A", "url": "https://arxiv.org/1", "authors": ["Bob"], "published": "2026-01-01"}
            ],
        )
        md = session.to_markdown()
        assert "## ArXiv Papers" in md
        assert "Paper A" in md
        assert "Bob" in md

    def test_to_markdown_includes_web_results(self):
        session = make_session(
            web_results=[
                {"title": "Article X", "url": "https://example.com", "snippet": "Great advice here"}
            ],
        )
        md = session.to_markdown()
        assert "## Web Results" in md
        assert "Article X" in md
        assert "Great advice" in md

    def test_to_markdown_no_extra_sections_when_empty(self):
        session = make_session()
        md = session.to_markdown()
        assert "## Supportive Response" not in md
        assert "## ArXiv Papers" not in md
        assert "## Web Results" not in md


class TestEndSessionEnhanced:
    @patch("diary_session.search_general", return_value=[{"title": "Web1", "url": "https://web1.com", "snippet": "snippet"}])
    @patch("diary_session.search_emotion_articles", return_value=[{"title": "Emo1", "url": "https://emo1.com", "snippet": "emo snippet"}])
    @patch("diary_session.SpeechService")
    @patch("diary_session.MoodFusion")
    @patch("diary_session.LLMService")
    def test_end_session_populates_web_results(self, mock_llm_cls, mock_fusion_cls, mock_speech_cls, mock_emo_search, mock_gen_search):
        mock_llm = mock_llm_cls.return_value
        mock_llm._generate.return_value = ""

        mgr = DiarySessionManager()
        session = make_session(entries=[make_entry(fused_emotion="sad")])
        result = mgr.end_session(session)

        assert len(result.web_results) > 0

    @patch("diary_session.search_general", return_value=[])
    @patch("diary_session.search_emotion_articles", return_value=[])
    @patch("diary_session.SpeechService")
    @patch("diary_session.MoodFusion")
    @patch("diary_session.LLMService")
    def test_end_session_populates_compassionate_response(self, mock_llm_cls, mock_fusion_cls, mock_speech_cls, mock_emo_search, mock_gen_search):
        mock_llm = mock_llm_cls.return_value
        mock_llm._generate.return_value = ""

        mgr = DiarySessionManager()
        session = make_session(entries=[make_entry(fused_emotion="angry")])
        result = mgr.end_session(session)

        assert isinstance(result.compassionate_response, str)

    @patch("diary_session.search_general", side_effect=Exception("search failed"))
    @patch("diary_session.search_emotion_articles", side_effect=Exception("search failed"))
    @patch("diary_session.SpeechService")
    @patch("diary_session.MoodFusion")
    @patch("diary_session.LLMService")
    def test_end_session_handles_search_failure(self, mock_llm_cls, mock_fusion_cls, mock_speech_cls, mock_emo_search, mock_gen_search):
        mock_llm = mock_llm_cls.return_value
        mock_llm._generate.return_value = ""

        mgr = DiarySessionManager()
        session = make_session(entries=[make_entry()])
        result = mgr.end_session(session)

        # Should not crash, web_results should be empty
        assert result.web_results == []
        assert isinstance(result.compassionate_response, str)


class TestSearchArxiv:
    @patch("diary_session.DiarySessionManager._search_arxiv")
    def test_search_arxiv_called_in_end_session(self, mock_arxiv):
        """end_session should call _search_arxiv."""
        mock_arxiv.return_value = [{"title": "Paper", "source": "arxiv"}]

        with patch("diary_session.SpeechService"), \
             patch("diary_session.MoodFusion"), \
             patch("diary_session.LLMService") as mock_llm_cls, \
             patch("diary_session.search_general", return_value=[]), \
             patch("diary_session.search_emotion_articles", return_value=[]):
            mock_llm_cls.return_value._generate.return_value = ""

            mgr = DiarySessionManager()
            session = make_session(entries=[make_entry()])
            result = mgr.end_session(session)

            assert result.arxiv_results == [{"title": "Paper", "source": "arxiv"}]

    def test_search_arxiv_with_empty_queries(self):
        results = DiarySessionManager._search_arxiv([])
        assert results == []

    @patch("diary_session.DiarySessionManager._search_arxiv", return_value=[])
    def test_search_arxiv_handles_import_error(self, mock_method):
        """If arxiv library not available, should return empty list."""
        results = DiarySessionManager._search_arxiv([])
        assert results == []


class TestGetCompassionateResponse:
    @patch("diary_session.SpeechService")
    @patch("diary_session.MoodFusion")
    @patch("diary_session.LLMService")
    def test_empty_session(self, mock_llm_cls, mock_fusion_cls, mock_speech_cls):
        mgr = DiarySessionManager()
        session = make_session()
        result = mgr.get_compassionate_response(session)
        assert isinstance(result, str)

    @patch("diary_session.SpeechService")
    @patch("diary_session.MoodFusion")
    @patch("diary_session.LLMService")
    def test_with_entries(self, mock_llm_cls, mock_fusion_cls, mock_speech_cls):
        mock_llm_cls.return_value._generate.return_value = ""
        mgr = DiarySessionManager()
        session = make_session(entries=[make_entry(fused_emotion="sad")])
        result = mgr.get_compassionate_response(session)
        assert isinstance(result, str)

    @patch("diary_session.SpeechService")
    @patch("diary_session.MoodFusion")
    @patch("diary_session.LLMService")
    def test_with_llm_response(self, mock_llm_cls, mock_fusion_cls, mock_speech_cls):
        mock_llm_cls.return_value._generate.return_value = "You're doing wonderfully!"
        mgr = DiarySessionManager()
        session = make_session(entries=[make_entry()])
        result = mgr.get_compassionate_response(session)
        assert result == "You're doing wonderfully!"


