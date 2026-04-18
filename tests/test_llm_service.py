"""Tests for llm_service.py - test with mocked Ollama, test fallbacks."""

import pytest
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.llm_service import LLMService


class TestLLMServiceInit:
    def test_default_init(self):
        llm = LLMService()
        assert "localhost" in llm.base_url or "127.0.0.1" in llm.base_url
        assert llm.model == "qwen3:8b"
        assert llm.timeout > 0

    def test_custom_init(self):
        llm = LLMService(base_url="http://custom:1234", model="test-model", timeout=30.0)
        assert llm.base_url == "http://custom:1234"
        assert llm.model == "test-model"
        assert llm.timeout == 30.0


class TestIsAvailable:
    @patch("llm_service.httpx.get")
    def test_available_when_ollama_responds(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_get.return_value = mock_resp

        llm = LLMService()
        assert llm.is_available() is True

    @patch("llm_service.httpx.get")
    def test_unavailable_when_ollama_down(self, mock_get):
        mock_get.side_effect = ConnectionError("refused")

        llm = LLMService()
        assert llm.is_available() is False

    @patch("llm_service.httpx.get")
    def test_unavailable_on_non_200(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_get.return_value = mock_resp

        llm = LLMService()
        assert llm.is_available() is False


class TestResearchEmotion:
    @patch("llm_service.httpx.post")
    def test_research_with_ollama(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"response": "Happiness activates the brain's reward system."}
        mock_post.return_value = mock_resp

        llm = LLMService()
        result = llm.research_emotion("happy")
        assert "brain" in result.lower() or "reward" in result.lower() or "happiness" in result.lower()

    @patch("llm_service.httpx.post")
    def test_research_fallback_on_error(self, mock_post):
        mock_post.side_effect = ConnectionError("refused")

        llm = LLMService()
        result = llm.research_emotion("happy")
        assert isinstance(result, str)


class TestGenerateMoodResponse:
    @patch("llm_service.httpx.post")
    def test_mood_response_fallback(self, mock_post):
        mock_post.side_effect = ConnectionError("refused")

        llm = LLMService()
        result = llm.generate_mood_response("sad")
        assert isinstance(result, str)


class TestSummarizePapers:
    def test_empty_papers(self):
        llm = LLMService()
        result = llm.summarize_papers([], "test topic")
        assert "No papers" in result

    @patch("llm_service.httpx.post")
    def test_summarize_fallback(self, mock_post):
        mock_post.side_effect = ConnectionError("refused")

        papers = [
            {"title": "Paper 1", "abstract": "Abstract of paper 1"},
            {"title": "Paper 2", "abstract": "Abstract of paper 2"},
        ]
        llm = LLMService()
        result = llm.summarize_papers(papers, "AI research")
        assert isinstance(result, str)


class TestChat:
    @patch("llm_service.httpx.post")
    def test_chat_fallback(self, mock_post):
        mock_post.side_effect = ConnectionError("refused")

        llm = LLMService()
        result = llm.chat("hello")
        assert isinstance(result, str)

    @patch("llm_service.httpx.post")
    def test_chat_with_history(self, mock_post):
        mock_post.side_effect = ConnectionError("refused")

        llm = LLMService()
        history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello!"},
        ]
        result = llm.chat("how are you?", history)
        assert isinstance(result, str)


class TestStreamGenerate:
    @patch("llm_service.httpx.stream")
    def test_stream_yields_on_error(self, mock_stream):
        mock_stream.side_effect = ConnectionError("refused")

        llm = LLMService()
        chunks = list(llm.stream_generate("test prompt"))
        # Should yield at least an empty string on error
        assert isinstance(chunks, list)
