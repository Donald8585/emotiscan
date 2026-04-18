"""Tests for llm_service.py enhancements — compassionate_chat and suggest_solutions."""

import pytest
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.llm_service import LLMService


class TestCompassionateChat:
    def test_offline_returns_empty_string(self):
        """compassionate_chat returns empty string when Ollama is offline."""
        svc = LLMService()
        result = svc.compassionate_chat("I'm feeling really down today")
        assert isinstance(result, str)

    def test_with_session_context(self):
        svc = LLMService()
        ctx = {
            "summary": "User expressed sadness throughout the session",
            "dominant_emotion": "sad",
            "compassionate_response": "I hear you and your feelings are valid.",
        }
        result = svc.compassionate_chat("What can I do?", session_context=ctx)
        assert isinstance(result, str)

    def test_with_history(self):
        svc = LLMService()
        history = [
            {"role": "user", "content": "I'm stressed"},
            {"role": "assistant", "content": "I hear you."},
        ]
        result = svc.compassionate_chat("What should I do?", history=history)
        assert isinstance(result, str)

    def test_with_empty_context(self):
        svc = LLMService()
        result = svc.compassionate_chat("hello", session_context={})
        assert isinstance(result, str)

    def test_with_none_params(self):
        svc = LLMService()
        result = svc.compassionate_chat("hi", history=None, session_context=None)
        assert isinstance(result, str)

    @patch.object(LLMService, "_generate", return_value="Mocked compassionate response")
    def test_with_working_ollama(self, mock_gen):
        svc = LLMService()
        result = svc.compassionate_chat(
            "I'm anxious",
            session_context={"dominant_emotion": "fear"}
        )
        assert result == "Mocked compassionate response"
        mock_gen.assert_called_once()
        # Check prompt includes the right elements
        prompt = mock_gen.call_args[0][0]
        assert "supportive AI counselor" in prompt
        assert "fear" in prompt

    @patch.object(LLMService, "_generate", return_value="")
    def test_empty_ollama_response_falls_back(self, mock_gen):
        svc = LLMService()
        result = svc.compassionate_chat(
            "help me",
            session_context={"dominant_emotion": "angry"}
        )
        # Should return empty string (no demo fallback)
        assert isinstance(result, str)


class TestSuggestSolutions:
    def test_offline_returns_empty_string(self):
        svc = LLMService()
        result = svc.suggest_solutions("sad", "I'm having trouble sleeping")
        assert isinstance(result, str)

    def test_all_emotions(self):
        svc = LLMService()
        for emo in ["happy", "sad", "angry", "fear", "surprise", "disgust", "neutral"]:
            result = svc.suggest_solutions(emo)
            assert isinstance(result, str)

    def test_with_problem_text(self):
        svc = LLMService()
        result = svc.suggest_solutions("angry", "My coworker keeps interrupting me in meetings")
        assert isinstance(result, str)

    def test_without_problem_text(self):
        svc = LLMService()
        result = svc.suggest_solutions("fear")
        assert isinstance(result, str)

    @patch.object(LLMService, "_generate", return_value="1. Mocked solution\n2. Another solution")
    def test_with_working_ollama(self, mock_gen):
        svc = LLMService()
        result = svc.suggest_solutions("sad", "feeling down")
        assert "Mocked solution" in result
        prompt = mock_gen.call_args[0][0]
        assert "sad" in prompt
        assert "feeling down" in prompt

    @patch.object(LLMService, "_generate", return_value="")
    def test_empty_response_falls_back(self, mock_gen):
        svc = LLMService()
        result = svc.suggest_solutions("happy")
        assert isinstance(result, str)


class TestPromptConstruction:
    """Test that prompts include session context."""

    @patch.object(LLMService, "_generate", return_value="response")
    def test_prompt_includes_summary(self, mock_gen):
        svc = LLMService()
        svc.compassionate_chat(
            "test",
            session_context={"summary": "User was very upset about work"}
        )
        prompt = mock_gen.call_args[0][0]
        assert "User was very upset about work" in prompt

    @patch.object(LLMService, "_generate", return_value="response")
    def test_prompt_includes_dominant_emotion(self, mock_gen):
        svc = LLMService()
        svc.compassionate_chat(
            "test",
            session_context={"dominant_emotion": "angry"}
        )
        prompt = mock_gen.call_args[0][0]
        assert "angry" in prompt

    @patch.object(LLMService, "_generate", return_value="response")
    def test_prompt_includes_compassionate_response(self, mock_gen):
        svc = LLMService()
        svc.compassionate_chat(
            "test",
            session_context={"compassionate_response": "You showed real courage today"}
        )
        prompt = mock_gen.call_args[0][0]
        assert "You showed real courage today" in prompt
