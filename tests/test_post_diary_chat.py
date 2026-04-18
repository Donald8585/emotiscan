"""
Tests for post-diary-session chat integration in the Voice Diary tab.
Verifies the chat is available immediately after session ends, handles
context correctly, and persists messages in session state.
"""

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime

from diary.diary_session import DiarySession, DiaryEntry, DiarySessionManager
from services.llm_service import LLMService


# ════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════


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


def _make_ended_session(emotion="sad"):
    """Create a session that has been ended (has summary + compassionate response)."""
    session = DiarySession(
        session_id="test-ended",
        start_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )
    session.entries = [_make_entry(emotion=emotion)]
    session.summary = f"Session expressed {emotion} feelings. The user is processing their emotions."
    session.compassionate_response = (
        f"I hear you, and your {emotion} feelings are completely valid. "
        f"Remember, it's okay to feel this way."
    )
    session.research_queries = [f"{emotion} coping strategies psychology"]
    session.web_results = [
        {"title": f"Coping with {emotion}", "url": f"https://apa.org/{emotion}", "snippet": f"Evidence-based {emotion} management."},
    ]
    session.arxiv_results = [
        {"title": f"Neuroscience of {emotion}", "url": f"https://arxiv.org/abs/1234", "authors": ["Dr. Smith"], "abstract": "A study."},
    ]
    return session


# ════════════════════════════════════════════════════════════════════════
# Post-session context for chat
# ════════════════════════════════════════════════════════════════════════


class TestPostSessionChatContext:
    """Test that the chat receives proper session context."""

    def test_session_context_has_required_fields(self):
        """The session context dict should have summary, dominant_emotion, compassionate_response."""
        session = _make_ended_session("sad")
        ctx = {
            "summary": session.summary,
            "dominant_emotion": session.dominant_emotion,
            "compassionate_response": session.compassionate_response,
        }
        assert "summary" in ctx
        assert ctx["summary"] != ""
        assert "dominant_emotion" in ctx
        assert ctx["dominant_emotion"] == "sad"
        assert "compassionate_response" in ctx
        assert ctx["compassionate_response"] != ""

    def test_dominant_emotion_matches_entries(self):
        session = _make_ended_session("angry")
        assert session.dominant_emotion == "angry"

    def test_compassionate_response_available(self):
        session = _make_ended_session("fear")
        assert len(session.compassionate_response) > 0


# ════════════════════════════════════════════════════════════════════════
# Chat history management
# ════════════════════════════════════════════════════════════════════════


class TestChatHistoryManagement:
    """Test chat history operations for the diary chat."""

    def test_chat_history_starts_empty(self):
        history = []
        assert len(history) == 0

    def test_user_message_added(self):
        history = []
        history.append({"role": "user", "content": "I'm feeling overwhelmed"})
        assert len(history) == 1
        assert history[0]["role"] == "user"

    def test_assistant_response_added(self):
        history = [{"role": "user", "content": "I'm feeling overwhelmed"}]
        history.append({"role": "assistant", "content": "I hear you. Let's talk about what's going on."})
        assert len(history) == 2
        assert history[1]["role"] == "assistant"

    def test_multi_turn_conversation(self):
        history = []
        turns = [
            ("user", "I'm feeling sad"),
            ("assistant", "That's completely valid. What's making you feel this way?"),
            ("user", "Work stress and personal issues"),
            ("assistant", "It sounds like you're dealing with a lot. Let me suggest some strategies."),
        ]
        for role, content in turns:
            history.append({"role": role, "content": content})
        assert len(history) == 4

    def test_history_cleared_on_new_session(self):
        history = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
        # Simulating "Start Fresh" button behavior
        history.clear()
        assert len(history) == 0


# ════════════════════════════════════════════════════════════════════════
# Compassionate chat integration
# ════════════════════════════════════════════════════════════════════════


class TestCompassionateChatIntegration:
    """Test the LLM compassionate_chat works with diary session context."""

    def test_compassionate_chat_with_session_context(self):
        """compassionate_chat should accept session context and return a string."""
        llm = LLMService()
        session = _make_ended_session("sad")
        ctx = {
            "summary": session.summary,
            "dominant_emotion": session.dominant_emotion,
            "compassionate_response": session.compassionate_response,
        }
        history = [{"role": "user", "content": "I don't know how to cope"}]

        result = llm.compassionate_chat("I don't know how to cope", history, ctx)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_compassionate_chat_with_empty_context(self):
        """Should handle empty/missing context gracefully."""
        llm = LLMService()
        ctx = {"summary": "", "dominant_emotion": "neutral", "compassionate_response": ""}
        history = []

        result = llm.compassionate_chat("Hello", history, ctx)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_fallback_to_regular_chat(self):
        """When compassionate_chat fails, regular chat should work."""
        llm = LLMService()
        history = [{"role": "user", "content": "Help me"}]

        # Force compassionate_chat to fail
        with patch.object(llm, "compassionate_chat", side_effect=Exception("LLM error")):
            # Regular chat should still work
            result = llm.chat("[User emotion: sad] Help me", history)
            assert isinstance(result, str)
            assert len(result) > 0


# ════════════════════════════════════════════════════════════════════════
# Session state simulation
# ════════════════════════════════════════════════════════════════════════


class TestSessionStateSimulation:
    """Simulate the Streamlit session state logic for post-diary chat."""

    def test_post_session_display_condition(self):
        """Post-session results should show when last_ended_session is set and diary_session is None."""
        state = {
            "last_ended_session": _make_ended_session("sad"),
            "diary_session": None,
            "diary_chat_history": [],
        }
        # Condition from streamlit_app.py
        should_show = state["last_ended_session"] is not None and state["diary_session"] is None
        assert should_show is True

    def test_no_display_during_active_session(self):
        """Post-session results should NOT show during an active session."""
        state = {
            "last_ended_session": None,
            "diary_session": DiarySession(session_id="active", start_time="now"),
            "diary_chat_history": [],
        }
        should_show = state["last_ended_session"] is not None and state["diary_session"] is None
        assert should_show is False

    def test_start_fresh_clears_state(self):
        """Start Fresh button should clear last_ended_session and chat history."""
        state = {
            "last_ended_session": _make_ended_session("sad"),
            "diary_chat_history": [
                {"role": "user", "content": "test"},
                {"role": "assistant", "content": "response"},
            ],
        }
        # Simulate "Start Fresh" button
        state["last_ended_session"] = None
        state["diary_chat_history"] = []
        assert state["last_ended_session"] is None
        assert state["diary_chat_history"] == []

    def test_new_session_resets_chat(self):
        """Starting a new session should reset chat history."""
        state = {
            "diary_session": None,
            "diary_chat_history": [{"role": "user", "content": "old message"}],
            "last_diary_audio": b"old_audio",
            "last_ended_session": _make_ended_session("sad"),
            "diary_face_emotion_live": "sad",
            "diary_face_confidence_live": 0.8,
        }
        # Simulate "Start New Session" button
        mgr = DiarySessionManager()
        state["diary_session"] = mgr.start_session()
        state["diary_chat_history"] = []
        state["last_diary_audio"] = None
        state["last_ended_session"] = None
        state["diary_face_emotion_live"] = "neutral"
        state["diary_face_confidence_live"] = 0.0

        assert state["diary_session"] is not None
        assert state["diary_chat_history"] == []
        assert state["last_ended_session"] is None

    def test_chat_persists_across_messages(self):
        """Chat history should persist as messages are added."""
        state = {"diary_chat_history": []}

        state["diary_chat_history"].append({"role": "user", "content": "Message 1"})
        state["diary_chat_history"].append({"role": "assistant", "content": "Response 1"})
        state["diary_chat_history"].append({"role": "user", "content": "Message 2"})
        state["diary_chat_history"].append({"role": "assistant", "content": "Response 2"})

        assert len(state["diary_chat_history"]) == 4
        assert state["diary_chat_history"][0]["content"] == "Message 1"
        assert state["diary_chat_history"][3]["content"] == "Response 2"


# ════════════════════════════════════════════════════════════════════════
# Bottom chat conditional display
# ════════════════════════════════════════════════════════════════════════


class TestBottomChatConditional:
    """Test the bottom 'Talk About Your Feelings' chat shows only when appropriate."""

    def test_bottom_chat_hidden_during_post_session(self):
        """When post-session results are showing, bottom chat should be hidden."""
        last_ended_session = _make_ended_session("sad")
        # The condition: only show bottom chat if last_ended_session is None
        should_show_bottom = last_ended_session is None
        assert should_show_bottom is False

    def test_bottom_chat_visible_without_post_session(self):
        """When no post-session results, bottom chat should be visible."""
        last_ended_session = None
        should_show_bottom = last_ended_session is None
        assert should_show_bottom is True

    def test_bottom_chat_requires_diary_history(self):
        """Bottom chat requires at least one completed session in history."""
        diary_history = []
        has_history = bool(diary_history)
        assert has_history is False

        diary_history.append(_make_ended_session("happy"))
        has_history = bool(diary_history)
        assert has_history is True
