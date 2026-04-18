"""Tests for diary_session.py — DiaryEntry, DiarySession, DiarySessionManager."""

import pytest
import sys
import os
import json
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from diary.diary_session import DiaryEntry, DiarySession, DiarySessionManager


# ── Fixtures ─────────────────────────────────────────────────

def make_entry(**overrides):
    """Factory for DiaryEntry with sensible defaults."""
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
        session_id="abc123",
        start_time="2026-03-18 10:00:00",
        entries=[],
        summary="",
        research_queries=[],
    )
    defaults.update(overrides)
    return DiarySession(**defaults)


# ── DiaryEntry Tests ─────────────────────────────────────────

class TestDiaryEntry:
    def test_create_entry(self):
        entry = make_entry()
        assert entry.face_emotion == "happy"
        assert entry.fused_emotion == "happy"
        assert entry.fused_confidence == 0.75

    def test_to_dict(self):
        entry = make_entry()
        d = entry.to_dict()
        assert isinstance(d, dict)
        assert d["timestamp"] == "12:30:00"
        assert d["text"] == "I feel great today"
        assert d["face_emotion"] == "happy"
        assert d["fused_emotion"] == "happy"
        assert isinstance(d["voice_sentiment"], dict)
        assert isinstance(d["audio_emotion"], dict)

    def test_to_markdown(self):
        entry = make_entry()
        md = entry.to_markdown()
        assert "### Entry at 12:30:00" in md
        assert "I feel great today" in md
        assert "happy" in md
        assert "85%" in md

    def test_to_markdown_contains_fused(self):
        entry = make_entry(fused_emotion="sad", fused_confidence=0.6)
        md = entry.to_markdown()
        assert "sad" in md
        assert "60%" in md


# ── DiarySession Tests ───────────────────────────────────────

class TestDiarySession:
    def test_create_session(self):
        session = make_session()
        assert session.session_id == "abc123"
        assert session.entries == []
        assert session.summary == ""

    def test_to_dict(self):
        entry = make_entry()
        session = make_session(entries=[entry])
        d = session.to_dict()
        assert d["session_id"] == "abc123"
        assert len(d["entries"]) == 1
        assert d["entries"][0]["face_emotion"] == "happy"

    def test_to_dict_with_dict_entries(self):
        """Session can hold raw dicts (from deserialization)."""
        session = make_session(entries=[{"fused_emotion": "sad", "text": "test"}])
        d = session.to_dict()
        assert d["entries"][0]["fused_emotion"] == "sad"

    def test_to_json(self):
        session = make_session(entries=[make_entry()])
        j = session.to_json()
        data = json.loads(j)
        assert data["session_id"] == "abc123"
        assert len(data["entries"]) == 1

    def test_to_markdown(self):
        session = make_session(
            entries=[make_entry()],
            summary="Great session!",
            research_queries=["query 1", "query 2"],
        )
        md = session.to_markdown()
        assert "# Diary Session: abc123" in md
        assert "Great session!" in md
        assert "query 1" in md
        assert "### Entry at" in md

    def test_to_markdown_empty(self):
        session = make_session()
        md = session.to_markdown()
        assert "abc123" in md
        assert "**Entries:** 0" in md

    def test_dominant_emotion_empty(self):
        session = make_session()
        assert session.dominant_emotion == "neutral"

    def test_dominant_emotion_single(self):
        session = make_session(entries=[make_entry(fused_emotion="sad")])
        assert session.dominant_emotion == "sad"

    def test_dominant_emotion_multiple(self):
        entries = [
            make_entry(fused_emotion="happy"),
            make_entry(fused_emotion="happy"),
            make_entry(fused_emotion="sad"),
        ]
        session = make_session(entries=entries)
        assert session.dominant_emotion == "happy"

    def test_dominant_emotion_with_dict_entries(self):
        session = make_session(entries=[
            {"fused_emotion": "angry"},
            {"fused_emotion": "angry"},
            {"fused_emotion": "happy"},
        ])
        assert session.dominant_emotion == "angry"

    def test_emotion_timeline(self):
        entries = [
            make_entry(timestamp="10:00:00", fused_emotion="happy", fused_confidence=0.8),
            make_entry(timestamp="10:05:00", fused_emotion="sad", fused_confidence=0.6),
        ]
        session = make_session(entries=entries)
        timeline = session.emotion_timeline
        assert len(timeline) == 2
        assert timeline[0] == ("10:00:00", "happy", 0.8)
        assert timeline[1] == ("10:05:00", "sad", 0.6)

    def test_emotion_timeline_with_dict_entries(self):
        session = make_session(entries=[
            {"timestamp": "11:00:00", "fused_emotion": "neutral", "fused_confidence": 0.5}
        ])
        timeline = session.emotion_timeline
        assert len(timeline) == 1
        assert timeline[0] == ("11:00:00", "neutral", 0.5)

    def test_emotion_timeline_empty(self):
        session = make_session()
        assert session.emotion_timeline == []


# ── DiarySessionManager Tests ────────────────────────────────

class TestDiarySessionManager:
    @patch("diary_session.SpeechService")
    @patch("diary_session.MoodFusion")
    @patch("diary_session.LLMService")
    def test_start_session(self, mock_llm, mock_fusion, mock_speech):
        mgr = DiarySessionManager()
        session = mgr.start_session()
        assert isinstance(session, DiarySession)
        assert len(session.session_id) == 8
        assert session.entries == []

    @patch("diary_session.SpeechService")
    @patch("diary_session.MoodFusion")
    @patch("diary_session.LLMService")
    def test_add_entry(self, mock_llm_cls, mock_fusion_cls, mock_speech_cls):
        mock_speech = mock_speech_cls.return_value
        mock_speech.transcribe_audio.return_value = {"text": "hello world", "language": "en", "duration": 1.0}
        mock_speech.analyze_sentiment.return_value = {
            "polarity": 0.5, "subjectivity": 0.5, "emotion": "happy", "confidence": 0.7
        }
        mock_speech.analyze_audio_emotion.return_value = {
            "energy": 0.4, "pitch_var": 0.3, "estimated_emotion": "happy", "confidence": 0.5
        }

        mock_fusion = mock_fusion_cls.return_value
        mock_fusion.fuse.return_value = {"emotion": "happy", "confidence": 0.75}

        mgr = DiarySessionManager()
        session = mgr.start_session()
        entry = mgr.add_entry(session, b"\x00" * 1000, face_emotion="happy", face_confidence=0.9)

        assert isinstance(entry, DiaryEntry)
        assert entry.text == "hello world"
        assert entry.fused_emotion == "happy"
        assert len(session.entries) == 1

    @patch("diary_session.SpeechService")
    @patch("diary_session.MoodFusion")
    @patch("diary_session.LLMService")
    def test_add_entry_respects_max(self, mock_llm_cls, mock_fusion_cls, mock_speech_cls):
        mock_speech = mock_speech_cls.return_value
        mock_speech.transcribe_audio.return_value = {"text": "x", "language": "en", "duration": 0.5}
        mock_speech.analyze_sentiment.return_value = {
            "polarity": 0.0, "subjectivity": 0.0, "emotion": "neutral", "confidence": 0.5
        }
        mock_speech.analyze_audio_emotion.return_value = {
            "energy": 0.3, "pitch_var": 0.2, "estimated_emotion": "neutral", "confidence": 0.4
        }
        mock_fusion = mock_fusion_cls.return_value
        mock_fusion.fuse.return_value = {"emotion": "neutral", "confidence": 0.5}

        mgr = DiarySessionManager()
        session = mgr.start_session()

        # Fill to MAX_DIARY_ENTRIES
        from config import MAX_DIARY_ENTRIES
        for _ in range(MAX_DIARY_ENTRIES + 5):
            mgr.add_entry(session, b"\x00" * 100)

        assert len(session.entries) == MAX_DIARY_ENTRIES


class TestBreathingExercise:
    def test_returns_text(self):
        text = DiarySessionManager.breathing_exercise_text()
        assert "Breathe IN" in text
        assert "4 seconds" in text
        assert "7 seconds" in text
        assert "8 seconds" in text
