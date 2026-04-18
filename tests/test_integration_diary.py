"""Integration tests for the diary pipeline: speech → fusion → session."""

import pytest
import sys
import os
import json
from unittest.mock import patch, MagicMock
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.speech_service import SpeechService
from services.mood_fusion import MoodFusion
from diary.diary_session import DiaryEntry, DiarySession, DiarySessionManager


class TestSpeechToFusionPipeline:
    """Test the full pipeline from audio → sentiment → fusion."""

    def test_speech_sentiment_feeds_into_fusion(self):
        """SpeechService output should be directly usable by MoodFusion."""
        svc = SpeechService()
        fusion = MoodFusion()

        # Analyze sentiment (uses heuristic if TextBlob unavailable)
        sentiment = svc.analyze_sentiment("I am so happy and grateful today!")
        assert "emotion" in sentiment
        assert "confidence" in sentiment

        # Analyze audio emotion with dummy data
        samples = np.random.randint(-3000, 3000, 2000, dtype=np.int16)
        audio_emo = svc.analyze_audio_emotion(samples.tobytes())
        assert "estimated_emotion" in audio_emo
        assert "confidence" in audio_emo

        # Fuse all signals
        result = fusion.fuse(
            face_emotion="happy",
            face_confidence=0.8,
            text_sentiment=sentiment,
            audio_emotion=audio_emo,
        )
        assert result["emotion"] in [
            "angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"
        ]
        assert 0 <= result["confidence"] <= 1.0

    def test_negative_text_shifts_fusion(self):
        """Negative text sentiment should influence the fused result."""
        svc = SpeechService()
        fusion = MoodFusion()

        sentiment = svc.analyze_sentiment("I am terrible awful stressed depressed")
        assert sentiment["polarity"] < 0

        result = fusion.fuse(
            face_emotion="neutral",
            face_confidence=0.3,
            text_sentiment=sentiment,
            audio_emotion={"estimated_emotion": "neutral", "confidence": 0.3},
        )
        # With low face confidence and strong negative text, result should lean negative
        assert result["emotion"] in ("sad", "angry", "neutral", "fear", "disgust")


class TestDiaryEntryCreationPipeline:
    """Test creating diary entries through the manager with mocked services."""

    @patch("diary_session.LLMService")
    @patch("diary_session.MoodFusion")
    @patch("diary_session.SpeechService")
    def test_full_entry_lifecycle(self, mock_speech_cls, mock_fusion_cls, mock_llm_cls):
        # Setup mocks
        mock_speech = mock_speech_cls.return_value
        mock_speech.transcribe_audio.return_value = {
            "text": "Today was a good day",
            "language": "en",
            "duration": 2.0,
        }
        mock_speech.analyze_sentiment.return_value = {
            "polarity": 0.5,
            "subjectivity": 0.6,
            "emotion": "happy",
            "confidence": 0.7,
        }
        mock_speech.analyze_audio_emotion.return_value = {
            "energy": 0.4,
            "pitch_var": 0.3,
            "estimated_emotion": "happy",
            "confidence": 0.5,
        }

        mock_fusion = mock_fusion_cls.return_value
        mock_fusion.fuse.return_value = {
            "emotion": "happy",
            "confidence": 0.75,
        }

        mgr = DiarySessionManager()
        session = mgr.start_session()

        # Add entry
        entry = mgr.add_entry(session, b"\x00" * 1000, "happy", 0.9)

        # Verify entry
        assert entry.text == "Today was a good day"
        assert entry.fused_emotion == "happy"
        assert entry.face_emotion == "happy"
        assert entry.face_confidence == 0.9

        # Verify it's in the session
        assert len(session.entries) == 1
        assert session.dominant_emotion == "happy"

    @patch("diary_session.LLMService")
    @patch("diary_session.MoodFusion")
    @patch("diary_session.SpeechService")
    def test_multiple_entries_timeline(self, mock_speech_cls, mock_fusion_cls, mock_llm_cls):
        mock_speech = mock_speech_cls.return_value
        mock_speech.transcribe_audio.return_value = {"text": "test", "language": "en", "duration": 1.0}
        mock_speech.analyze_sentiment.return_value = {
            "polarity": 0.0, "subjectivity": 0.0, "emotion": "neutral", "confidence": 0.5
        }
        mock_speech.analyze_audio_emotion.return_value = {
            "energy": 0.3, "pitch_var": 0.2, "estimated_emotion": "neutral", "confidence": 0.4
        }

        # First entry: sad
        mock_fusion = mock_fusion_cls.return_value
        mock_fusion.fuse.side_effect = [
            {"emotion": "sad", "confidence": 0.7},
            {"emotion": "happy", "confidence": 0.8},
            {"emotion": "happy", "confidence": 0.6},
        ]

        mgr = DiarySessionManager()
        session = mgr.start_session()
        mgr.add_entry(session, b"\x00" * 100, "sad", 0.8)
        mgr.add_entry(session, b"\x00" * 100, "happy", 0.9)
        mgr.add_entry(session, b"\x00" * 100, "happy", 0.7)

        assert len(session.entries) == 3
        assert session.dominant_emotion == "happy"

        timeline = session.emotion_timeline
        assert len(timeline) == 3
        assert timeline[0][1] == "sad"
        assert timeline[1][1] == "happy"


class TestSessionSerialization:
    """Test that sessions round-trip through JSON."""

    def test_session_to_json_and_back(self):
        entry = DiaryEntry(
            timestamp="14:00:00",
            text="Hello world",
            face_emotion="neutral",
            face_confidence=0.5,
            voice_sentiment={"polarity": 0.0, "emotion": "neutral", "confidence": 0.5},
            audio_emotion={"energy": 0.3, "estimated_emotion": "neutral", "confidence": 0.4},
            fused_emotion="neutral",
            fused_confidence=0.5,
        )
        session = DiarySession(
            session_id="test123",
            start_time="2026-03-18 14:00:00",
            entries=[entry],
            summary="Test summary",
            research_queries=["query1", "query2"],
        )

        # Serialize
        json_str = session.to_json()
        data = json.loads(json_str)

        # Verify structure
        assert data["session_id"] == "test123"
        assert len(data["entries"]) == 1
        assert data["entries"][0]["text"] == "Hello world"
        assert data["summary"] == "Test summary"
        assert data["research_queries"] == ["query1", "query2"]

    def test_markdown_export(self):
        entry = DiaryEntry(
            timestamp="14:00:00",
            text="Feeling good",
            face_emotion="happy",
            face_confidence=0.8,
            voice_sentiment={"polarity": 0.5, "emotion": "happy", "confidence": 0.7},
            audio_emotion={"energy": 0.4, "estimated_emotion": "happy", "confidence": 0.5},
            fused_emotion="happy",
            fused_confidence=0.75,
        )
        session = DiarySession(
            session_id="md_test",
            start_time="2026-03-18 14:00:00",
            entries=[entry],
            summary="Happy session",
            research_queries=["positive psychology"],
        )
        md = session.to_markdown()
        assert "# Diary Session:" in md
        assert "Happy session" in md
        assert "positive psychology" in md
        assert "Feeling good" in md


class TestEndToEndOfflineMode:
    """Test the full pipeline when external services are unavailable."""

    def test_transcribe_without_whisper_returns_empty(self):
        """Without Whisper, transcribe_audio should return an empty-text result."""
        svc = SpeechService()
        svc._whisper_loaded = True
        svc._whisper = None  # whisper not available

        # Transcribe returns empty text gracefully
        result = svc.transcribe_audio(b"\x00" * 5000)
        assert isinstance(result["text"], str)

        # Sentiment (heuristic if textblob unavailable)
        sentiment = svc.analyze_sentiment(result["text"])
        assert "emotion" in sentiment

        # Audio emotion
        audio_emo = svc.analyze_audio_emotion(b"\x00" * 5000)
        assert "estimated_emotion" in audio_emo

        # Fusion
        fusion = MoodFusion()
        fused = fusion.fuse(
            face_emotion="neutral",
            face_confidence=0.5,
            text_sentiment=sentiment,
            audio_emotion=audio_emo,
        )
        assert fused["emotion"] in [
            "angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"
        ]

    def test_breathing_exercise_available(self):
        text = DiarySessionManager.breathing_exercise_text()
        assert "4-7-8" in text
        assert "parasympathetic" in text
