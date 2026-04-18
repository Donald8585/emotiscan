"""Tests for speech_service.py — STT, sentiment, audio emotion heuristics."""

import pytest
import sys
import os
import struct
import numpy as np
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.speech_service import SpeechService


class TestSpeechServiceInit:
    def test_creates_instance(self):
        svc = SpeechService()
        assert svc is not None
        assert svc._whisper is None
        assert svc._whisper_loaded is False

    def test_custom_params(self):
        svc = SpeechService(whisper_model="tiny", sample_rate=8000)
        assert svc._whisper_model_name == "tiny"
        assert svc._sample_rate == 8000


class TestTranscribeAudio:
    def test_empty_bytes_returns_empty(self):
        svc = SpeechService()
        result = svc.transcribe_audio(b"")
        assert result["text"] == ""
        assert result["language"] == "unknown"
        assert result["duration"] == 0.0

    def test_none_bytes_returns_empty(self):
        svc = SpeechService()
        result = svc.transcribe_audio(None)
        assert result["text"] == ""

    def test_whisper_unavailable_returns_empty_text(self):
        svc = SpeechService()
        svc._whisper_loaded = True
        svc._whisper = None  # no whisper available
        result = svc.transcribe_audio(b"\x00" * 1000)
        assert "text" in result
        assert isinstance(result["text"], str)
        assert result["language"] == "unknown"

    @patch("speech_service.SpeechService._load_whisper")
    def test_whisper_transcription(self, mock_load):
        svc = SpeechService()
        svc._whisper_loaded = True

        mock_segment = MagicMock()
        mock_segment.text = "hello world"
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.duration = 1.5

        mock_whisper = MagicMock()
        mock_whisper.transcribe.return_value = ([mock_segment], mock_info)
        svc._whisper = mock_whisper

        result = svc.transcribe_audio(b"\x00" * 500)
        assert result["text"] == "hello world"
        assert result["language"] == "en"
        assert result["duration"] == 1.5

    @patch("speech_service.SpeechService._load_whisper")
    def test_whisper_exception_falls_back_to_demo(self, mock_load):
        svc = SpeechService()
        svc._whisper_loaded = True

        mock_whisper = MagicMock()
        mock_whisper.transcribe.side_effect = RuntimeError("model crash")
        svc._whisper = mock_whisper

        result = svc.transcribe_audio(b"\x00" * 500)
        assert "text" in result


class TestAnalyzeSentiment:
    def test_empty_text_returns_neutral(self):
        svc = SpeechService()
        result = svc.analyze_sentiment("")
        assert result["emotion"] == "neutral"
        assert result["polarity"] == 0.0

    def test_none_text_returns_neutral(self):
        svc = SpeechService()
        result = svc.analyze_sentiment(None)
        assert result["emotion"] == "neutral"

    def test_whitespace_text_returns_neutral(self):
        svc = SpeechService()
        result = svc.analyze_sentiment("   ")
        assert result["emotion"] == "neutral"

    def test_positive_text(self):
        svc = SpeechService()
        result = svc.analyze_sentiment("I am so happy and wonderful today!")
        assert result["polarity"] > 0
        assert result["emotion"] in ("happy", "surprise")
        assert 0 <= result["confidence"] <= 1.0

    def test_negative_text(self):
        svc = SpeechService()
        result = svc.analyze_sentiment("I am sad and terrible")
        assert result["polarity"] < 0
        assert result["emotion"] in ("sad", "angry")

    def test_result_keys(self):
        svc = SpeechService()
        result = svc.analyze_sentiment("test text")
        assert "polarity" in result
        assert "subjectivity" in result
        assert "emotion" in result
        assert "confidence" in result

    @patch("speech_service.SpeechService._heuristic_sentiment")
    def test_textblob_import_fallback(self, mock_heuristic):
        """When TextBlob import fails, should use heuristic."""
        mock_heuristic.return_value = (0.5, 0.5)
        svc = SpeechService()
        with patch.dict("sys.modules", {"textblob": None}):
            # Force reimport to fail
            result = svc.analyze_sentiment("happy text")
        # Should still return a valid result
        assert "emotion" in result


class TestHeuristicSentiment:
    def test_positive_keywords(self):
        polarity, subjectivity = SpeechService._heuristic_sentiment("I am happy and grateful")
        assert polarity > 0

    def test_negative_keywords(self):
        polarity, subjectivity = SpeechService._heuristic_sentiment("I am sad and stressed")
        assert polarity < 0

    def test_mixed_keywords(self):
        polarity, _ = SpeechService._heuristic_sentiment("happy but sad")
        assert -1.0 <= polarity <= 1.0

    def test_no_keywords(self):
        polarity, subjectivity = SpeechService._heuristic_sentiment("the quick brown fox")
        assert polarity == 0.0
        assert subjectivity == 0.3


class TestPolarityToEmotion:
    def test_very_negative(self):
        assert SpeechService._polarity_to_emotion(-0.8) == "angry"

    def test_slightly_negative(self):
        assert SpeechService._polarity_to_emotion(-0.3) == "sad"

    def test_neutral_range(self):
        assert SpeechService._polarity_to_emotion(0.0) == "neutral"
        assert SpeechService._polarity_to_emotion(0.1) == "neutral"
        assert SpeechService._polarity_to_emotion(-0.1) == "neutral"

    def test_positive(self):
        assert SpeechService._polarity_to_emotion(0.4) == "happy"

    def test_very_positive(self):
        assert SpeechService._polarity_to_emotion(0.8) == "surprise"


class TestAnalyzeAudioEmotion:
    def test_empty_bytes(self):
        svc = SpeechService()
        result = svc.analyze_audio_emotion(b"")
        assert result["estimated_emotion"] == "neutral"
        assert result["confidence"] == 0.3

    def test_none_bytes(self):
        svc = SpeechService()
        result = svc.analyze_audio_emotion(None)
        assert result["estimated_emotion"] == "neutral"

    def test_small_bytes(self):
        svc = SpeechService()
        result = svc.analyze_audio_emotion(b"\x00" * 50)
        assert result["estimated_emotion"] == "neutral"

    def test_silent_audio_is_sad_or_neutral(self):
        """Very quiet audio should give sad or neutral."""
        svc = SpeechService()
        # Near-silence: very small int16 values
        samples = np.zeros(2000, dtype=np.int16)
        raw = samples.tobytes()
        result = svc.analyze_audio_emotion(raw)
        assert result["estimated_emotion"] in ("sad", "neutral")
        assert result["energy"] < 0.2

    def test_loud_audio_is_energetic(self):
        """Loud audio should indicate high energy."""
        svc = SpeechService()
        # Loud signal
        samples = np.full(2000, 20000, dtype=np.int16)
        raw = samples.tobytes()
        result = svc.analyze_audio_emotion(raw)
        assert result["energy"] > 0.3

    def test_result_keys(self):
        svc = SpeechService()
        samples = np.random.randint(-5000, 5000, 2000, dtype=np.int16)
        result = svc.analyze_audio_emotion(samples.tobytes())
        assert "energy" in result
        assert "pitch_var" in result
        assert "estimated_emotion" in result
        assert "confidence" in result

    def test_wav_header_handling(self):
        """Audio with RIFF WAV header should skip the header."""
        svc = SpeechService()
        samples = np.random.randint(-3000, 3000, 2000, dtype=np.int16)
        raw = samples.tobytes()
        # Create a minimal WAV header (44 bytes)
        wav_header = b'RIFF' + b'\x00' * 40
        wav_bytes = wav_header + raw
        result = svc.analyze_audio_emotion(wav_bytes)
        assert "estimated_emotion" in result


class TestBytesToSamples:
    def test_raw_pcm(self):
        original = np.array([100, -200, 300], dtype=np.int16)
        raw = original.tobytes()
        samples = SpeechService._bytes_to_samples(raw)
        np.testing.assert_array_equal(samples, original)

    def test_wav_header_stripped(self):
        original = np.array([500, -500, 1000], dtype=np.int16)
        raw = original.tobytes()
        wav = b'RIFF' + b'\x00' * 40 + raw
        samples = SpeechService._bytes_to_samples(wav)
        np.testing.assert_array_equal(samples, original)

    def test_empty_returns_none(self):
        result = SpeechService._bytes_to_samples(b'RIFF' + b'\x00' * 40)
        assert result is None

    def test_non_pcm_fallback(self):
        """Non-PCM, non-WAV bytes: parsed as int16 pairs, odd byte dropped."""
        data = bytes([128, 200, 50, 128, 100])
        samples = SpeechService._bytes_to_samples(data)
        assert samples is not None
        assert len(samples) == 2  # 5 bytes → 2 int16 samples (last byte dropped)


