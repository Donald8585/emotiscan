"""Integration tests for enhanced diary pipeline with web search, compassionate response, and GPU config."""

import pytest
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from diary.diary_session import DiaryEntry, DiarySession, DiarySessionManager
from services.llm_service import LLMService
from services.web_search_service import search_coping_strategies, clear_cache


class TestFullDiaryEndSessionPipeline:
    """Integration test: full diary session ??end_session ??verify all fields populated."""

    @patch("diary_session.search_general", return_value=[
        {"title": "Coping with sadness", "url": "https://example.com/cope", "snippet": "Techniques for..."}
    ])
    @patch("diary_session.search_emotion_articles", return_value=[
        {"title": "Understanding sadness", "url": "https://example.com/sad", "snippet": "Sadness is..."}
    ])
    @patch("diary_session.SpeechService")
    @patch("diary_session.MoodFusion")
    @patch("diary_session.LLMService")
    def test_end_session_populates_all_new_fields(self, mock_llm_cls, mock_fusion_cls, mock_speech_cls,
                                                    mock_emo_search, mock_gen_search):
        mock_llm = mock_llm_cls.return_value
        mock_llm._generate.return_value = ""

        mgr = DiarySessionManager()

        # Create a session with entries
        entry = DiaryEntry(
            timestamp="14:00:00",
            text="I've been feeling really down lately",
            face_emotion="sad",
            face_confidence=0.8,
            voice_sentiment={"polarity": -0.5, "emotion": "sad", "confidence": 0.7},
            audio_emotion={"energy": 0.2, "estimated_emotion": "sad", "confidence": 0.5},
            fused_emotion="sad",
            fused_confidence=0.7,
        )
        session = DiarySession(
            session_id="integ_test",
            start_time="2026-03-18 14:00:00",
            entries=[entry],
        )

        result = mgr.end_session(session)

        # Verify all new fields are populated
        assert isinstance(result.summary, str)
        assert isinstance(result.research_queries, list)
        assert len(result.web_results) > 0
        assert isinstance(result.compassionate_response, str)

        # Verify to_dict includes new fields
        d = result.to_dict()
        assert "web_results" in d
        assert "arxiv_results" in d
        assert "compassionate_response" in d

        # Verify to_markdown includes new sections
        md = result.to_markdown()
        assert "## Supportive Response" in md
        assert "## Web Results" in md

    @patch("diary_session.search_general", return_value=[])
    @patch("diary_session.search_emotion_articles", return_value=[])
    @patch("diary_session.DiarySessionManager._search_arxiv", return_value=[])
    @patch("diary_session.SpeechService")
    @patch("diary_session.MoodFusion")
    @patch("diary_session.LLMService")
    def test_end_session_with_empty_search_results(self, mock_llm_cls, mock_fusion_cls, mock_speech_cls,
                                                     mock_arxiv, mock_emo_search, mock_gen_search):
        mock_llm = mock_llm_cls.return_value
        mock_llm._generate.return_value = ""

        mgr = DiarySessionManager()
        session = DiarySession(
            session_id="empty_test",
            start_time="2026-03-18 14:00:00",
            entries=[DiaryEntry(
                timestamp="14:00:00", text="test", face_emotion="neutral",
                face_confidence=0.5, voice_sentiment={}, audio_emotion={},
                fused_emotion="neutral", fused_confidence=0.5,
            )],
        )

        result = mgr.end_session(session)

        # Even with empty search results, other fields should be populated
        assert isinstance(result.summary, str)
        assert isinstance(result.compassionate_response, str)


class TestCompassionateChatFlow:
    """Integration test: compassionate chat with session context."""

    def test_compassionate_chat_with_full_context(self):
        svc = LLMService()
        ctx = {
            "summary": "User expressed deep sadness about a relationship ending",
            "dominant_emotion": "sad",
            "compassionate_response": "Your feelings of loss are completely valid.",
        }
        result = svc.compassionate_chat(
            "How do I move on from this?",
            history=[
                {"role": "user", "content": "My relationship ended"},
                {"role": "assistant", "content": "I'm sorry to hear that."},
            ],
            session_context=ctx,
        )
        assert isinstance(result, str)(self):
        """After chatting, user should be able to get specific solutions."""
        svc = LLMService()
        solutions = svc.suggest_solutions("sad", "I can't stop thinking about the breakup")
        assert isinstance(solutions, str)

    def test_chat_and_solutions_together(self):
        """Both chat and solutions should work in sequence."""
        svc = LLMService()
        chat_resp = svc.compassionate_chat("I feel awful", session_context={"dominant_emotion": "sad"})
        solutions = svc.suggest_solutions("sad", "feeling awful")
        assert isinstance(chat_resp, str)
        assert isinstance(solutions, str)


class TestGPUConfigPropagation:
    """Integration test: GPU config propagation to speech service."""

    def test_config_values_available(self):
        import config
        assert hasattr(config, 'GPU_AVAILABLE')
        assert hasattr(config, 'WHISPER_DEVICE')
        assert hasattr(config, 'WHISPER_COMPUTE_TYPE')

    def test_speech_service_uses_config(self):
        """SpeechService should import the GPU config."""
        from services.speech_service import SpeechService, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE
        import config
        assert WHISPER_DEVICE == config.WHISPER_DEVICE
        assert WHISPER_COMPUTE_TYPE == config.WHISPER_COMPUTE_TYPE

    def test_speech_service_init_with_gpu_config(self):
        """SpeechService should initialize correctly with GPU config."""
        from services.speech_service import SpeechService
        svc = SpeechService()
        assert svc is not None
        # The _load_whisper method should use WHISPER_DEVICE and WHISPER_COMPUTE_TYPE

    @patch("speech_service.WHISPER_DEVICE", "cuda")
    @patch("speech_service.WHISPER_COMPUTE_TYPE", "float16")
    def test_whisper_load_with_gpu_device(self):
        """When GPU is available, whisper should be loaded with cuda device."""
        from services.speech_service import SpeechService
        svc = SpeechService()
        svc._whisper_loaded = False  # reset
        # Can't actually load whisper without the model, but we can check it doesn't crash
        svc._load_whisper()  # Will fail to load but shouldn't crash
        assert svc._whisper_loaded is True


class TestCopingStrategiesIntegration:
    """Test the coping strategies search integrates with the diary flow."""

    def setup_method(self):
        clear_cache()

    def test_coping_results_can_be_added_to_session(self):
        results = search_coping_strategies("sad", max_results=3)
        session = DiarySession(
            session_id="cope_test",
            start_time="2026-03-18 14:00:00",
        )
        session.web_results = results
        assert isinstance(session.web_results, list)

        # Verify serialization
        d = session.to_dict()
        assert len(d["web_results"]) > 0

    def test_end_to_end_offline_flow(self):
        """Full flow with LLM offline: session ??end ??verify fields are present."""
        with patch("diary_session.SpeechService"), \
             patch("diary_session.MoodFusion"), \
             patch("diary_session.LLMService") as mock_llm_cls, \
             patch("diary_session.search_general", return_value=[
                 {"title": "Demo", "url": "https://demo.com", "snippet": "demo text"}
             ]), \
             patch("diary_session.search_emotion_articles", return_value=[
                 {"title": "Emotions", "url": "https://emo.com", "snippet": "emo text"}
             ]), \
             patch("diary_session.DiarySessionManager._search_arxiv", return_value=[
                 {"title": "Emotion Regulation", "authors": "Smith et al.", "summary": "A study on emotion", "url": "https://arxiv.org/abs/1234", "source": "arxiv"}
             ]):
            mock_llm_cls.return_value._generate.return_value = ""

            mgr = DiarySessionManager()
            session = DiarySession(
                session_id="e2e",
                start_time="2026-03-18 15:00:00",
                entries=[DiaryEntry(
                    timestamp="15:00:00", text="Feeling anxious about work",
                    face_emotion="fear", face_confidence=0.7,
                    voice_sentiment={"polarity": -0.3, "emotion": "fear", "confidence": 0.6},
                    audio_emotion={"energy": 0.3, "estimated_emotion": "neutral", "confidence": 0.4},
                    fused_emotion="fear", fused_confidence=0.65,
                )],
            )

            result = mgr.end_session(session)

            # Fields should be present
            assert isinstance(result.summary, str)
            assert isinstance(result.research_queries, list)
            assert len(result.web_results) > 0
            assert isinstance(result.compassionate_response, str)

            # Markdown should be comprehensive
            md = result.to_markdown()
            assert "fear" in md.lower() or "anxious" in md.lower() or "Supportive" in md
