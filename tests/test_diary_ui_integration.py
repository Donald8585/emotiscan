"""Integration tests for Diary UI (v2): async detection, st.audio_input, session state, post-session display."""

import pytest
import sys
import os
import threading
from unittest.mock import patch, MagicMock, PropertyMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from diary.diary_session import DiaryEntry, DiarySession, DiarySessionManager
from services.web_search_service import search_coping_strategies
from services.llm_service import LLMService


# ── EmotionVideoProcessor module-level accessibility ────────────


class TestEmotionVideoProcessorModuleLevel:
    """Verify EmotionVideoProcessor is accessible at module level from streamlit_app."""

    def test_module_defines_webrtc_flag(self):
        """streamlit_app should export _webrtc_imports_ok boolean."""
        import ast

        with open(os.path.join(os.path.dirname(__file__), "..", "streamlit_app.py")) as f:
            source = f.read()

        tree = ast.parse(source)
        found_flag = False
        found_class = False
        found_rtc_config = False

        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "_webrtc_imports_ok":
                        found_flag = True
                    if isinstance(target, ast.Name) and target.id == "rtc_config":
                        found_rtc_config = True
            if isinstance(node, ast.ClassDef) and node.name == "EmotionVideoProcessor":
                found_class = True

        assert found_flag, "_webrtc_imports_ok not found at module level"
        assert found_class, "EmotionVideoProcessor class not found at module level"
        assert found_rtc_config, "rtc_config not found at module level"

    def test_class_not_inside_tab_block(self):
        """EmotionVideoProcessor should NOT be indented inside a 'with' block."""
        with open(os.path.join(os.path.dirname(__file__), "..", "streamlit_app.py")) as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            if "class EmotionVideoProcessor" in line:
                indent = len(line) - len(line.lstrip())
                assert indent <= 4, (
                    f"EmotionVideoProcessor at line {i+1} has indent {indent} — "
                    f"should be at module level (indent <= 4)"
                )
                break
        else:
            pytest.fail("EmotionVideoProcessor class definition not found in streamlit_app.py")


# ── Session state variables ─────────────────────────────────────


class TestSessionStateVariables:
    """Verify new session state variables are defined in defaults."""

    def test_defaults_include_new_vars(self):
        with open(os.path.join(os.path.dirname(__file__), "..", "streamlit_app.py")) as f:
            source = f.read()

        required_vars = [
            "last_diary_audio",
            "last_ended_session",
            "diary_face_emotion_live",
            "diary_face_confidence_live",
        ]
        for var in required_vars:
            assert f'"{var}"' in source or f"'{var}'" in source, (
                f"Session state variable '{var}' not found in streamlit_app.py defaults"
            )

    def test_diary_session_var_exists(self):
        with open(os.path.join(os.path.dirname(__file__), "..", "streamlit_app.py")) as f:
            source = f.read()

        assert '"diary_session"' in source
        assert '"diary_history"' in source
        assert '"diary_chat_history"' in source


# ── Auto-processing logic ───────────────────────────────────────


class TestAutoProcessingLogic:
    """Test the auto-processing audio entry logic."""

    @patch("diary_session.LLMService")
    @patch("diary_session.MoodFusion")
    @patch("diary_session.SpeechService")
    def test_add_entry_called_with_audio_bytes(self, mock_speech_cls, mock_fusion_cls, mock_llm_cls):
        mock_speech = mock_speech_cls.return_value
        mock_speech.transcribe_audio.return_value = {
            "text": "I feel really happy today",
            "language": "en",
            "duration": 3.0,
        }
        mock_speech.analyze_sentiment.return_value = {
            "polarity": 0.7, "subjectivity": 0.5, "emotion": "happy", "confidence": 0.8,
        }
        mock_speech.analyze_audio_emotion.return_value = {
            "energy": 0.6, "pitch_var": 0.4, "estimated_emotion": "happy", "confidence": 0.6,
        }
        mock_fusion = mock_fusion_cls.return_value
        mock_fusion.fuse.return_value = {"emotion": "happy", "confidence": 0.85}

        mgr = DiarySessionManager()
        session = mgr.start_session()

        audio_bytes = b"\x00\x01\x02" * 500
        entry = mgr.add_entry(session, audio_bytes, face_emotion="happy", face_confidence=0.9)

        assert entry is not None
        assert entry.fused_emotion == "happy"
        assert entry.text == "I feel really happy today"
        assert len(session.entries) == 1
        mock_speech.transcribe_audio.assert_called_once()

    @patch("diary_session.LLMService")
    @patch("diary_session.MoodFusion")
    @patch("diary_session.SpeechService")
    def test_same_audio_not_reprocessed_logic(self, mock_speech_cls, mock_fusion_cls, mock_llm_cls):
        mock_speech = mock_speech_cls.return_value
        mock_speech.transcribe_audio.return_value = {
            "text": "test", "language": "en", "duration": 1.0,
        }
        mock_speech.analyze_sentiment.return_value = {
            "polarity": 0.0, "subjectivity": 0.0, "emotion": "neutral", "confidence": 0.5,
        }
        mock_speech.analyze_audio_emotion.return_value = {
            "energy": 0.3, "pitch_var": 0.2, "estimated_emotion": "neutral", "confidence": 0.4,
        }
        mock_fusion = mock_fusion_cls.return_value
        mock_fusion.fuse.return_value = {"emotion": "neutral", "confidence": 0.5}

        mgr = DiarySessionManager()
        session = mgr.start_session()

        audio_bytes = b"\x00" * 1000
        last_diary_audio = None

        if audio_bytes is not None and audio_bytes != last_diary_audio:
            last_diary_audio = audio_bytes
            mgr.add_entry(session, audio_bytes, "neutral", 0.5)

        assert len(session.entries) == 1

        if audio_bytes is not None and audio_bytes != last_diary_audio:
            mgr.add_entry(session, audio_bytes, "neutral", 0.5)

        assert len(session.entries) == 1

        new_audio = b"\x01" * 1000
        if new_audio is not None and new_audio != last_diary_audio:
            last_diary_audio = new_audio
            mgr.add_entry(session, new_audio, "happy", 0.7)

        assert len(session.entries) == 2

    @patch("diary_session.LLMService")
    @patch("diary_session.MoodFusion")
    @patch("diary_session.SpeechService")
    def test_face_emotion_passed_to_add_entry(self, mock_speech_cls, mock_fusion_cls, mock_llm_cls):
        mock_speech = mock_speech_cls.return_value
        mock_speech.transcribe_audio.return_value = {
            "text": "test", "language": "en", "duration": 1.0,
        }
        mock_speech.analyze_sentiment.return_value = {
            "polarity": 0.0, "subjectivity": 0.0, "emotion": "neutral", "confidence": 0.5,
        }
        mock_speech.analyze_audio_emotion.return_value = {
            "energy": 0.3, "pitch_var": 0.2, "estimated_emotion": "neutral", "confidence": 0.4,
        }
        mock_fusion = mock_fusion_cls.return_value
        mock_fusion.fuse.return_value = {"emotion": "sad", "confidence": 0.7}

        mgr = DiarySessionManager()
        session = mgr.start_session()

        entry = mgr.add_entry(session, b"\x00" * 500, face_emotion="sad", face_confidence=0.85)

        assert entry.face_emotion == "sad"
        assert entry.face_confidence == 0.85


# ── End session produces all expected fields ────────────────────


class TestEndSessionFields:
    """Test that end_session produces all fields needed for post-session display."""

    @patch("diary_session.search_general", return_value=[])
    @patch("diary_session.search_emotion_articles", return_value=[])
    @patch("diary_session.DiarySessionManager._search_arxiv")
    @patch("diary_session.LLMService")
    @patch("diary_session.MoodFusion")
    @patch("diary_session.SpeechService")
    def test_end_session_has_required_fields(
        self, mock_speech_cls, mock_fusion_cls, mock_llm_cls, mock_arxiv, mock_web_articles, mock_web_general
    ):
        mock_speech = mock_speech_cls.return_value
        mock_speech.transcribe_audio.return_value = {
            "text": "I feel stressed about work",
            "language": "en",
            "duration": 2.0,
        }
        mock_speech.analyze_sentiment.return_value = {
            "polarity": -0.5, "subjectivity": 0.7, "emotion": "sad", "confidence": 0.7,
        }
        mock_speech.analyze_audio_emotion.return_value = {
            "energy": 0.2, "pitch_var": 0.1, "estimated_emotion": "sad", "confidence": 0.5,
        }
        mock_fusion = mock_fusion_cls.return_value
        mock_fusion.fuse.return_value = {"emotion": "sad", "confidence": 0.75}

        mock_llm = mock_llm_cls.return_value
        mock_llm.summarize.return_value = "User expressed stress about work."
        mock_llm.compassionate_response.return_value = "I hear you. Work stress is tough."
        mock_llm.suggest_solutions.return_value = "1. Take breaks\n2. Deep breathing"

        mock_arxiv.return_value = [
            {"title": "Stress and Coping", "url": "https://arxiv.org/abs/1234", "abstract": "..."},
        ]

        mgr = DiarySessionManager()
        session = mgr.start_session()
        mgr.add_entry(session, b"\x00" * 500, "sad", 0.8)

        ended = mgr.end_session(session)

        assert ended.summary != ""
        assert ended.compassionate_response != ""
        assert isinstance(ended.arxiv_results, list)
        assert isinstance(ended.web_results, list)
        assert ended.dominant_emotion == "sad"
        assert ended.session_id is not None

    @patch("diary_session.search_general", return_value=[])
    @patch("diary_session.search_emotion_articles", return_value=[])
    @patch("diary_session.DiarySessionManager._search_arxiv", return_value=[])
    @patch("diary_session.LLMService")
    @patch("diary_session.MoodFusion")
    @patch("diary_session.SpeechService")
    def test_end_session_stores_in_history(
        self, mock_speech_cls, mock_fusion_cls, mock_llm_cls, mock_arxiv, mock_web_articles, mock_web_general
    ):
        mock_speech = mock_speech_cls.return_value
        mock_speech.transcribe_audio.return_value = {
            "text": "test", "language": "en", "duration": 1.0,
        }
        mock_speech.analyze_sentiment.return_value = {
            "polarity": 0.0, "subjectivity": 0.0, "emotion": "neutral", "confidence": 0.5,
        }
        mock_speech.analyze_audio_emotion.return_value = {
            "energy": 0.3, "pitch_var": 0.2, "estimated_emotion": "neutral", "confidence": 0.4,
        }
        mock_fusion = mock_fusion_cls.return_value
        mock_fusion.fuse.return_value = {"emotion": "neutral", "confidence": 0.5}

        mock_llm = mock_llm_cls.return_value
        mock_llm.summarize.return_value = "Neutral session."
        mock_llm.compassionate_response.return_value = "Thanks for sharing."

        mgr = DiarySessionManager()
        session = mgr.start_session()
        mgr.add_entry(session, b"\x00" * 500, "neutral", 0.5)

        ended = mgr.end_session(session)

        diary_history = []
        diary_history.append(ended)
        last_ended_session = ended

        assert len(diary_history) == 1
        assert last_ended_session is ended
        assert isinstance(last_ended_session, DiarySession)
        assert last_ended_session.summary != ""


# ── Coping strategies integration ───────────────────────────────


class TestCopingStrategiesIntegration:
    """Test search_coping_strategies works for post-session display."""

    @patch("duckduckgo_search.DDGS")
    def test_coping_strategies_returns_list(self, mock_ddgs):
        mock_ddgs.return_value.__enter__ = lambda s: s
        mock_ddgs.return_value.__exit__ = lambda *a: None
        mock_ddgs.return_value.text.return_value = [
            {"title": "Coping with sadness", "href": "https://example.com", "body": "Try journaling."},
        ]
        results = search_coping_strategies("sad", max_results=3)
        assert isinstance(results, list)
        assert len(results) > 0

    @patch("duckduckgo_search.DDGS")
    def test_coping_strategies_have_required_keys(self, mock_ddgs):
        mock_ddgs.return_value.__enter__ = lambda s: s
        mock_ddgs.return_value.__exit__ = lambda *a: None
        mock_ddgs.return_value.text.return_value = [
            {"title": "Anger mgmt", "href": "https://example.com/anger", "body": "Breathe deeply."},
        ]
        results = search_coping_strategies("angry", max_results=3)
        for r in results:
            assert "title" in r
            assert "url" in r
            assert "snippet" in r

    @patch("duckduckgo_search.DDGS")
    def test_coping_strategies_all_emotions(self, mock_ddgs):
        mock_ddgs.return_value.__enter__ = lambda s: s
        mock_ddgs.return_value.__exit__ = lambda *a: None
        mock_ddgs.return_value.text.return_value = [
            {"title": "Coping tip", "href": "https://example.com/tip", "body": "You can do it."},
        ]
        emotions = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
        for emo in emotions:
            results = search_coping_strategies(emo, max_results=2)
            assert isinstance(results, list), f"No results for {emo}"
            assert len(results) > 0, f"Empty results for {emo}"


# ── LLM suggest_solutions integration ───────────────────────────


class TestSuggestSolutionsIntegration:
    """Test LLM suggest_solutions for post-session display."""

    def test_suggest_solutions_returns_string(self):
        llm = LLMService()
        result = llm.suggest_solutions("sad", "I feel stressed about work deadlines")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_suggest_solutions_different_emotions(self):
        llm = LLMService()
        for emo in ["sad", "angry", "fear", "happy"]:
            result = llm.suggest_solutions(emo, "General concern")
            assert isinstance(result, str)
            assert len(result) > 0


# ── Post-session display data ───────────────────────────────────


class TestPostSessionDisplay:
    """Test that ended session data is suitable for the UI display."""

    def test_diary_session_has_compassionate_response(self):
        session = DiarySession(
            session_id="test",
            start_time="2026-03-18 12:00:00",
            compassionate_response="I'm here for you."
        )
        assert session.compassionate_response == "I'm here for you."

    def test_diary_session_has_arxiv_results(self):
        session = DiarySession(
            session_id="test",
            start_time="2026-03-18 12:00:00",
            arxiv_results=[{"title": "Paper 1", "url": "https://arxiv.org/abs/1"}],
        )
        assert len(session.arxiv_results) == 1

    def test_diary_session_has_web_results(self):
        session = DiarySession(
            session_id="test",
            start_time="2026-03-18 12:00:00",
            web_results=[{"title": "Resource", "url": "https://example.com", "snippet": "Helpful"}],
        )
        assert len(session.web_results) == 1

    def test_dominant_emotion_with_entries(self):
        entries = [
            DiaryEntry("12:00", "t1", "sad", 0.8, {}, {}, "sad", 0.8),
            DiaryEntry("12:01", "t2", "sad", 0.7, {}, {}, "sad", 0.7),
            DiaryEntry("12:02", "t3", "happy", 0.9, {}, {}, "happy", 0.9),
        ]
        session = DiarySession(session_id="test", start_time="2026-03-18", entries=entries)
        assert session.dominant_emotion == "sad"

    def test_dominant_emotion_empty_session(self):
        session = DiarySession(session_id="test", start_time="2026-03-18")
        assert session.dominant_emotion == "neutral"


# ── Diary tab source code structure ─────────────────────────────


class TestDiaryTabStructure:
    """Verify the diary tab has required UI elements in its source."""

    @classmethod
    def _read_source(cls):
        with open(os.path.join(os.path.dirname(__file__), "..", "streamlit_app.py")) as f:
            return f.read()

    def test_post_session_display_block(self):
        src = self._read_source()
        assert "last_ended_session" in src
        assert "Session complete! Here are your insights:" in src
        assert "Start Fresh" in src

    def test_auto_processing_block(self):
        src = self._read_source()
        assert "last_diary_audio" in src
        assert "st.session_state.last_diary_audio" in src

    def test_video_processor_in_diary(self):
        """Diary tab should use DiaryVideoProcessor for video."""
        src = self._read_source()
        assert "Face Emotion" in src
        assert "DiaryVideoProcessor" in src

    def test_audio_input_in_diary(self):
        """Diary tab should use st.audio_input for recording."""
        src = self._read_source()
        assert "st.audio_input(" in src

    def test_diary_video_stream(self):
        """Diary tab should create webrtc_streamer with diary video key."""
        src = self._read_source()
        assert "diary-video-" in src

    def test_live_face_fragment(self):
        src = self._read_source()
        assert "diary_face_emotion_live" in src
        assert "diary_face_confidence_live" in src

    def test_session_history_has_coping_and_solutions(self):
        """Session history uses pre-computed solutions/coping from end_session."""
        src = self._read_source()
        history_idx = src.find("Session History")
        assert history_idx > 0
        after_history = src[history_idx:]
        # Solutions and coping are now pre-computed and stored on the session
        # object — the history section reads them from attributes, not re-calling
        assert "suggested_solutions" in after_history
        assert "coping_strategies" in after_history

    def test_save_entry_button_exists(self):
        src = self._read_source()
        assert "Save Entry" in src

    def test_fallback_manual_emotion_exists(self):
        """Fallback manual emotion override should be present."""
        src = self._read_source()
        assert "Manual Emotion Override" in src


# ── Full diary flow end-to-end ──────────────────────────────────


class TestFullDiaryFlow:
    """Integration test: complete diary flow with all auto features."""

    @patch("diary_session.search_general", return_value=[])
    @patch("diary_session.search_emotion_articles", return_value=[])
    @patch("diary_session.DiarySessionManager._search_arxiv")
    @patch("diary_session.LLMService")
    @patch("diary_session.MoodFusion")
    @patch("diary_session.SpeechService")
    def test_full_flow_start_record_end(
        self, mock_speech_cls, mock_fusion_cls, mock_llm_cls, mock_arxiv, mock_web_articles, mock_web_general
    ):
        mock_speech = mock_speech_cls.return_value
        mock_speech.transcribe_audio.return_value = {
            "text": "I'm feeling anxious about my presentation",
            "language": "en",
            "duration": 4.0,
        }
        mock_speech.analyze_sentiment.return_value = {
            "polarity": -0.3, "subjectivity": 0.6, "emotion": "fear", "confidence": 0.7,
        }
        mock_speech.analyze_audio_emotion.return_value = {
            "energy": 0.5, "pitch_var": 0.4, "estimated_emotion": "fear", "confidence": 0.6,
        }
        mock_fusion = mock_fusion_cls.return_value
        mock_fusion.fuse.side_effect = [
            {"emotion": "fear", "confidence": 0.75},
            {"emotion": "fear", "confidence": 0.65},
        ]

        mock_llm = mock_llm_cls.return_value
        mock_llm.summarize.return_value = "User expressed anxiety about a presentation."
        mock_llm.compassionate_response.return_value = (
            "It's completely normal to feel anxious before a presentation. "
            "Your feelings are valid."
        )
        mock_llm.suggest_solutions.return_value = (
            "1. Practice deep breathing\n2. Prepare thoroughly\n3. Visualize success"
        )

        mock_arxiv.return_value = [
            {"title": "Anxiety and Performance", "url": "https://arxiv.org/abs/5678", "abstract": "Study..."},
        ]

        mgr = DiarySessionManager()

        # 1. Start session
        session = mgr.start_session()
        assert session.session_id is not None
        assert len(session.entries) == 0

        # 2. Simulate audio_input: two recordings
        last_diary_audio = None

        audio1 = b"\xaa" * 1000
        if audio1 != last_diary_audio:
            last_diary_audio = audio1
            entry1 = mgr.add_entry(session, audio1, face_emotion="fear", face_confidence=0.8)
            assert entry1.fused_emotion == "fear"

        audio2 = b"\xbb" * 1000
        if audio2 != last_diary_audio:
            last_diary_audio = audio2
            entry2 = mgr.add_entry(session, audio2, face_emotion="fear", face_confidence=0.7)
            assert entry2.fused_emotion == "fear"

        # Same audio again — should NOT process
        if audio2 != last_diary_audio:
            pytest.fail("Same audio should not trigger reprocessing")

        assert len(session.entries) == 2

        # 3. End session
        ended = mgr.end_session(session)

        # 4. Verify all post-session display data
        assert ended.summary != ""
        assert ended.compassionate_response != ""
        assert ended.dominant_emotion == "fear"
        assert len(ended.arxiv_results) >= 1
        assert isinstance(ended.web_results, list)

        # 5. Simulate storing for UI display
        diary_history = [ended]
        last_ended_session = ended

        coping = search_coping_strategies(ended.dominant_emotion, max_results=3)
        assert isinstance(coping, list)
        assert len(coping) > 0

        assert hasattr(ended, "compassionate_response")
        assert hasattr(ended, "summary")
        assert hasattr(ended, "arxiv_results")
        assert hasattr(ended, "web_results")
        assert hasattr(ended, "dominant_emotion")

    @patch("diary_session.LLMService")
    @patch("diary_session.MoodFusion")
    @patch("diary_session.SpeechService")
    def test_multiple_sessions_history(self, mock_speech_cls, mock_fusion_cls, mock_llm_cls):
        mock_speech = mock_speech_cls.return_value
        mock_speech.transcribe_audio.return_value = {
            "text": "test", "language": "en", "duration": 1.0,
        }
        mock_speech.analyze_sentiment.return_value = {
            "polarity": 0.0, "subjectivity": 0.0, "emotion": "neutral", "confidence": 0.5,
        }
        mock_speech.analyze_audio_emotion.return_value = {
            "energy": 0.3, "pitch_var": 0.2, "estimated_emotion": "neutral", "confidence": 0.4,
        }
        mock_fusion = mock_fusion_cls.return_value
        mock_fusion.fuse.return_value = {"emotion": "neutral", "confidence": 0.5}

        mock_llm = mock_llm_cls.return_value
        mock_llm.summarize.return_value = "Session summary."
        mock_llm.compassionate_response.return_value = "Thanks for sharing."

        mgr = DiarySessionManager()
        diary_history = []

        for _ in range(3):
            session = mgr.start_session()
            mgr.add_entry(session, b"\x00" * 500, "neutral", 0.5)
            with patch.object(DiarySessionManager, "_search_arxiv", return_value=[]):
                with patch("diary_session.search_emotion_articles", return_value=[]):
                    with patch("diary_session.search_general", return_value=[]):
                        ended = mgr.end_session(session)
            diary_history.append(ended)

        assert len(diary_history) == 3
        for sess in diary_history:
            assert isinstance(sess, DiarySession)
            assert sess.summary != ""
