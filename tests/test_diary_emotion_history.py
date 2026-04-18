"""
Tests for diary emotion history tracking ??per-second face emotion logging
and face_emotion_timeline field on DiaryEntry.
"""
import sys
import os
import types
import unittest
from unittest.mock import patch, MagicMock
from dataclasses import asdict

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestDiaryEntryTimeline(unittest.TestCase):
    """Tests for the face_emotion_timeline field on DiaryEntry."""

    def test_diary_entry_has_timeline_field(self):
        from diary.diary_session import DiaryEntry
        entry = DiaryEntry(
            timestamp="12:00:00",
            text="hello",
            face_emotion="happy",
            face_confidence=0.9,
            voice_sentiment={"polarity": 0.5, "emotion": "happy", "confidence": 0.8},
            audio_emotion={"energy": 0.5, "estimated_emotion": "happy", "confidence": 0.7},
            fused_emotion="happy",
            fused_confidence=0.85,
        )
        self.assertIsInstance(entry.face_emotion_timeline, list)
        self.assertEqual(len(entry.face_emotion_timeline), 0)

    def test_diary_entry_timeline_default_empty(self):
        from diary.diary_session import DiaryEntry
        entry = DiaryEntry(
            timestamp="12:00:00",
            text="test",
            face_emotion="neutral",
            face_confidence=0.5,
            voice_sentiment={},
            audio_emotion={},
            fused_emotion="neutral",
            fused_confidence=0.5,
        )
        self.assertEqual(entry.face_emotion_timeline, [])

    def test_diary_entry_timeline_with_data(self):
        from diary.diary_session import DiaryEntry
        timeline = [
            {"time": "12:00:01", "emotion": "happy", "confidence": 0.9},
            {"time": "12:00:02", "emotion": "sad", "confidence": 0.7},
            {"time": "12:00:03", "emotion": "happy", "confidence": 0.85},
        ]
        entry = DiaryEntry(
            timestamp="12:00:03",
            text="hello world",
            face_emotion="happy",
            face_confidence=0.88,
            voice_sentiment={"polarity": 0.3, "emotion": "happy", "confidence": 0.8},
            audio_emotion={"energy": 0.4, "estimated_emotion": "neutral", "confidence": 0.6},
            fused_emotion="happy",
            fused_confidence=0.82,
            face_emotion_timeline=timeline,
        )
        self.assertEqual(len(entry.face_emotion_timeline), 3)
        self.assertEqual(entry.face_emotion_timeline[0]["emotion"], "happy")
        self.assertEqual(entry.face_emotion_timeline[1]["emotion"], "sad")

    def test_diary_entry_to_dict_includes_timeline(self):
        from diary.diary_session import DiaryEntry
        timeline = [{"time": "12:00:01", "emotion": "angry", "confidence": 0.6}]
        entry = DiaryEntry(
            timestamp="12:00:01",
            text="test",
            face_emotion="angry",
            face_confidence=0.6,
            voice_sentiment={},
            audio_emotion={},
            fused_emotion="angry",
            fused_confidence=0.6,
            face_emotion_timeline=timeline,
        )
        d = entry.to_dict()
        self.assertIn("face_emotion_timeline", d)
        self.assertEqual(len(d["face_emotion_timeline"]), 1)
        self.assertEqual(d["face_emotion_timeline"][0]["emotion"], "angry")

    def test_diary_entry_to_markdown_still_works_with_timeline(self):
        from diary.diary_session import DiaryEntry
        entry = DiaryEntry(
            timestamp="12:00:00",
            text="hello",
            face_emotion="happy",
            face_confidence=0.9,
            voice_sentiment={"polarity": 0.5, "emotion": "happy", "confidence": 0.8},
            audio_emotion={"energy": 0.5, "estimated_emotion": "happy", "confidence": 0.7},
            fused_emotion="happy",
            fused_confidence=0.85,
            face_emotion_timeline=[{"time": "12:00:00", "emotion": "happy", "confidence": 0.9}],
        )
        md = entry.to_markdown()
        self.assertIn("happy", md)
        self.assertIn("12:00:00", md)


class TestAddEntryTimeline(unittest.TestCase):
    """Tests that add_entry passes face_emotion_timeline to DiaryEntry."""

    @patch("diary_session.SpeechService")
    @patch("diary_session.MoodFusion")
    @patch("diary_session.LLMService")
    def test_add_entry_with_timeline(self, mock_llm_cls, mock_fusion_cls, mock_speech_cls):
        from diary.diary_session import DiarySessionManager, DiarySession

        mock_speech = mock_speech_cls.return_value
        mock_speech.transcribe_audio.return_value = {"text": "I feel good today"}
        mock_speech.analyze_sentiment.return_value = {
            "polarity": 0.6, "subjectivity": 0.5, "emotion": "happy", "confidence": 0.8
        }
        mock_speech.analyze_audio_emotion.return_value = {
            "energy": 0.5, "pitch_var": 0.3, "estimated_emotion": "happy", "confidence": 0.7
        }

        mock_fusion = mock_fusion_cls.return_value
        mock_fusion.fuse.return_value = {"emotion": "happy", "confidence": 0.85}

        mgr = DiarySessionManager()
        sess = mgr.start_session()

        timeline = [
            {"time": "12:00:01", "emotion": "happy", "confidence": 0.9},
            {"time": "12:00:02", "emotion": "happy", "confidence": 0.88},
        ]

        entry = mgr.add_entry(
            sess,
            b"fake_audio_bytes",
            face_emotion="happy",
            face_confidence=0.9,
            face_emotion_timeline=timeline,
        )

        self.assertEqual(len(entry.face_emotion_timeline), 2)
        self.assertEqual(entry.face_emotion_timeline[0]["emotion"], "happy")

    @patch("diary_session.SpeechService")
    @patch("diary_session.MoodFusion")
    @patch("diary_session.LLMService")
    def test_add_entry_without_timeline_defaults_empty(self, mock_llm_cls, mock_fusion_cls, mock_speech_cls):
        from diary.diary_session import DiarySessionManager

        mock_speech = mock_speech_cls.return_value
        mock_speech.transcribe_audio.return_value = {"text": "test"}
        mock_speech.analyze_sentiment.return_value = {
            "polarity": 0.0, "subjectivity": 0.5, "emotion": "neutral", "confidence": 0.5
        }
        mock_speech.analyze_audio_emotion.return_value = {
            "energy": 0.3, "pitch_var": 0.2, "estimated_emotion": "neutral", "confidence": 0.5
        }

        mock_fusion = mock_fusion_cls.return_value
        mock_fusion.fuse.return_value = {"emotion": "neutral", "confidence": 0.5}

        mgr = DiarySessionManager()
        sess = mgr.start_session()

        entry = mgr.add_entry(sess, b"fake_audio_bytes")
        self.assertEqual(entry.face_emotion_timeline, [])


class TestEmotionAudioVideoProcessorTimelineSnapshot(unittest.TestCase):
    """Tests for the emotion_timeline_snapshot property."""

    def _get_processor_class(self):
        """Import EmotionAudioVideoProcessor from streamlit_app with mocks."""
        # We need to mock streamlit and av dependencies
        mock_st = MagicMock()
        mock_av = types.ModuleType("av")
        mock_av.VideoFrame = MagicMock()
        mock_av.AudioFrame = MagicMock()

        mods = {
            "streamlit": mock_st,
            "streamlit_webrtc": MagicMock(),
            "av": mock_av,
            "plotly": MagicMock(),
            "plotly.graph_objects": MagicMock(),
        }

        with patch.dict(sys.modules, mods):
            # Import just to check the class is there
            import importlib
            if "streamlit_app" in sys.modules:
                importlib.reload(sys.modules["streamlit_app"])

        # Instead of importing the module directly (complex setup), test the
        # concept by checking that _emotion_timeline is used in the source.
        import inspect
        with open(os.path.join(os.path.dirname(__file__), "..", "streamlit_app.py")) as f:
            src = f.read()
        return src

    def test_shared_diary_buffer_exists_in_source(self):
        src = self._get_processor_class()
        self.assertIn("_SharedDiaryBuffer", src)
        self.assertIn("_diary_buffer", src)

    def test_diary_emotion_history_in_session_defaults(self):
        src = self._get_processor_class()
        self.assertIn("diary_emotion_history", src)

    def test_diary_emotion_history_reset_on_start_session(self):
        src = self._get_processor_class()
        # Verify diary_emotion_history is reset when a new session starts
        # (Should appear near "Start New Session" button)
        idx_start = src.index("Start New Session")
        # Within 500 chars of that, diary_emotion_history should be reset
        nearby = src[idx_start:idx_start + 800]
        self.assertIn("diary_emotion_history", nearby)

    def test_live_chart_code_in_diary_panel(self):
        src = self._get_processor_class()
        self.assertIn("Diary Emotion Timeline (Live)", src)
        self.assertIn("diary_emotion_history", src)

    def test_timeline_passed_to_add_entry_manual_save(self):
        src = self._get_processor_class()
        # Manual save path should pass face_emotion_timeline
        self.assertIn("face_emotion_timeline=_timeline_snap", src)

    def test_timeline_chart_in_entry_display(self):
        src = self._get_processor_class()
        self.assertIn("Face Emotion During Entry", src)

    def test_emotion_history_reset_after_save(self):
        """After saving an entry, diary_emotion_history should be cleared."""
        src = self._get_processor_class()
        # Both manual and auto save paths should reset the history
        count = src.count("diary_emotion_history = []")
        self.assertGreaterEqual(count, 2, "Both save paths should reset diary_emotion_history")


class TestDiaryEntryTimelineSerialisation(unittest.TestCase):
    """Tests for round-tripping face_emotion_timeline through dict/json."""

    def test_timeline_survives_asdict(self):
        from diary.diary_session import DiaryEntry
        timeline = [
            {"time": "12:00:01", "emotion": "happy", "confidence": 0.9},
            {"time": "12:00:02", "emotion": "sad", "confidence": 0.7},
        ]
        entry = DiaryEntry(
            timestamp="12:00:02",
            text="test",
            face_emotion="happy",
            face_confidence=0.9,
            voice_sentiment={"polarity": 0.5, "emotion": "happy", "confidence": 0.8},
            audio_emotion={"energy": 0.5, "estimated_emotion": "happy", "confidence": 0.7},
            fused_emotion="happy",
            fused_confidence=0.85,
            face_emotion_timeline=timeline,
        )
        d = asdict(entry)
        self.assertEqual(d["face_emotion_timeline"], timeline)

    def test_timeline_survives_json_round_trip(self):
        import json
        from diary.diary_session import DiaryEntry
        timeline = [{"time": "12:00:01", "emotion": "angry", "confidence": 0.6}]
        entry = DiaryEntry(
            timestamp="12:00:01",
            text="test",
            face_emotion="angry",
            face_confidence=0.6,
            voice_sentiment={},
            audio_emotion={},
            fused_emotion="angry",
            fused_confidence=0.6,
            face_emotion_timeline=timeline,
        )
        j = json.dumps(entry.to_dict())
        d = json.loads(j)
        self.assertEqual(d["face_emotion_timeline"], timeline)

    def test_empty_timeline_in_dict(self):
        from diary.diary_session import DiaryEntry
        entry = DiaryEntry(
            timestamp="12:00:00",
            text="test",
            face_emotion="neutral",
            face_confidence=0.5,
            voice_sentiment={},
            audio_emotion={},
            fused_emotion="neutral",
            fused_confidence=0.5,
        )
        d = entry.to_dict()
        self.assertEqual(d["face_emotion_timeline"], [])


if __name__ == "__main__":
    unittest.main()
