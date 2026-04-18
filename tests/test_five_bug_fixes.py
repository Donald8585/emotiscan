"""
Tests for the five bug fixes:
1. No duplicate research/search after end_session
2. No equalizeHist in face preprocessing (was causing 14% equal scores)
3. Pre-computed coping strategies & solutions
4. Diary fragment uses 2s interval (not 1s)
5. Emotion detection tab has real-time chart
"""
import inspect
import unittest
import numpy as np


class TestNoDuplicateResearch(unittest.TestCase):
    """Fix #1: Session history + post-session view use pre-computed data."""

    def setUp(self):
        with open("streamlit_app.py") as f:
            self.src = f.read()

    def test_post_session_does_not_call_suggest_solutions(self):
        """Post-session view should NOT call llm.suggest_solutions (pre-computed)."""
        # Find the post-session block (between last_ended_session and Session controls)
        idx_start = self.src.find("last_ended_session is not None and st.session_state.diary_session is None")
        idx_end = self.src.find("# ?€?€ Session controls ?€?€")
        post_session_block = self.src[idx_start:idx_end]
        self.assertNotIn("llm.suggest_solutions", post_session_block,
                         "Post-session should use pre-computed solutions, not re-call LLM")

    def test_post_session_does_not_call_search_coping(self):
        """Post-session view should NOT call search_coping_strategies (pre-computed)."""
        idx_start = self.src.find("last_ended_session is not None and st.session_state.diary_session is None")
        idx_end = self.src.find("# ?€?€ Session controls ?€?€")
        post_session_block = self.src[idx_start:idx_end]
        self.assertNotIn("search_coping_strategies", post_session_block,
                         "Post-session should use pre-computed coping, not re-search")

    def test_session_history_does_not_call_suggest_solutions(self):
        """Session History section should NOT call llm.suggest_solutions."""
        idx = self.src.find("Session History")
        after = self.src[idx:]
        self.assertNotIn("llm.suggest_solutions", after,
                         "Session History should use pre-computed solutions")

    def test_session_history_does_not_call_search_coping(self):
        """Session History section should NOT call search_coping_strategies."""
        idx = self.src.find("Session History")
        after = self.src[idx:]
        self.assertNotIn("search_coping_strategies", after,
                         "Session History should use pre-computed coping")

    def test_end_session_precomputes_solutions(self):
        """End session flow should compute suggested_solutions."""
        # Find the end session block
        idx_start = self.src.find("End Session")
        idx_end = self.src.find("st.rerun()", idx_start)
        end_block = self.src[idx_start:idx_end]
        self.assertIn("suggested_solutions", end_block,
                      "End session must pre-compute suggested_solutions")

    def test_end_session_precomputes_coping(self):
        """End session flow should compute coping_strategies."""
        idx_start = self.src.find("End Session")
        idx_end = self.src.find("st.rerun()", idx_start)
        end_block = self.src[idx_start:idx_end]
        self.assertIn("coping_strategies", end_block,
                      "End session must pre-compute coping_strategies")


class TestDeepFaceGetsValidInput(unittest.TestCase):
    """Fix #2: DeepFace must receive 3-channel BGR images.
    
    Root cause: passing grayscale 2D arrays to DeepFace.analyze() causes
    'not enough values to unpack (expected 3, got 2)' which the except
    block silently catches ??uniform 14.3% scores for all emotions.
    """

    def test_preprocess_no_equalize_hist(self):
        """_preprocess_face_for_emotion must NOT use equalizeHist."""
        from services.emotion_detector import EmotionDetector
        src = inspect.getsource(EmotionDetector._preprocess_face_for_emotion)
        self.assertNotIn("equalizeHist", src)

    def test_preprocess_no_clahe(self):
        """_preprocess_face_for_emotion must NOT use CLAHE."""
        from services.emotion_detector import EmotionDetector
        src = inspect.getsource(EmotionDetector._preprocess_face_for_emotion)
        self.assertNotIn("createCLAHE", src)

    def test_bgr_input_returns_3channel(self):
        """BGR input should pass through as-is (3 channels)."""
        from services.emotion_detector import EmotionDetector
        face = np.random.randint(0, 255, (100, 80, 3), dtype=np.uint8)
        result = EmotionDetector._preprocess_face_for_emotion(face)
        self.assertEqual(result.ndim, 3, "BGR input must stay 3-channel")
        self.assertEqual(result.shape, (100, 80, 3))

    def test_grayscale_input_converted_to_3channel(self):
        """Grayscale 2D input must be converted to 3-channel for DeepFace."""
        from services.emotion_detector import EmotionDetector
        face = np.random.randint(0, 255, (60, 60), dtype=np.uint8)
        result = EmotionDetector._preprocess_face_for_emotion(face)
        self.assertEqual(result.ndim, 3, "Grayscale must be converted to 3-channel")
        self.assertEqual(result.shape[2], 3)

    def test_no_manual_resize_to_48x48(self):
        """Must NOT resize manually ??DeepFace handles its own resizing."""
        from services.emotion_detector import EmotionDetector
        src = inspect.getsource(EmotionDetector._preprocess_face_for_emotion)
        self.assertNotIn("(48, 48)", src,
                         "Do not manually resize ??DeepFace resizes internally")

    def test_output_not_resized(self):
        """Output should keep original dimensions (not forced to 48x48)."""
        from services.emotion_detector import EmotionDetector
        face = np.random.randint(0, 255, (100, 80, 3), dtype=np.uint8)
        result = EmotionDetector._preprocess_face_for_emotion(face)
        self.assertEqual(result.shape[:2], (100, 80),
                         "Should preserve original face crop size")

    def test_preserves_pixel_values(self):
        """Output should NOT modify pixel values (no equalization)."""
        from services.emotion_detector import EmotionDetector
        face = np.zeros((50, 50, 3), dtype=np.uint8)
        face[:, :25, :] = 30
        face[:, 25:, :] = 220
        result = EmotionDetector._preprocess_face_for_emotion(face)
        # Pixel values should be unchanged
        np.testing.assert_array_equal(result, face)


class TestDiarySessionHasPrecomputedFields(unittest.TestCase):
    """Fix #3: DiarySession stores suggested_solutions and coping_strategies."""

    def test_diary_session_has_suggested_solutions(self):
        from diary.diary_session import DiarySession
        session = DiarySession(session_id="test", start_time="now")
        self.assertTrue(hasattr(session, "suggested_solutions"))
        self.assertEqual(session.suggested_solutions, "")

    def test_diary_session_has_coping_strategies(self):
        from diary.diary_session import DiarySession
        session = DiarySession(session_id="test", start_time="now")
        self.assertTrue(hasattr(session, "coping_strategies"))
        self.assertEqual(session.coping_strategies, [])

    def test_to_dict_includes_precomputed_fields(self):
        from diary.diary_session import DiarySession
        session = DiarySession(
            session_id="test",
            start_time="now",
            suggested_solutions="Try breathing exercises",
            coping_strategies=[{"title": "Coping", "url": "https://example.com", "snippet": "test"}],
        )
        d = session.to_dict()
        self.assertIn("suggested_solutions", d)
        self.assertEqual(d["suggested_solutions"], "Try breathing exercises")
        self.assertIn("coping_strategies", d)
        self.assertEqual(len(d["coping_strategies"]), 1)


class TestDiaryFragmentInterval(unittest.TestCase):
    """Fix #4: Diary tab fragment uses 2s interval to reduce lag."""

    def test_diary_fragment_uses_2s_interval(self):
        with open("streamlit_app.py") as f:
            src = f.read()
        # Find the diary live panel fragment
        idx = src.find("def _diary_av_live_panel")
        # Look backwards for the decorator
        before = src[:idx]
        decorator_line = before[before.rfind("@st.fragment"):]
        self.assertIn("run_every=2.0", decorator_line,
                      "Diary fragment should use 2.0s interval to reduce lag")

    def test_emotion_tab_fragment_uses_1s_interval(self):
        """Emotion detection tab should still use 1s (it has no audio overhead)."""
        with open("streamlit_app.py") as f:
            src = f.read()
        # Find the emotion tab live panel
        idx = src.find("def _live_emotion_panel")
        before = src[:idx]
        decorator_line = before[before.rfind("@st.fragment"):]
        self.assertIn("run_every=1.0", decorator_line,
                      "Emotion tab fragment should use 1.0s interval")


class TestEmotionDetectionTabHasChart(unittest.TestCase):
    """Fix #5: Emotion detection tab has a real-time chart."""

    def test_emotion_tab_has_plotly_chart(self):
        with open("streamlit_app.py") as f:
            src = f.read()
        # Find the emotion detection tab section
        tab_start = src.find("# TAB 1: Emotion Detection")
        tab_end = src.find("# TAB 2:")
        emotion_tab = src[tab_start:tab_end]
        self.assertIn("plotly_chart", emotion_tab,
                      "Emotion detection tab should have a plotly chart")

    def test_emotion_tab_has_emotion_history(self):
        with open("streamlit_app.py") as f:
            src = f.read()
        tab_start = src.find("# TAB 1: Emotion Detection")
        tab_end = src.find("# TAB 2:")
        emotion_tab = src[tab_start:tab_end]
        self.assertIn("emotion_history", emotion_tab,
                      "Emotion detection tab should track emotion_history")
        self.assertIn("Emotion History", emotion_tab,
                      "Should have Emotion History chart title")


if __name__ == "__main__":
    unittest.main()
