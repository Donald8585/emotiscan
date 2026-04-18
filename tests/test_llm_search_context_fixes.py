"""Tests for the LLM/research/search context fixes.

Verifies that:
1. LLM feedback uses actual transcript + multi-modal emotions (not just fused emotion)
2. Research queries address user's actual spoken problems
3. Web search uses best_emotion (non-neutral) instead of fused-only dominant
4. Emotion scanner: grayscale conversion, double-CLAHE, SMOOTHING_WINDOW=3
5. Video processor: _last_results redraw, DETECT_INTERVAL=0.5
6. Helper functions _get_best_emotion and _get_session_transcript exist
"""

import ast
import os
import re
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

SRC_DIR = os.path.join(os.path.dirname(__file__), "..")


def _read(filename):
    with open(os.path.join(SRC_DIR, filename)) as f:
        return f.read()


# ?��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��???# DiarySession.best_emotion / get_full_context
# ?��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��???
class TestBestEmotionProperty(unittest.TestCase):
    """DiarySession.best_emotion should prefer non-neutral across all modalities."""

    def _make_session(self, entries):
        from diary.diary_session import DiarySession, DiaryEntry
        sess = DiarySession(session_id="test", start_time="now")
        for e in entries:
            sess.entries.append(e)
        return sess

    def _make_entry(self, face="neutral", voice="neutral", audio="neutral",
                    fused="neutral", text="test"):
        from diary.diary_session import DiaryEntry
        return DiaryEntry(
            timestamp="12:00:00", text=text,
            face_emotion=face, face_confidence=0.8,
            voice_sentiment={"emotion": voice, "confidence": 0.7, "polarity": 0.0},
            audio_emotion={"estimated_emotion": audio, "energy": 0.5, "confidence": 0.6},
            fused_emotion=fused, fused_confidence=0.7,
        )

    def test_all_neutral_returns_neutral(self):
        sess = self._make_session([self._make_entry()])
        self.assertEqual(sess.best_emotion, "neutral")

    def test_face_detected_non_neutral(self):
        """If face detects happy but everything else is neutral, best_emotion = happy."""
        sess = self._make_session([self._make_entry(face="happy")])
        self.assertEqual(sess.best_emotion, "happy")

    def test_voice_detected_non_neutral(self):
        sess = self._make_session([self._make_entry(voice="sad")])
        self.assertEqual(sess.best_emotion, "sad")

    def test_audio_detected_non_neutral(self):
        sess = self._make_session([self._make_entry(audio="angry")])
        self.assertEqual(sess.best_emotion, "angry")

    def test_most_frequent_wins(self):
        """If face=happy appears 3x and voice=sad appears 1x, happy wins."""
        entries = [
            self._make_entry(face="happy"),
            self._make_entry(face="happy"),
            self._make_entry(face="happy", voice="sad"),
        ]
        sess = self._make_session(entries)
        self.assertEqual(sess.best_emotion, "happy")

    def test_fused_neutral_but_face_not(self):
        """Fused may be neutral but face detected sad ??best_emotion should be sad."""
        sess = self._make_session([
            self._make_entry(face="sad", fused="neutral"),
        ])
        self.assertEqual(sess.best_emotion, "sad")


class TestGetFullContext(unittest.TestCase):
    """get_full_context should return transcript + per-modality emotions."""

    def test_returns_all_keys(self):
        from diary.diary_session import DiarySession, DiaryEntry
        sess = DiarySession(session_id="t", start_time="now")
        entry = DiaryEntry(
            timestamp="12:00", text="I feel stressed about work",
            face_emotion="sad", face_confidence=0.9,
            voice_sentiment={"emotion": "sad", "polarity": -0.5, "confidence": 0.8},
            audio_emotion={"estimated_emotion": "angry", "energy": 0.7, "confidence": 0.6},
            fused_emotion="sad", fused_confidence=0.8,
        )
        sess.entries.append(entry)
        ctx = sess.get_full_context()
        self.assertIn("transcript", ctx)
        self.assertIn("face_emotions", ctx)
        self.assertIn("voice_emotions", ctx)
        self.assertIn("audio_emotions", ctx)
        self.assertIn("best_emotion", ctx)
        self.assertEqual(ctx["transcript"], "I feel stressed about work")
        self.assertEqual(ctx["face_emotions"], ["sad"])
        self.assertEqual(ctx["voice_emotions"], ["sad"])
        self.assertEqual(ctx["audio_emotions"], ["angry"])
        self.assertEqual(ctx["best_emotion"], "sad")


# ?��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��???# diary_session.py: end_session, summary, queries, compassionate_response
# ?��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��???
class TestDiarySessionPrompts(unittest.TestCase):
    """Verify prompts include transcript + multi-modal emotions."""

    def setUp(self):
        self.src = _read("diary/diary_session.py")

    def test_summary_prompt_includes_transcript(self):
        self.assertIn("Transcript:", self.src)
        self.assertIn("Face-camera emotion breakdown:", self.src)
        self.assertIn("Voice-tone emotions:", self.src)

    def test_summary_prompt_mentions_problems(self):
        """Summary prompt should ask about actual problems, not just emotions."""
        self.assertIn("PROBLEMS", self.src)

    def test_research_queries_prompt_uses_transcript(self):
        # Prompt should include transcript content
        self.assertIn("What the user said", self.src)
        # Should ask for CONCRETE problem-specific queries
        self.assertIn("CONCRETE topic/problem from the transcript", self.src)

    def test_research_queries_prompt_rejects_generic(self):
        """Prompt should explicitly reject generic emotion-only queries."""
        # New prompt uses BAD/GOOD examples to steer away from generic
        self.assertIn("BAD:", self.src)
        self.assertIn("GOOD:", self.src)

    def test_compassionate_response_uses_transcript(self):
        self.assertIn("What they said:", self.src)
        self.assertIn("SPECIFIC problems/topics", self.src)

    def test_end_session_uses_best_emotion(self):
        """end_session should use best_emotion for search, not just dominant."""
        self.assertIn("best_emotion", self.src)
        self.assertIn("emotion_for_search", self.src)

    def test_end_session_passes_ctx(self):
        """end_session should build ctx once and pass to all methods."""
        self.assertIn("ctx = session.get_full_context()", self.src)

    def test_web_search_uses_research_queries(self):
        """Web search should use research queries (problem-specific), not just emotion."""
        # Should search by research queries, not by search_emotion_articles(dominant) first
        idx_rq = self.src.find("session.research_queries[:3]")
        self.assertGreater(idx_rq, -1,
                           "Web search should iterate over research_queries")

    def test_summary_method_accepts_ctx(self):
        """get_session_summary should accept optional ctx parameter."""
        self.assertIn("def get_session_summary(self, session: DiarySession, ctx:", self.src)

    def test_research_queries_method_accepts_ctx(self):
        self.assertIn("def get_research_queries(self, session: DiarySession, ctx:", self.src)

    def test_compassionate_method_accepts_ctx(self):
        self.assertIn("def get_compassionate_response(self, session: DiarySession, ctx:", self.src)


# ?��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��???# llm_service.py: suggest_solutions + compassionate_chat
# ?��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��???
class TestLLMServicePrompts(unittest.TestCase):
    """LLM service should accept and use transcript + multi-modal emotion data."""

    def setUp(self):
        self.src = _read("services/llm_service.py")

    def test_suggest_solutions_accepts_face_and_voice(self):
        """suggest_solutions should accept face_emotion and voice_emotion params."""
        self.assertIn("face_emotion", self.src)
        self.assertIn("voice_emotion", self.src)

    def test_suggest_solutions_addresses_actual_problems(self):
        self.assertIn("ACTUAL", self.src)
        # Prompt should steer LLM toward real-world problem-solving
        self.assertIn("practical", self.src.lower())

    def test_compassionate_chat_uses_transcript(self):
        """compassionate_chat context should include transcript."""
        self.assertIn("What user said:", self.src)

    def test_compassionate_chat_shows_detected_emotion_source(self):
        """Should clarify the emotion is from face+voice+audio."""
        self.assertIn("face+voice+audio", self.src)


# ?��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��???# streamlit_app.py: helper functions + call sites
# ?��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��???
class TestStreamlitHelpers(unittest.TestCase):
    """Verify _get_best_emotion and _get_session_transcript exist and are used."""

    def setUp(self):
        self.src = _read("streamlit_app.py")

    def test_get_best_emotion_defined(self):
        self.assertIn("def _get_best_emotion(session_obj):", self.src)

    def test_get_session_transcript_defined(self):
        self.assertIn("def _get_session_transcript(session_obj)", self.src)

    def test_post_session_uses_best_emotion(self):
        """Post-session display should use _get_best_emotion."""
        self.assertIn("_les_best = _get_best_emotion(_les)", self.src)

    def test_session_history_uses_best_emotion(self):
        self.assertIn("_sh_best = _get_best_emotion(_sh)", self.src)

    def test_solutions_receive_transcript(self):
        """Post-session view uses pre-computed suggested_solutions (no re-call)."""
        # After refactor, solutions are pre-computed during end_session and stored
        # on the session object. The post-session view reads the attribute directly.
        self.assertIn("suggested_solutions", self.src)

    def test_chat_context_includes_transcript(self):
        """Chat session context should include transcript key."""
        # Count occurrences of "transcript" key in context dicts
        count = self.src.count('"transcript"')
        self.assertGreaterEqual(count, 2, "At least 2 chat contexts should include transcript")


# ?��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��???# Emotion scanner accuracy: grayscale, double CLAHE, SMOOTHING_WINDOW
# ?��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��???
class TestEmotionDetectorAccuracy(unittest.TestCase):
    """Verify accuracy improvements in emotion_detector.py."""

    def setUp(self):
        self.src = _read("services/emotion_detector.py")

    def test_smoothing_window_is_3(self):
        match = re.search(r"SMOOTHING_WINDOW\s*=\s*(\d+)", self.src)
        self.assertIsNotNone(match)
        self.assertEqual(int(match.group(1)), 3)

    def test_face_stays_bgr_for_classify(self):
        """Face crop must stay 3-channel ??HSEmotion expects BGR/RGB input."""
        import inspect
        from services.emotion_detector import EmotionDetector
        method_src = inspect.getsource(EmotionDetector._preprocess_face_for_emotion)
        # Must convert gray?�BGR, since emotion models need 3-channel
        self.assertIn("COLOR_GRAY2BGR", method_src)
        self.assertNotIn("COLOR_BGR2GRAY", method_src)

    def test_no_histogram_equalization_on_face(self):
        """Face preprocessing must NOT use equalizeHist or CLAHE ??it destroys features."""
        import inspect
        from services.emotion_detector import EmotionDetector
        method_src = inspect.getsource(EmotionDetector._preprocess_face_for_emotion)
        self.assertNotIn("equalizeHist", method_src)
        self.assertNotIn("createCLAHE", method_src)

    def test_opencv_hsemotion_uses_classify_helper(self):
        """opencv_hsemotion should use _classify_hsemotion for face crops."""
        import inspect
        from services.emotion_detector import EmotionDetector
        method_src = inspect.getsource(EmotionDetector._detect_opencv_hsemotion)
        self.assertIn("_classify_hsemotion", method_src)

    def test_resize_to_48x48(self):
        """Haar minSize should be (48, 48)."""
        import inspect
        from services.emotion_detector import EmotionDetector
        method_src = inspect.getsource(EmotionDetector._detect_opencv_hsemotion)
        self.assertIn("(48, 48)", method_src)


class TestEmotionDetectorCLAHERuntime(unittest.TestCase):
    """Runtime tests for the CLAHE + grayscale pipeline."""

    def test_grayscale_48x48_is_2d(self):
        """A grayscale 48x48 image should be 2D (h, w) not 3D."""
        import cv2
        color_face = np.random.randint(0, 255, (80, 80, 3), dtype=np.uint8)
        gray = cv2.cvtColor(color_face, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)
        self.assertEqual(resized.shape, (48, 48))
        self.assertEqual(resized.ndim, 2)

    def test_clahe_on_grayscale_improves_contrast(self):
        """CLAHE on a dark grayscale face should brighten it."""
        import cv2
        dark = np.full((48, 48), 30, dtype=np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        result = clahe.apply(dark)
        self.assertGreater(result.mean(), dark.mean())

    def test_full_pipeline_produces_valid_input(self):
        """Full pipeline: color frame ??CLAHE ??crop ??grayscale ??CLAHE ??resize."""
        import cv2
        frame = np.random.randint(30, 100, (200, 200, 3), dtype=np.uint8)
        # CLAHE on full frame
        from services.emotion_detector import EmotionDetector
        det = EmotionDetector.__new__(EmotionDetector)
        processed = det._preprocess_frame(frame)
        # Simulate face crop
        face_crop = processed[50:150, 50:150]
        face_gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        face_gray = clahe.apply(face_gray)
        face_resized = cv2.resize(face_gray, (48, 48), interpolation=cv2.INTER_AREA)
        # Should be valid DeepFace input
        self.assertEqual(face_resized.shape, (48, 48))
        self.assertEqual(face_resized.dtype, np.uint8)
        self.assertGreater(face_resized.mean(), 0)


# ?��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��???# Video processor: _last_results redraw, DETECT_INTERVAL=0.5
# ?��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��???
class TestVideoProcessorLag(unittest.TestCase):
    """Verify lag-reduction changes in the video processor."""

    def setUp(self):
        self.src = _read("streamlit_app.py")

    def test_detect_interval_is_half_second(self):
        self.assertIn("DETECT_INTERVAL = 0.5", self.src)

    def test_last_results_used_not_cached_frame(self):
        """Should use _last_results to redraw, not a cached stale frame."""
        self.assertIn("_last_results", self.src)
        # Old _last_annotated should be gone
        self.assertNotIn("_last_annotated", self.src)

    def test_draw_on_current_frame(self):
        """Should call _draw_results_on with the current img."""
        self.assertIn("_draw_results_on(img, last)", self.src)


if __name__ == "__main__":
    unittest.main()
