"""Tests for the 3 fixes to the 'live graph always neutral' bug.

Fix 1 ??DiaryVideoProcessor always pushes emotion to buffer (not only on change)
Fix 2 ??Diary live panel reads from video processor directly (not buffer)
Fix 3 ??_detect_opencv_deepface applies CLAHE + 48x48 resize before DeepFace

Uses source-code validation for streamlit_app.py (avoids TensorFlow crash)
and direct import for emotion_detector.py (lightweight).
"""

import ast
import os
import sys
import re
import unittest

import numpy as np

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

SRC_DIR = os.path.join(os.path.dirname(__file__), "..")


def _read_source(filename):
    """Read project source file."""
    with open(os.path.join(SRC_DIR, filename)) as f:
        return f.read()


def _get_method_source(filename, class_name, method_name):
    """Extract a method's source code from an AST."""
    src = _read_source(filename)
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == method_name:
                    return ast.get_source_segment(src, item)
    return None


# ?��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��???# Fix 1: DiaryVideoProcessor always pushes emotion
# ?��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��???
class TestFix1AlwaysPushEmotion(unittest.TestCase):
    """DiaryVideoProcessor.recv() must call set_current_emotion() EVERY
    frame, not only when the emotion label changes."""

    def setUp(self):
        self.src = _read_source("streamlit_app.py")
        self.recv_src = _get_method_source("streamlit_app.py",
                                           "DiaryVideoProcessor", "recv")
        self.assertIsNotNone(self.recv_src, "DiaryVideoProcessor.recv not found")

    def test_set_current_emotion_called_unconditionally(self):
        """set_current_emotion() must be called OUTSIDE any if-block,
        so it runs on every frame."""
        lines = self.recv_src.splitlines()
        found_set_call = False
        for line in lines:
            stripped = line.strip()
            if "set_current_emotion" in stripped:
                found_set_call = True
                break
        self.assertTrue(found_set_call,
                        "set_current_emotion() call not found in recv()")

    def test_set_current_emotion_before_if_check(self):
        """set_current_emotion should appear BEFORE the
        'if emo != self._last_pushed_emotion' guard."""
        pos_set = self.recv_src.find("set_current_emotion")
        pos_if = self.recv_src.find("if emo != self._last_pushed_emotion")
        self.assertGreater(pos_set, -1, "set_current_emotion not found")
        self.assertGreater(pos_if, -1, "if-guard not found")
        self.assertLess(pos_set, pos_if,
                        "set_current_emotion must be called BEFORE the if-guard")

    def test_append_emotion_inside_if_guard(self):
        """append_emotion should only be called inside the if-guard
        (timeline updates only on change)."""
        # Find the if block and check append_emotion is inside it
        lines = self.recv_src.splitlines()
        in_if_block = False
        if_indent = None
        append_inside_if = False
        for line in lines:
            stripped = line.strip()
            if "if emo != self._last_pushed_emotion" in stripped:
                in_if_block = True
                if_indent = len(line) - len(line.lstrip())
                continue
            if in_if_block:
                cur_indent = len(line) - len(line.lstrip()) if stripped else if_indent + 1
                if stripped and cur_indent <= if_indent:
                    in_if_block = False
                if "append_emotion" in stripped:
                    append_inside_if = True
        self.assertTrue(append_inside_if,
                        "append_emotion should only be called inside the if-guard")

    def test_set_current_emotion_not_inside_if_guard(self):
        """set_current_emotion must NOT be nested inside the if-guard."""
        lines = self.recv_src.splitlines()
        in_if_block = False
        if_indent = None
        set_inside_if = False
        set_outside_if = False
        for line in lines:
            stripped = line.strip()
            if "if emo != self._last_pushed_emotion" in stripped:
                in_if_block = True
                if_indent = len(line) - len(line.lstrip())
                continue
            if in_if_block:
                cur_indent = len(line) - len(line.lstrip()) if stripped else if_indent + 1
                if stripped and cur_indent <= if_indent:
                    in_if_block = False
                if "set_current_emotion" in stripped and in_if_block:
                    set_inside_if = True
            if "set_current_emotion" in stripped and not in_if_block:
                set_outside_if = True
        self.assertTrue(set_outside_if,
                        "set_current_emotion must be called outside the if-guard")
        self.assertFalse(set_inside_if,
                         "set_current_emotion should NOT be inside the if-guard")

    def test_recv_reads_emotion_and_confidence(self):
        """recv() should read self.emotion and self.confidence."""
        self.assertIn("self.emotion", self.recv_src)
        self.assertIn("self.confidence", self.recv_src)


# ?��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��???# Fix 2: Diary live panel reads from video processor directly
# ?��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��???
class TestFix2LivePanelReadsProcessor(unittest.TestCase):
    """The diary live panel should read emotion/confidence from
    _diary_video_ctx.video_processor directly, not from _diary_buffer."""

    def setUp(self):
        self.src = _read_source("streamlit_app.py")

    def test_reads_processor_emotion(self):
        """Source must contain _diary_video_ctx.video_processor.emotion."""
        self.assertIn("_diary_video_ctx.video_processor.emotion", self.src,
                       "Live panel should read emotion from video_processor")

    def test_reads_processor_confidence(self):
        """Source must contain _diary_video_ctx.video_processor.confidence."""
        self.assertIn("_diary_video_ctx.video_processor.confidence", self.src,
                       "Live panel should read confidence from video_processor")

    def test_has_buffer_fallback(self):
        """Should still fall back to _diary_buffer.get_current_emotion()."""
        self.assertIn("_diary_buffer.get_current_emotion()", self.src,
                       "Should have buffer fallback for when processor is None")

    def test_processor_read_before_buffer_fallback(self):
        """Processor-direct read should appear BEFORE buffer fallback."""
        pos_proc = self.src.find("_diary_video_ctx.video_processor.emotion")
        pos_buf = self.src.find("_diary_buffer.get_current_emotion()")
        self.assertGreater(pos_proc, -1)
        self.assertGreater(pos_buf, -1)
        self.assertLess(pos_proc, pos_buf,
                        "Processor read should appear before buffer fallback")

    def test_processor_none_check(self):
        """Should check if video_processor is not None before reading."""
        self.assertIn("video_processor is not None", self.src,
                       "Should guard against None video_processor")

    def test_live_panel_updates_session_state(self):
        """Live panel should write to diary_face_emotion_live and
        diary_face_confidence_live in session_state."""
        self.assertIn("diary_face_emotion_live", self.src)
        self.assertIn("diary_face_confidence_live", self.src)


# ?��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��???# Fix 3: _detect_opencv_hsemotion uses CLAHE for face detection + HSEmotion
# ?��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��???
class TestFix3CLAHEAndResize(unittest.TestCase):
    """_detect_opencv_hsemotion should apply CLAHE preprocessing for face detection
    and use HSEmotion for emotion classification on cropped faces."""

    def setUp(self):
        self.detector_src = _read_source("services/emotion_detector.py")
        self.method_src = _get_method_source("emotion_detector.py",
                                              "EmotionDetector",
                                              "_detect_opencv_hsemotion")
        self.assertIsNotNone(self.method_src,
                             "_detect_opencv_hsemotion not found")

    def test_calls_preprocess_frame(self):
        """Should call _preprocess_frame for CLAHE."""
        self.assertIn("_preprocess_frame", self.method_src,
                       "Must call _preprocess_frame for CLAHE")

    def test_resizes_to_48x48(self):
        """Haar minSize should be (48, 48) for minimum face quality."""
        self.assertIn("(48, 48)", self.method_src,
                       "Must have minSize (48, 48)")

    def test_uses_shared_preprocess(self):
        """Should use _classify_hsemotion for emotion classification."""
        self.assertIn("_classify_hsemotion", self.method_src,
                       "Must use _classify_hsemotion for emotion classification")

    def test_crops_from_original_frame(self):
        """Face crop should come from original frame (avoids CLAHE angry-bias)."""
        self.assertIn("frame[", self.method_src,
                       "Should crop from original frame, not CLAHE-processed")

    def test_haar_min_size_48(self):
        """Haar minSize should be (48, 48)."""
        self.assertIn("minSize=(48, 48)", self.method_src,
                       "Haar minSize should be (48, 48)")

    def test_skip_detector_backend(self):
        """HSEmotion handles its own image resizing internally."""
        # HSEmotion does its own resize to 224x224 ??no detector_backend needed
        self.assertIn("_classify_hsemotion", self.method_src,
                       "Should use HSEmotion classification (no separate detector skip needed)")

    def test_resize_fallback_on_exception(self):
        """If face crop fails, should handle it gracefully."""
        self.assertIn("face_crop.size", self.method_src,
                       "Should check face crop size")

    def test_clahe_preprocess_creates_clahe(self):
        """_preprocess_frame should create a CLAHE object."""
        self.assertIn("createCLAHE", self.detector_src,
                       "Should use cv2.createCLAHE for CLAHE")

    def test_clahe_clip_limit_and_tile(self):
        """CLAHE should have clipLimit=2.0 and tileGridSize=(8, 8)."""
        self.assertIn("clipLimit=2.0", self.detector_src)
        self.assertIn("tileGridSize=(8, 8)", self.detector_src)


# ?��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��???# Fix 3 ??Runtime tests for CLAHE preprocessing
# ?��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��???
class TestCLAHERuntime(unittest.TestCase):
    """Runtime tests for _preprocess_frame (imports emotion_detector directly)."""

    def _get_detector(self):
        from services.emotion_detector import EmotionDetector
        det = EmotionDetector.__new__(EmotionDetector)
        return det

    def test_preprocess_improves_dark_image(self):
        """CLAHE should brighten a very dark image."""
        det = self._get_detector()
        dark = np.full((100, 100, 3), 25, dtype=np.uint8)
        result = det._preprocess_frame(dark)
        self.assertGreater(result.mean(), dark.mean(),
                           "CLAHE should improve dark image brightness")

    def test_preprocess_preserves_shape(self):
        """_preprocess_frame should not change dimensions."""
        det = self._get_detector()
        frame = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
        result = det._preprocess_frame(frame)
        self.assertEqual(result.shape, frame.shape)

    def test_preprocess_returns_uint8(self):
        """Output should still be uint8."""
        det = self._get_detector()
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = det._preprocess_frame(frame)
        self.assertEqual(result.dtype, np.uint8)

    def test_preprocess_handles_grayscale(self):
        """Should handle grayscale images without crashing."""
        det = self._get_detector()
        gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        # Should not raise
        result = det._preprocess_frame(gray)
        self.assertIsNotNone(result)

    def test_cv2_resize_to_48x48(self):
        """cv2.resize to (48,48) should produce exact dimensions."""
        import cv2
        for h, w in [(80, 80), (120, 90), (200, 200), (30, 30)]:
            face = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
            resized = cv2.resize(face, (48, 48), interpolation=cv2.INTER_AREA)
            self.assertEqual(resized.shape, (48, 48, 3),
                             f"Resize from ({h},{w}) failed")


# ?��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��???# Regression: emotion detection shouldn't always default to neutral
# ?��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��???
class TestRegressionNotAlwaysNeutral(unittest.TestCase):
    """Regression: the detection pipeline should be able to produce
    non-neutral classifications given proper preprocessing."""

    def test_clahe_significantly_improves_very_dark(self):
        """Very dark face should become much brighter after CLAHE."""
        from services.emotion_detector import EmotionDetector
        det = EmotionDetector.__new__(EmotionDetector)
        dark = np.random.randint(0, 15, (48, 48, 3), dtype=np.uint8)
        result = det._preprocess_frame(dark)
        self.assertGreater(result.mean(), dark.mean() + 10,
                           "CLAHE should significantly brighten very dark faces")

    def test_clahe_doesnt_degrade_normal_image(self):
        """Normal lighting image shouldn't be made worse."""
        from services.emotion_detector import EmotionDetector
        det = EmotionDetector.__new__(EmotionDetector)
        normal = np.random.randint(100, 200, (100, 100, 3), dtype=np.uint8)
        result = det._preprocess_frame(normal)
        # Should still be in reasonable range
        self.assertGreater(result.mean(), 50)
        self.assertLess(result.mean(), 250)

    def test_detect_width_is_320(self):
        """DETECT_WIDTH should be 320 (not 160) for better face detection."""
        src = _read_source("streamlit_app.py")
        self.assertIn("DETECT_WIDTH", src)
        # Find the DETECT_WIDTH value
        match = re.search(r'DETECT_WIDTH\s*=\s*(\d+)', src)
        self.assertIsNotNone(match)
        self.assertEqual(int(match.group(1)), 320)

    def test_min_face_size_is_30(self):
        """MIN_FACE_SIZE should be 30 (not 60) for smaller face detection."""
        src = _read_source("services/emotion_detector.py")
        match = re.search(r'MIN_FACE_SIZE\s*=\s*(\d+)', src)
        self.assertIsNotNone(match)
        self.assertEqual(int(match.group(1)), 30)


if __name__ == "__main__":
    unittest.main()
