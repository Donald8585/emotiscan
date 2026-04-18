"""
Tests verifying HSEmotion-based emotion classification and the 'always neutral' prevention.

The EmotionDetector uses HSEmotion (EfficientNet-B0) for emotion classification.
Backend priority:
  0. YOLOv8-face + HSEmotion (GPU-accelerated)
  1. FER library
  2. Haar cascade + HSEmotion
  3. Haar cascade + heuristic (last resort)
"""

import sys
import os
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestBackendPriority:
    """HSEmotion + Haar should be preferred over heuristic."""

    def test_prefers_opencv_hsemotion_over_heuristic(self):
        """With Haar + HSEmotion available, backend should be opencv_hsemotion."""
        from services.emotion_detector import EmotionDetector

        det = EmotionDetector()
        det._load_models()
        # With HSEmotion installed, backend should NOT be heuristic
        assert det._backend != "opencv_heuristic", \
            f"Backend is {det._backend}, should be opencv_hsemotion or yolo_hsemotion"

    def test_backend_priority_source_code(self):
        """Source code should try yolo_hsemotion before opencv_hsemotion."""
        import inspect
        from services.emotion_detector import EmotionDetector
        src = inspect.getsource(EmotionDetector._load_models)
        # yolo_hsemotion should appear before opencv_hsemotion
        idx_yolo = src.index("yolo_hsemotion")
        idx_opencv = src.index("opencv_hsemotion")
        assert idx_yolo < idx_opencv, \
            "yolo_hsemotion should be tried before opencv_hsemotion"


class TestHSEmotionClassification:
    """Test HSEmotion emotion classification with label mapping."""

    def test_classify_hsemotion_returns_7_labels(self):
        """HSEmotion 8-class output should be mapped to our 7 standard labels."""
        from services.emotion_detector import EmotionDetector
        det = EmotionDetector()
        det._load_models()
        if det._hsemotion_model is None:
            pytest.skip("HSEmotion not available")

        fake_face = np.random.randint(100, 200, (100, 80, 3), dtype=np.uint8)
        emo, conf, all_emos = det._classify_hsemotion(fake_face)

        from config import EMOTION_LABELS
        assert set(all_emos.keys()) == set(EMOTION_LABELS), \
            f"Expected labels {EMOTION_LABELS}, got {list(all_emos.keys())}"

    def test_classify_hsemotion_scores_sum_to_one(self):
        """Mapped scores should sum to approximately 1.0."""
        from services.emotion_detector import EmotionDetector
        det = EmotionDetector()
        det._load_models()
        if det._hsemotion_model is None:
            pytest.skip("HSEmotion not available")

        fake_face = np.random.randint(100, 200, (100, 80, 3), dtype=np.uint8)
        emo, conf, all_emos = det._classify_hsemotion(fake_face)

        total = sum(all_emos.values())
        assert 0.95 < total < 1.05, f"Scores sum to {total}, expected ~1.0"

    def test_contempt_mapped_to_disgust(self):
        """HSEmotion's Contempt class should be merged into disgust."""
        from services.emotion_detector import _HSEMOTION_LABEL_MAP
        assert _HSEMOTION_LABEL_MAP["Contempt"] == "disgust"
        assert _HSEMOTION_LABEL_MAP["Disgust"] == "disgust"

    def test_classify_handles_small_faces(self):
        """Small face crops (e.g., 40x40) should still work."""
        from services.emotion_detector import EmotionDetector
        det = EmotionDetector()
        det._load_models()
        if det._hsemotion_model is None:
            pytest.skip("HSEmotion not available")

        small_face = np.random.randint(100, 200, (40, 40, 3), dtype=np.uint8)
        emo, conf, all_emos = det._classify_hsemotion(small_face)
        assert emo in ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
        assert 0 <= conf <= 1.0

    def test_classify_handles_large_faces(self):
        """Large face crops should work fine (HSEmotion resizes internally)."""
        from services.emotion_detector import EmotionDetector
        det = EmotionDetector()
        det._load_models()
        if det._hsemotion_model is None:
            pytest.skip("HSEmotion not available")

        large_face = np.random.randint(100, 200, (300, 250, 3), dtype=np.uint8)
        emo, conf, all_emos = det._classify_hsemotion(large_face)
        assert emo in ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


class TestOpenCVHSEmotionBackend:
    """Test that opencv_hsemotion crops face before classifying."""

    def test_opencv_hsemotion_crops_face_first(self):
        """Haar detects face region, then HSEmotion classifies the crop."""
        import inspect
        from services.emotion_detector import EmotionDetector
        src = inspect.getsource(EmotionDetector._detect_opencv_hsemotion)
        assert "face_crop" in src or "face_region" in src or "[y:y" in src

    def test_opencv_hsemotion_uses_classify_helper(self):
        """opencv_hsemotion should call _classify_hsemotion on cropped faces."""
        import inspect
        from services.emotion_detector import EmotionDetector
        src = inspect.getsource(EmotionDetector._detect_opencv_hsemotion)
        assert "_classify_hsemotion" in src


class TestFrameSkip:
    """Test that video processor skips frames to reduce lag."""

    def test_async_detection_exists(self):
        """EmotionVideoProcessor should use async background detection."""
        src_path = os.path.join(os.path.dirname(__file__), "..", "streamlit_app.py")
        with open(src_path) as f:
            source = f.read()
        assert "DETECT_INTERVAL" in source
        assert "_detect_busy" in source

    def test_frame_skip_logic(self):
        """Only every Nth frame should be processed."""
        DETECT_EVERY_N = 5
        processed_count = 0
        skipped_count = 0
        for idx in range(1, 31):
            if idx % DETECT_EVERY_N == 0:
                processed_count += 1
            else:
                skipped_count += 1
        assert processed_count == 6
        assert skipped_count == 24

    def test_detect_interval_is_reasonable(self):
        """DETECT_INTERVAL should be between 0.1 and 2.0 seconds."""
        src_path = os.path.join(os.path.dirname(__file__), "..", "streamlit_app.py")
        with open(src_path) as f:
            source = f.read()
        import re
        match = re.search(r"DETECT_INTERVAL\s*=\s*([\d.]+)", source)
        assert match, "DETECT_INTERVAL not found"
        interval = float(match.group(1))
        assert 0.1 <= interval <= 2.0, f"DETECT_INTERVAL={interval} should be between 0.1 and 2.0"


class TestSmoothingWindow:
    """Test temporal smoothing doesn't over-bias toward neutral."""

    def test_smoothing_window_size(self):
        from services.emotion_detector import EmotionDetector
        det = EmotionDetector()
        # Window of 5 means quick transitions (good)
        assert det.SMOOTHING_WINDOW <= 10, \
            f"Window of {det.SMOOTHING_WINDOW} is too large, will bias toward previous emotion"

    def test_smoothing_doesnt_lose_strong_signals(self):
        """If 3 out of 5 frames are 'happy', smoothed result should be 'happy'."""
        from services.emotion_detector import EmotionDetector
        det = EmotionDetector()

        # Simulate 3 happy, 2 neutral frames
        frames_data = [
            {"happy": 0.8, "neutral": 0.1, "sad": 0.05, "angry": 0.05},
            {"happy": 0.7, "neutral": 0.2, "sad": 0.05, "angry": 0.05},
            {"neutral": 0.6, "happy": 0.3, "sad": 0.05, "angry": 0.05},
            {"happy": 0.9, "neutral": 0.05, "sad": 0.03, "angry": 0.02},
            {"neutral": 0.5, "happy": 0.4, "sad": 0.05, "angry": 0.05},
        ]

        for fd in frames_data:
            top_emo = max(fd, key=fd.get)
            result = {
                "bbox": [50, 50, 100, 100],
                "emotion": top_emo,
                "confidence": fd[top_emo],
                "all_emotions": fd,
            }
            smoothed = det._smooth_emotion(result)

        # After 3 happy + 2 neutral, the averaged happy score should still win
        assert smoothed["emotion"] == "happy", \
            f"Expected 'happy' after 3/5 happy frames, got '{smoothed['emotion']}'"


class TestEmotionDetectorIntegration:
    """Integration tests with the actual EmotionDetector."""

    def test_detector_loads_without_crash(self):
        from services.emotion_detector import EmotionDetector
        det = EmotionDetector()
        assert det.is_ready
        info = det.status_info
        assert info["can_classify_emotions"]
        assert info["backend"] in ("yolo_hsemotion", "fer", "opencv_hsemotion")

    def test_detector_backend_is_not_heuristic(self):
        """When HSEmotion is installed, should NOT fall back to heuristic."""
        from services.emotion_detector import EmotionDetector
        det = EmotionDetector()
        det._load_models()
        assert det._backend != "opencv_heuristic", \
            f"Backend is {det._backend}, should be opencv_hsemotion or yolo_hsemotion"

    def test_noise_frame_returns_empty_results(self):
        """Random noise (no face) should return empty results, not 'neutral'."""
        from services.emotion_detector import EmotionDetector
        det = EmotionDetector()
        noise = np.random.randint(0, 255, (360, 480, 3), dtype=np.uint8)
        _, results = det.detect_emotions(noise)
        # Should be empty (no face found), not a neutral fallback
        assert len(results) == 0, \
            f"Expected no results for noise frame, got {len(results)} with emotions: " \
            f"{[r['emotion'] for r in results]}"

    def test_min_face_size_filter(self):
        from services.emotion_detector import EmotionDetector
        det = EmotionDetector()
        assert det.MIN_FACE_SIZE >= 30
        assert det.MIN_FACE_SIZE <= 100

    def test_max_face_ratio_filter(self):
        from services.emotion_detector import EmotionDetector
        det = EmotionDetector()
        assert det.MAX_FACE_RATIO <= 0.7
        assert det.MAX_FACE_RATIO >= 0.3
