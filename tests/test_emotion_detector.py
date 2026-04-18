"""Tests for emotion_detector.py - test detection with mock frames, test fallbacks."""

import pytest
import sys
import os
import numpy as np
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.emotion_detector import EmotionDetector, EMOTION_COLORS


class TestEmotionDetectorInit:
    def test_init_creates_instance(self):
        detector = EmotionDetector()
        assert detector is not None
        assert detector._models_loaded is False

    def test_init_has_lock(self):
        detector = EmotionDetector()
        assert detector._lock is not None

    def test_init_defaults(self):
        detector = EmotionDetector()
        assert detector._backend is None
        assert detector._fer_detector is None
        assert detector._haar_cascade is None
        assert detector._hsemotion_model is None
        assert detector._frame_count == 0


class TestDetectEmotions:
    def test_none_frame_returns_none_and_empty(self):
        detector = EmotionDetector()
        detector._models_loaded = True
        result_frame, results = detector.detect_emotions(None)
        assert result_frame is None
        assert results == []

    def test_empty_frame_returns_original_on_no_backend(self):
        """If no backend loaded, should return original frame + empty results."""
        detector = EmotionDetector()
        detector._models_loaded = True
        detector._backend = None
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result_frame, results = detector.detect_emotions(frame)
        assert result_frame is not None
        assert results == []

    @patch("emotion_detector.EmotionDetector._load_models")
    def test_detect_with_mocked_fer(self, mock_load):
        detector = EmotionDetector()
        detector._models_loaded = True
        detector._backend = "fer"

        mock_fer = MagicMock()
        mock_fer.detect_emotions.return_value = [
            {
                "box": [100, 100, 200, 200],
                "emotions": {
                    "happy": 0.8,
                    "sad": 0.05,
                    "angry": 0.03,
                    "surprise": 0.02,
                    "fear": 0.05,
                    "disgust": 0.02,
                    "neutral": 0.03,
                },
            }
        ]
        detector._fer_detector = mock_fer

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result_frame, results = detector.detect_emotions(frame)

        assert len(results) == 1
        assert results[0]["emotion"] == "happy"
        assert results[0]["confidence"] == 0.8
        assert "all_emotions" in results[0]
        assert "bbox" in results[0]

    @patch("emotion_detector.EmotionDetector._load_models")
    def test_detect_with_empty_fer_results(self, mock_load):
        detector = EmotionDetector()
        detector._models_loaded = True
        detector._backend = "fer"

        mock_fer = MagicMock()
        mock_fer.detect_emotions.return_value = []
        detector._fer_detector = mock_fer

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result_frame, results = detector.detect_emotions(frame)

        assert len(results) == 0
        assert result_frame is not None

    @patch("emotion_detector.EmotionDetector._load_models")
    def test_detect_handles_fer_exception(self, mock_load):
        detector = EmotionDetector()
        detector._models_loaded = True
        detector._backend = "fer"

        mock_fer = MagicMock()
        mock_fer.detect_emotions.side_effect = RuntimeError("model crash")
        detector._fer_detector = mock_fer

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result_frame, results = detector.detect_emotions(frame)

        assert results == []
        assert result_frame is not None

    @patch("emotion_detector.EmotionDetector._load_models")
    def test_multiple_faces_fer(self, mock_load):
        detector = EmotionDetector()
        detector._models_loaded = True
        detector._backend = "fer"

        mock_fer = MagicMock()
        mock_fer.detect_emotions.return_value = [
            {"box": [50, 50, 100, 100], "emotions": {"happy": 0.9, "sad": 0.1}},
            {"box": [300, 50, 100, 100], "emotions": {"sad": 0.7, "happy": 0.3}},
        ]
        detector._fer_detector = mock_fer

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        _, results = detector.detect_emotions(frame)

        assert len(results) == 2
        assert results[0]["emotion"] == "happy"
        assert results[1]["emotion"] == "sad"

    def test_detect_opencv_backend(self):
        """Test OpenCV Haar cascade backend with a real frame."""
        detector = EmotionDetector()
        detector._models_loaded = True
        detector._backend = "opencv"

        import cv2
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        detector._haar_cascade = cv2.CascadeClassifier(cascade_path)

        # Plain black frame - no face to detect (should return empty results)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result_frame, results = detector.detect_emotions(frame)
        assert result_frame is not None
        assert isinstance(results, list)

    def test_frame_count_increments(self):
        detector = EmotionDetector()
        detector._models_loaded = True
        detector._backend = None
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        detector.detect_emotions(frame)
        detector.detect_emotions(frame)
        assert detector._frame_count == 2


class TestEstimateEmotion:
    def test_estimate_emotion_from_valid_face(self):
        detector = EmotionDetector()
        face = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        emotion, confidence, all_emotions = detector._estimate_emotion_heuristic(face)
        assert emotion in ["happy", "sad", "angry", "surprise", "fear", "disgust", "neutral"]
        assert 0 <= confidence <= 1.0
        assert len(all_emotions) == 7
        assert abs(sum(all_emotions.values()) - 1.0) < 0.05  # roughly sums to 1

    def test_estimate_emotion_from_empty_face(self):
        detector = EmotionDetector()
        emotion, confidence, all_emotions = detector._estimate_emotion_heuristic(np.array([]))
        assert emotion == "neutral"
        assert confidence >= 0.5  # fallback returns neutral-dominant scores

    def test_estimate_emotion_from_none(self):
        detector = EmotionDetector()
        emotion, confidence, all_emotions = detector._estimate_emotion_heuristic(None)
        assert emotion == "neutral"

    def test_estimate_varies_with_frame_count(self):
        """Different frame counts should produce slightly different results."""
        detector = EmotionDetector()
        face = np.full((100, 100, 3), 128, dtype=np.uint8)
        detector._frame_count = 1
        _, _, all1 = detector._estimate_emotion_heuristic(face)
        detector._frame_count = 100
        _, _, all2 = detector._estimate_emotion_heuristic(face)
        # At least some values should differ due to different random seeds
        assert all1 != all2


class TestDrawAnnotation:
    def test_draw_annotation_no_crash(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        EmotionDetector._draw_annotation(frame, 10, 50, 100, 100, "happy", 0.9)

    def test_draw_all_emotions(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for emo in ["happy", "sad", "angry", "surprise", "fear", "disgust", "neutral"]:
            EmotionDetector._draw_annotation(frame, 10, 50, 100, 100, emo, 0.5)

    def test_draw_unknown_emotion(self):
        """Unknown emotion should use default white color, not crash."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        EmotionDetector._draw_annotation(frame, 10, 50, 100, 100, "unknown_emo", 0.5)


class TestIsReady:
    @patch("emotion_detector.EmotionDetector._load_models")
    def test_is_ready_with_fer_backend(self, mock_load):
        detector = EmotionDetector()
        detector._models_loaded = True
        detector._backend = "fer"
        assert detector.is_ready is True

    @patch("emotion_detector.EmotionDetector._load_models")
    def test_is_ready_with_opencv_backend(self, mock_load):
        detector = EmotionDetector()
        detector._models_loaded = True
        detector._backend = "opencv"
        assert detector.is_ready is True

    @patch("emotion_detector.EmotionDetector._load_models")
    def test_is_not_ready_without_backend(self, mock_load):
        detector = EmotionDetector()
        detector._models_loaded = True
        detector._backend = None
        assert detector.is_ready is False


class TestBackendName:
    @patch("emotion_detector.EmotionDetector._load_models")
    def test_backend_name_fer(self, mock_load):
        detector = EmotionDetector()
        detector._models_loaded = True
        detector._backend = "fer"
        assert detector.backend_name == "fer"

    @patch("emotion_detector.EmotionDetector._load_models")
    def test_backend_name_none(self, mock_load):
        detector = EmotionDetector()
        detector._models_loaded = True
        detector._backend = None
        assert detector.backend_name == "none"


class TestEmotionColors:
    def test_all_emotions_have_colors(self):
        for emo in ["happy", "sad", "angry", "surprise", "fear", "disgust", "neutral"]:
            assert emo in EMOTION_COLORS
            assert len(EMOTION_COLORS[emo]) == 3