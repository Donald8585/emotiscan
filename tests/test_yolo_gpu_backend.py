"""Tests for YOLOv8 GPU-accelerated face detection + HSEmotion backend.

Covers:
- Config: YOLO_FACE_CONF, YOLO_FACE_IOU, YOLO_DEVICE, YOLO_MODEL
- EmotionDetector._load_yolo() static method
- EmotionDetector._detect_yolo_hsemotion() backend
- Backend priority: yolo_hsemotion is tried first when HSEmotion is available
- GPU vs CPU device routing
- Integration with temporal smoothing and annotation drawing
"""

import sys
import os
import pytest
import numpy as np
from unittest.mock import patch, MagicMock, PropertyMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ?€?€ Config tests ?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€

class TestYOLOConfig:
    """Verify config.py YOLO settings."""

    def test_yolo_model_is_face_model(self):
        from config import YOLO_MODEL
        assert isinstance(YOLO_MODEL, str)
        assert YOLO_MODEL.endswith(".pt")
        assert "face" in YOLO_MODEL.lower(), "Default model should be face-specific"

    def test_yolo_face_conf_default(self):
        from config import YOLO_FACE_CONF
        assert YOLO_FACE_CONF == 0.40

    def test_yolo_face_iou_default(self):
        from config import YOLO_FACE_IOU
        assert YOLO_FACE_IOU == 0.50

    def test_yolo_device_auto_detected(self):
        from config import YOLO_DEVICE
        assert YOLO_DEVICE in ("cuda", "cpu")

    def test_yolo_device_matches_gpu_available(self):
        from config import YOLO_DEVICE, GPU_AVAILABLE
        if GPU_AVAILABLE:
            assert YOLO_DEVICE == "cuda"
        else:
            assert YOLO_DEVICE == "cpu"


# ?€?€ Backend priority tests ?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€

class TestBackendPriority:
    """YOLOv8 + HSEmotion should be the first backend tried."""

    def test_yolo_hsemotion_is_first_backend_tried(self):
        """When both ultralytics and hsemotion are available, yolo_hsemotion wins."""
        from services.emotion_detector import EmotionDetector

        det = EmotionDetector()

        mock_model = MagicMock()
        mock_model.overrides = {}
        mock_hsemotion = MagicMock()

        with patch.object(EmotionDetector, "_load_yolo", return_value=(mock_model, "cpu")):
            with patch.object(EmotionDetector, "_load_hsemotion", return_value=(mock_hsemotion, "cpu")):
                det._models_loaded = False
                det._load_models()
                assert det._backend == "yolo_hsemotion"
                assert det._yolo_model is mock_model

    def test_falls_back_to_fer_when_yolo_unavailable(self):
        """When YOLO fails, should fall back to FER or opencv_hsemotion."""
        from services.emotion_detector import EmotionDetector

        det = EmotionDetector()
        mock_hsemotion = MagicMock()

        with patch.object(EmotionDetector, "_load_yolo", side_effect=ImportError("no ultralytics")):
            with patch.object(EmotionDetector, "_load_hsemotion", return_value=(mock_hsemotion, "cpu")):
                try:
                    from fer import FER  # noqa
                    det._models_loaded = False
                    det._load_models()
                    if det._backend == "fer":
                        assert True
                    else:
                        assert det._backend in ("fer", "opencv_hsemotion")
                except ImportError:
                    det._models_loaded = False
                    det._load_models()
                    assert det._backend in ("opencv_hsemotion", "opencv_heuristic", None)

    def test_backend_name_includes_yolo(self):
        from services.emotion_detector import EmotionDetector
        det = EmotionDetector()
        det._models_loaded = True
        det._backend = "yolo_hsemotion"
        assert det.backend_name == "yolo_hsemotion"

    def test_yolo_hsemotion_can_classify_emotions(self):
        from services.emotion_detector import EmotionDetector
        det = EmotionDetector()
        det._models_loaded = True
        det._backend = "yolo_hsemotion"
        det._yolo_model = MagicMock()
        det._yolo_device = "cpu"
        det._hsemotion_model = MagicMock()
        status = det.status_info
        assert status["can_classify_emotions"] is True
        assert status["gpu_device"] == "cpu"


# ?€?€ _load_yolo tests ?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€

class TestLoadYolo:
    """Test the YOLO model loading logic."""

    def test_load_yolo_tries_huggingface_first(self):
        """Should try HuggingFace face model before local fallback."""
        from services.emotion_detector import EmotionDetector

        mock_hf = MagicMock(return_value="/fake/model.pt")
        mock_yolo_class = MagicMock()
        mock_model_instance = MagicMock()
        mock_model_instance.overrides = {}
        mock_yolo_class.return_value = mock_model_instance

        with patch.dict("sys.modules", {
            "huggingface_hub": MagicMock(hf_hub_download=mock_hf),
            "ultralytics": MagicMock(YOLO=mock_yolo_class),
            "torch": MagicMock(cuda=MagicMock(is_available=MagicMock(return_value=False))),
        }):
            with patch("config.YOLO_DEVICE", "cpu"):
                model, device = EmotionDetector._load_yolo()
                mock_hf.assert_called_once()
                assert device == "cpu"

    def test_load_yolo_falls_back_to_local_model(self):
        """When HuggingFace fails, should load local YOLO_MODEL."""
        from services.emotion_detector import EmotionDetector

        mock_yolo_class = MagicMock()
        mock_model_instance = MagicMock()
        mock_model_instance.overrides = {}
        mock_yolo_class.return_value = mock_model_instance

        # Make hf_hub_download fail
        mock_hf = MagicMock(side_effect=Exception("offline"))

        with patch.dict("sys.modules", {
            "huggingface_hub": MagicMock(hf_hub_download=mock_hf),
            "ultralytics": MagicMock(YOLO=mock_yolo_class),
            "torch": MagicMock(cuda=MagicMock(is_available=MagicMock(return_value=False))),
        }):
            with patch("config.YOLO_DEVICE", "cpu"):
                model, device = EmotionDetector._load_yolo()
                assert device == "cpu"
                assert mock_yolo_class.call_count == 1  # only local call

    def test_load_yolo_sets_overrides(self):
        """Model should have conf, iou, verbose overrides set."""
        from services.emotion_detector import EmotionDetector

        mock_model = MagicMock()
        mock_model.overrides = {}
        mock_yolo_class = MagicMock(return_value=mock_model)

        with patch.dict("sys.modules", {
            "huggingface_hub": MagicMock(hf_hub_download=MagicMock(side_effect=Exception)),
            "ultralytics": MagicMock(YOLO=mock_yolo_class),
            "torch": MagicMock(cuda=MagicMock(is_available=MagicMock(return_value=False))),
        }):
            with patch("config.YOLO_DEVICE", "cpu"):
                model, device = EmotionDetector._load_yolo()
                assert model.overrides["verbose"] is False
                assert "conf" in model.overrides
                assert "iou" in model.overrides

    def test_load_yolo_gpu_routing(self):
        """When YOLO_DEVICE is cuda and torch.cuda.is_available, should use GPU."""
        from services.emotion_detector import EmotionDetector

        mock_model = MagicMock()
        mock_model.overrides = {}
        mock_yolo_class = MagicMock(return_value=mock_model)

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "NVIDIA RTX 4090"

        with patch.dict("sys.modules", {
            "huggingface_hub": MagicMock(hf_hub_download=MagicMock(side_effect=Exception)),
            "ultralytics": MagicMock(YOLO=mock_yolo_class),
            "torch": mock_torch,
        }):
            with patch("config.YOLO_DEVICE", "cuda"):
                model, device = EmotionDetector._load_yolo()
                assert device == "cuda"
                mock_model.to.assert_called_once_with("cuda")


# ?€?€ _detect_yolo_hsemotion tests ?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€

class TestDetectYoloHSEmotion:
    """Test the actual YOLOv8 + HSEmotion detection pipeline."""

    def _make_detector_with_yolo(self):
        """Create a detector wired to yolo_hsemotion backend with mock YOLO model."""
        from services.emotion_detector import EmotionDetector
        det = EmotionDetector()
        det._models_loaded = True
        det._backend = "yolo_hsemotion"
        det._yolo_model = MagicMock()
        det._yolo_device = "cpu"
        det._hsemotion_model = MagicMock()
        det._hsemotion_device = "cpu"
        return det

    def _make_fake_frame(self, h=480, w=640):
        return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)

    def test_returns_empty_when_no_faces(self):
        det = self._make_detector_with_yolo()
        frame = self._make_fake_frame()

        # YOLO returns empty boxes
        mock_result = MagicMock()
        mock_result.boxes = MagicMock()
        mock_result.boxes.__len__ = MagicMock(return_value=0)
        det._yolo_model.return_value = [mock_result]

        results = det._detect_yolo_hsemotion(frame)
        assert results == []

    def test_returns_empty_when_yolo_returns_none(self):
        det = self._make_detector_with_yolo()
        frame = self._make_fake_frame()
        det._yolo_model.return_value = []
        results = det._detect_yolo_hsemotion(frame)
        assert results == []

    def test_detects_face_and_classifies_emotion(self):
        """Full pipeline: YOLO finds face ??crop ??HSEmotion classifies."""
        det = self._make_detector_with_yolo()
        frame = self._make_fake_frame()

        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")

        mock_box = MagicMock()
        # Real YOLO: box.xyxy is a (N,4) tensor; box.xyxy[0] is shape (4,)
        mock_box.xyxy = torch.tensor([[100, 50, 200, 180]], dtype=torch.float32)
        mock_box.conf = torch.tensor([0.92])

        mock_result = MagicMock()
        mock_result.boxes = [mock_box]
        det._yolo_model.return_value = [mock_result]

        # Mock _classify_hsemotion
        mock_all_emos = {"happy": 0.65, "sad": 0.1, "angry": 0.05,
                         "surprise": 0.08, "fear": 0.03, "disgust": 0.02, "neutral": 0.07}
        with patch.object(det, "_classify_hsemotion", return_value=("happy", 0.65, mock_all_emos)):
            results = det._detect_yolo_hsemotion(frame)

        assert len(results) == 1
        assert results[0]["emotion"] == "happy"
        assert results[0]["bbox"] == [100, 50, 100, 130]  # x, y, w, h
        assert "all_emotions" in results[0]
        assert "face_det_conf" in results[0]
        assert results[0]["face_det_conf"] == 0.92

    def test_skips_small_faces(self):
        """Faces smaller than MIN_FACE_SIZE should be skipped."""
        det = self._make_detector_with_yolo()
        frame = self._make_fake_frame()

        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")

        # Box is only 20px wide (below MIN_FACE_SIZE=30)
        mock_box = MagicMock()
        mock_box.xyxy = torch.tensor([[100, 100, 120, 120]], dtype=torch.float32)
        mock_box.conf = torch.tensor([0.9])

        mock_result = MagicMock()
        mock_result.boxes = [mock_box]
        det._yolo_model.return_value = [mock_result]

        results = det._detect_yolo_hsemotion(frame)
        assert results == []

    def test_handles_hsemotion_failure_gracefully(self):
        """If HSEmotion fails on a crop, should still return a result with neutral."""
        det = self._make_detector_with_yolo()
        frame = self._make_fake_frame()

        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")

        mock_box = MagicMock()
        mock_box.xyxy = torch.tensor([[100, 50, 200, 180]], dtype=torch.float32)
        mock_box.conf = torch.tensor([0.85])

        mock_result = MagicMock()
        mock_result.boxes = [mock_box]
        det._yolo_model.return_value = [mock_result]

        fallback_emos = {e: round(1.0/7, 3) for e in ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]}
        with patch.object(det, "_classify_hsemotion", return_value=("neutral", 0.5, fallback_emos)):
            results = det._detect_yolo_hsemotion(frame)

        assert len(results) == 1
        assert results[0]["emotion"] == "neutral"
        assert results[0]["confidence"] == 0.5

    def test_face_crop_passed_to_classify(self):
        """Face crop from YOLO should be passed to _classify_hsemotion."""
        det = self._make_detector_with_yolo()
        frame = self._make_fake_frame()

        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")

        mock_box = MagicMock()
        mock_box.xyxy = torch.tensor([[50, 50, 200, 200]], dtype=torch.float32)
        mock_box.conf = torch.tensor([0.9])

        mock_result = MagicMock()
        mock_result.boxes = [mock_box]
        det._yolo_model.return_value = [mock_result]

        captured_input = {}

        def capture_classify(face_crop):
            captured_input["shape"] = face_crop.shape
            captured_input["ndim"] = face_crop.ndim
            return "neutral", 0.5, {e: round(1.0/7, 3) for e in ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]}

        with patch.object(det, "_classify_hsemotion", side_effect=capture_classify):
            det._detect_yolo_hsemotion(frame)

        # Input to HSEmotion should be the cropped face (3-channel BGR)
        assert captured_input["ndim"] == 3
        assert captured_input["shape"][2] == 3  # BGR


# ?€?€ Integration with detect_emotions() ?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€

class TestYoloIntegration:
    """Test that detect_emotions() dispatches to yolo_hsemotion correctly."""

    @patch("emotion_detector.EmotionDetector._load_models")
    def test_dispatch_to_yolo_backend(self, mock_load):
        from services.emotion_detector import EmotionDetector
        det = EmotionDetector()
        det._models_loaded = True
        det._backend = "yolo_hsemotion"

        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        with patch.object(det, "_detect_yolo_hsemotion", return_value=[]) as mock_detect:
            det.detect_emotions(frame)
            mock_detect.assert_called_once_with(frame)

    @patch("emotion_detector.EmotionDetector._load_models")
    def test_yolo_results_get_smoothed(self, mock_load):
        """Results from YOLO backend should go through temporal smoothing."""
        from services.emotion_detector import EmotionDetector
        det = EmotionDetector()
        det._models_loaded = True
        det._backend = "yolo_hsemotion"

        face_result = {
            "bbox": [100, 50, 100, 130],
            "emotion": "happy",
            "confidence": 0.65,
            "all_emotions": {"happy": 0.65, "neutral": 0.2, "sad": 0.15},
        }

        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        with patch.object(det, "_detect_yolo_hsemotion", return_value=[face_result]):
            _, results = det.detect_emotions(frame)
            assert len(results) == 1
            assert "all_emotions" in results[0]

    @patch("emotion_detector.EmotionDetector._load_models")
    def test_yolo_results_get_annotated(self, mock_load):
        """Results from YOLO backend should get bounding boxes drawn."""
        from services.emotion_detector import EmotionDetector
        det = EmotionDetector()
        det._models_loaded = True
        det._backend = "yolo_hsemotion"

        face_result = {
            "bbox": [100, 50, 100, 130],
            "emotion": "happy",
            "confidence": 0.65,
            "all_emotions": {"happy": 0.65, "neutral": 0.2, "sad": 0.15},
        }

        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        original = frame.copy()

        with patch.object(det, "_detect_yolo_hsemotion", return_value=[face_result]):
            annotated, results = det.detect_emotions(frame)
            assert not np.array_equal(annotated, original)


# ?€?€ Status info tests ?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€

class TestStatusInfoYolo:

    def test_status_shows_gpu_device(self):
        from services.emotion_detector import EmotionDetector
        det = EmotionDetector()
        det._models_loaded = True
        det._backend = "yolo_hsemotion"
        det._yolo_model = MagicMock()
        det._yolo_device = "cuda"
        det._hsemotion_model = MagicMock()
        status = det.status_info
        assert status["gpu_device"] == "cuda"

    def test_status_shows_cpu_when_no_gpu(self):
        from services.emotion_detector import EmotionDetector
        det = EmotionDetector()
        det._models_loaded = True
        det._backend = "yolo_hsemotion"
        det._yolo_model = MagicMock()
        det._yolo_device = "cpu"
        det._hsemotion_model = MagicMock()
        status = det.status_info
        assert status["gpu_device"] == "cpu"

    def test_status_shows_na_when_no_yolo(self):
        from services.emotion_detector import EmotionDetector
        det = EmotionDetector()
        det._models_loaded = True
        det._backend = "opencv_hsemotion"
        det._yolo_model = None
        det._hsemotion_model = MagicMock()
        status = det.status_info
        assert status["gpu_device"] == "n/a"

    def test_tip_mentions_ultralytics_when_no_classifier(self):
        from services.emotion_detector import EmotionDetector
        det = EmotionDetector()
        det._models_loaded = True
        det._backend = "opencv_heuristic"
        det._yolo_model = None
        det._hsemotion_model = None
        status = det.status_info
        assert "ultralytics" in status["tip"]


# ?€?€ Streamlit display tests ?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€

class TestStreamlitYoloDisplay:
    """Test that streamlit_app.py handles yolo_hsemotion backend display."""

    def test_streamlit_app_references_yolo_hsemotion(self):
        """streamlit_app.py should have display logic for yolo_hsemotion."""
        with open(os.path.join(os.path.dirname(__file__), "..", "streamlit_app.py")) as f:
            source = f.read()
        assert "yolo_hsemotion" in source
        assert "YOLOv8 + HSEmotion" in source
        assert "GPU-accelerated" in source


# ?€?€ Singleton detector tests ?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€

class TestSingletonDetector:
    """Test the shared detector singleton."""

    def test_singleton_returns_same_instance(self):
        import emotion_detector as ed
        ed._singleton_detector = None
        d1 = ed.get_shared_detector()
        d2 = ed.get_shared_detector()
        assert d1 is d2

    def test_singleton_is_emotion_detector(self):
        import emotion_detector as ed
        ed._singleton_detector = None
        d = ed.get_shared_detector()
        assert isinstance(d, ed.EmotionDetector)

    def test_streamlit_uses_shared_detector(self):
        """streamlit_app.py should use get_shared_detector, not EmotionDetector()."""
        with open(os.path.join(os.path.dirname(__file__), "..", "streamlit_app.py")) as f:
            source = f.read()
        assert "get_shared_detector" in source
        assert "self._detector = EmotionDetector()" not in source


# ?€?€ Preprocessing method tests ?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€

class TestPreprocessFaceForEmotion:
    """Test _preprocess_face_for_emotion uses minimal preprocessing."""

    def test_no_histogram_equalization(self):
        """Must NOT use equalizeHist or createCLAHE."""
        import inspect
        from services.emotion_detector import EmotionDetector
        src = inspect.getsource(EmotionDetector._preprocess_face_for_emotion)
        assert "equalizeHist" not in src
        assert "createCLAHE" not in src

    def test_bgr_passthrough(self):
        """3-channel BGR input should be returned as-is (models resize internally)."""
        from services.emotion_detector import EmotionDetector
        face = np.random.randint(0, 255, (100, 80, 3), dtype=np.uint8)
        result = EmotionDetector._preprocess_face_for_emotion(face)
        assert result.shape == (100, 80, 3)
        assert result.ndim == 3
        assert np.array_equal(result, face)

    def test_grayscale_to_bgr_conversion(self):
        """2D grayscale input must be converted to 3-channel BGR."""
        from services.emotion_detector import EmotionDetector
        face = np.random.randint(0, 255, (48, 48), dtype=np.uint8)
        result = EmotionDetector._preprocess_face_for_emotion(face)
        assert result.ndim == 3
        assert result.shape == (48, 48, 3)

    def test_handles_tiny_face(self):
        """Tiny faces pass through ??no resize."""
        from services.emotion_detector import EmotionDetector
        face = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        result = EmotionDetector._preprocess_face_for_emotion(face)
        assert result.shape == (10, 10, 3)
        assert np.array_equal(result, face)

    def test_handles_large_face(self):
        """Large faces pass through ??no resize."""
        from services.emotion_detector import EmotionDetector
        face = np.random.randint(0, 255, (500, 400, 3), dtype=np.uint8)
        result = EmotionDetector._preprocess_face_for_emotion(face)
        assert result.shape == (500, 400, 3)
        assert np.array_equal(result, face)
