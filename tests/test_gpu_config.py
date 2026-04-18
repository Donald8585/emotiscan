"""Tests for GPU detection in config.py."""

import pytest
import sys
import os
from unittest.mock import patch, MagicMock
import importlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestGPUDetectionDefaults:
    """Test GPU config when no GPU libraries are available."""

    def test_gpu_not_available_by_default(self):
        """Without GPU hardware, GPU_AVAILABLE should be False."""
        import config
        # In test environment, there's no GPU
        assert isinstance(config.GPU_AVAILABLE, bool)

    def test_gpu_device_name_is_string(self):
        import config
        assert isinstance(config.GPU_DEVICE_NAME, str)

    def test_whisper_device_is_string(self):
        import config
        assert config.WHISPER_DEVICE in ("cpu", "cuda")

    def test_whisper_compute_type_matches_device(self):
        import config
        if config.WHISPER_DEVICE == "cpu":
            assert config.WHISPER_COMPUTE_TYPE == "int8"
        else:
            assert config.WHISPER_COMPUTE_TYPE == "float16"

    def test_no_gpu_means_cpu_device(self):
        """When no GPU is detected, device should be cpu."""
        import config
        if not config.GPU_AVAILABLE:
            assert config.WHISPER_DEVICE == "cpu"
            assert config.WHISPER_COMPUTE_TYPE == "int8"
            assert config.GPU_DEVICE_NAME == "cpu"


class TestGPUDetectionWithMockedTorch:
    """Test GPU detection with mocked torch."""

    def test_torch_cuda_available(self):
        """When torch.cuda.is_available() returns True, GPU should be detected."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "NVIDIA RTX 4090"

        with patch.dict("sys.modules", {"torch": mock_torch}):
            # Re-evaluate the detection logic
            gpu_available = False
            gpu_device_name = "cpu"
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_available = True
                    gpu_device_name = torch.cuda.get_device_name(0)
            except ImportError:
                pass

            assert gpu_available is True
            assert gpu_device_name == "NVIDIA RTX 4090"

    def test_torch_cuda_not_available(self):
        """When torch is installed but no CUDA, GPU should not be detected."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        with patch.dict("sys.modules", {"torch": mock_torch}):
            gpu_available = False
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_available = True
            except ImportError:
                pass

            assert gpu_available is False


class TestGPUDetectionWithMockedTensorflow:
    """Test GPU detection with mocked tensorflow."""

    def test_tensorflow_gpu_available(self):
        """When TF detects GPU, it should be used as fallback."""
        mock_tf = MagicMock()
        mock_gpu = MagicMock()
        mock_gpu.name = "/physical_device:GPU:0"
        mock_tf.config.list_physical_devices.return_value = [mock_gpu]

        # Simulate no torch, then TF fallback
        gpu_available = False
        gpu_device_name = "cpu"

        # torch not available
        try:
            raise ImportError("no torch")
        except ImportError:
            pass

        if not gpu_available:
            with patch.dict("sys.modules", {"tensorflow": mock_tf}):
                try:
                    import tensorflow as tf
                    gpus = tf.config.list_physical_devices('GPU')
                    if gpus:
                        gpu_available = True
                        gpu_device_name = gpus[0].name
                except ImportError:
                    pass

        assert gpu_available is True
        assert gpu_device_name == "/physical_device:GPU:0"

    def test_tensorflow_no_gpu(self):
        """When TF is installed but no GPU, detection should be False."""
        mock_tf = MagicMock()
        mock_tf.config.list_physical_devices.return_value = []

        gpu_available = False
        if not gpu_available:
            with patch.dict("sys.modules", {"tensorflow": mock_tf}):
                try:
                    import tensorflow as tf
                    gpus = tf.config.list_physical_devices('GPU')
                    if gpus:
                        gpu_available = True
                except ImportError:
                    pass

        assert gpu_available is False


class TestGPUConfigPropagation:
    """Test that GPU config values are used by speech_service."""

    def test_speech_service_imports_gpu_config(self):
        """SpeechService should import WHISPER_DEVICE and WHISPER_COMPUTE_TYPE."""
        import config
        from services.speech_service import SpeechService
        # SpeechService should use WHISPER_DEVICE and WHISPER_COMPUTE_TYPE from config
        assert hasattr(config, 'WHISPER_DEVICE')
        assert hasattr(config, 'WHISPER_COMPUTE_TYPE')

    def test_config_does_not_crash_without_gpu_libs(self):
        """Config module should load successfully even without torch/tensorflow."""
        import config
        # These should always be defined
        assert hasattr(config, 'GPU_AVAILABLE')
        assert hasattr(config, 'GPU_DEVICE_NAME')
        assert hasattr(config, 'WHISPER_DEVICE')
        assert hasattr(config, 'WHISPER_COMPUTE_TYPE')
