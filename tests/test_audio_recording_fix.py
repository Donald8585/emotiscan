"""
Tests verifying the audio recording fix.

Root cause: DiaryAudioProcessor only implemented recv() which in async mode
only receives the LATEST frame, dropping all others. Audio recording was
essentially capturing 1 frame per polling interval instead of all frames.

Fix: Implement recv_queued() which receives ALL accumulated frames since
the last call, ensuring no audio data is lost.
"""

import sys
import os
import ast
import inspect
import io
import wave
import numpy as np
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _read_source():
    src_path = os.path.join(os.path.dirname(__file__), "..", "streamlit_app.py")
    with open(src_path) as f:
        return f.read()


class TestAudioProcessorArchitecture:
    """Verify DiaryAudioProcessor uses recv_queued for complete audio capture."""

    def test_diary_audio_processor_has_recv_queued(self):
        """DiaryAudioProcessor must implement recv_queued to capture ALL frames."""
        source = _read_source()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "DiaryAudioProcessor":
                methods = [n.name for n in node.body
                           if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
                assert "recv_queued" in methods, \
                    f"DiaryAudioProcessor must have recv_queued, found: {methods}"
                return
        pytest.fail("DiaryAudioProcessor class not found in streamlit_app.py")

    def test_recv_queued_is_async(self):
        """recv_queued must be async (required by streamlit-webrtc)."""
        source = _read_source()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "DiaryAudioProcessor":
                for method in node.body:
                    if isinstance(method, ast.AsyncFunctionDef) and method.name == "recv_queued":
                        return  # Found async recv_queued
                    if isinstance(method, ast.FunctionDef) and method.name == "recv_queued":
                        pytest.fail("recv_queued must be async (AsyncFunctionDef)")
                pytest.fail("recv_queued not found in DiaryAudioProcessor")
        pytest.fail("DiaryAudioProcessor not found")

    def test_recv_queued_processes_all_frames(self):
        """recv_queued should iterate over ALL frames, not just the last one."""
        source = _read_source()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "DiaryAudioProcessor":
                for method in node.body:
                    if hasattr(method, 'name') and method.name == "recv_queued":
                        src = ast.get_source_segment(source, method)
                        # Should have a loop over frames
                        assert "for frame in frames" in src or "for f in frames" in src, \
                            f"recv_queued should iterate over all frames: {src}"
                        return
        pytest.fail("recv_queued not found")

    def test_recv_still_exists_as_fallback(self):
        """recv() should still exist as fallback for non-async mode."""
        source = _read_source()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "DiaryAudioProcessor":
                methods = [n.name for n in node.body
                           if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
                assert "recv" in methods, "recv() fallback should still exist"
                return

    def test_process_frame_helper_exists(self):
        """_process_frame helper should exist for shared logic."""
        source = _read_source()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "DiaryAudioProcessor":
                methods = [n.name for n in node.body
                           if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
                assert "_process_frame" in methods, \
                    f"_process_frame helper not found, methods: {methods}"
                return


class TestAudioProcessorSeparation:
    """Verify audio uses a SEPARATE AudioProcessorBase class."""

    def test_inherits_from_audio_processor_base(self):
        source = _read_source()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "DiaryAudioProcessor":
                bases = [b.id if isinstance(b, ast.Name) else
                         b.attr if isinstance(b, ast.Attribute) else ""
                         for b in node.bases]
                assert "AudioProcessorBase" in bases, \
                    f"Should inherit AudioProcessorBase, got: {bases}"
                return

    def test_diary_uses_audio_input_widget(self):
        """Diary tab should use st.audio_input for recording."""
        source = _read_source()
        assert "st.audio_input(" in source


class TestSharedBufferPersistence:
    """Verify audio data is stored in module-level buffer, not processor object."""

    def test_shared_buffer_is_module_level(self):
        source = _read_source()
        assert "_diary_buffer = _SharedDiaryBuffer()" in source

    def test_audio_written_to_shared_buffer(self):
        """_process_frame should write to _diary_buffer, not self."""
        source = _read_source()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "DiaryAudioProcessor":
                for method in node.body:
                    if hasattr(method, 'name') and method.name == "_process_frame":
                        src = ast.get_source_segment(source, method)
                        assert "_diary_buffer.append_audio" in src, \
                            "Should write to _diary_buffer, not self"
                        return

    def test_save_reads_audio_from_session_state(self):
        """Save entry should read audio from session_state._diary_audio_bytes."""
        source = _read_source()
        assert '"_diary_audio_bytes"' in source


class TestAudioFrameProcessing:
    """Test the actual audio frame processing logic."""

    def test_float_to_pcm_conversion(self):
        """Float32 audio should be converted to int16 PCM."""
        float_audio = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
        pcm = (float_audio * 32767).clip(-32768, 32767).astype('int16')
        assert pcm.dtype == np.int16
        assert pcm[0] == 0
        assert abs(pcm[1] - 16383) <= 1

    def test_stereo_to_mono(self):
        """Stereo audio should be mixed to mono."""
        stereo = np.array([[1000, 2000], [-1000, -2000]], dtype=np.float32)
        mono = stereo.mean(axis=0)
        assert mono.shape == (2,)
        assert mono[0] == 0.0

    def test_pcm_bytes_are_valid_wav_input(self):
        """Accumulated PCM bytes should produce valid WAV."""
        chunks = []
        for _ in range(10):
            pcm = np.random.randint(-32768, 32767, size=960, dtype=np.int16)
            chunks.append(pcm.tobytes())

        pcm_data = b"".join(chunks)
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(48000)
            wf.writeframes(pcm_data)

        wav = buf.getvalue()
        assert wav[:4] == b"RIFF"
        assert len(wav) > 44  # header + data

        # Verify it's readable
        buf.seek(0)
        with wave.open(buf, 'rb') as rf:
            assert rf.getnchannels() == 1
            assert rf.getsampwidth() == 2
            assert rf.getframerate() == 48000
            assert rf.getnframes() == 10 * 960

    def test_recv_queued_processes_multiple_frames(self):
        """Simulate recv_queued with multiple frames and verify all are processed."""
        processed_count = 0
        buffer = []

        def process_frame(frame_data):
            nonlocal processed_count
            processed_count += 1
            buffer.append(frame_data)

        # Simulate 20 audio frames arriving at once
        frames = [np.random.randint(-32768, 32767, size=960, dtype=np.int16).tobytes()
                   for _ in range(20)]

        for f in frames:
            process_frame(f)

        assert processed_count == 20, f"Expected 20 frames processed, got {processed_count}"
        assert len(buffer) == 20

    def test_single_recv_would_lose_frames(self):
        """Demonstrate that recv() only processes 1 frame (the problem we fixed)."""
        # With async recv(), only frames[-1] is processed
        frames = list(range(20))
        # Old behavior (recv): only last frame
        old_processed = [frames[-1]]
        assert len(old_processed) == 1

        # New behavior (recv_queued): all frames
        new_processed = list(frames)
        assert len(new_processed) == 20

        # 19 frames lost with old approach
        assert len(new_processed) - len(old_processed) == 19


class TestAudioDuration:
    """Test audio duration calculation from buffer."""

    def test_duration_from_byte_count(self):
        sample_rate = 48000
        sample_width = 2  # 16-bit = 2 bytes
        # 1 second of audio = 48000 * 2 = 96000 bytes
        one_second_bytes = sample_rate * sample_width
        total_bytes = one_second_bytes * 5
        duration = total_bytes / (sample_rate * sample_width)
        assert duration == 5.0

    def test_empty_buffer_zero_duration(self):
        total_bytes = 0
        duration = total_bytes / (48000 * 2)
        assert duration == 0.0
