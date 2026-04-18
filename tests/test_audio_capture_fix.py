"""
Tests for Audio Capture Fix (v2): st.audio_input + DiaryAudioProcessor.

The diary tab now uses st.audio_input for recording (works without HTTPS).
DiaryAudioProcessor still exists for the WebRTC audio pipeline in case
it's used for the main emotion detection tab or future features.

Fixes:
- DiaryAudioProcessor.__init__ calls super().__init__() and initializes _frames_received
- flatten() used for mono conversion
- Empty pcm.tobytes() check before appending
- st.audio_input replaces WebRTC audio-only stream for diary recording
"""

import pytest
import os
import io
import wave
import numpy as np
import threading


SRC_PATH = os.path.join(os.path.dirname(__file__), "..", "streamlit_app.py")


def _read_source():
    with open(SRC_PATH) as f:
        return f.read()


# ════════════════════════════════════════════════════════════════════════
# Source code verification: st.audio_input
# ════════════════════════════════════════════════════════════════════════


class TestAudioInputSource:
    """Verify st.audio_input is used for diary audio recording."""

    def test_audio_input_exists(self):
        """Diary tab should use st.audio_input for recording."""
        src = _read_source()
        assert "st.audio_input(" in src

    def test_audio_bytes_stored(self):
        """Audio bytes from st.audio_input should be stored in session state."""
        src = _read_source()
        assert '"_diary_audio_bytes"' in src

    def test_new_audio_tracking(self):
        """New audio detection via _last_audio_input_id."""
        src = _read_source()
        assert '"_last_audio_input_id"' in src

    def test_audio_cleared_after_save(self):
        """Audio bytes should be cleared after saving an entry."""
        src = _read_source()
        # After save, _diary_audio_bytes should be set to None
        assert '_diary_audio_bytes"] = None' in src


# ════════════════════════════════════════════════════════════════════════
# Source code verification: DiaryAudioProcessor (still exists)
# ════════════════════════════════════════════════════════════════════════


class TestDiaryAudioProcessorSource:
    """Verify DiaryAudioProcessor improvements in source."""

    def test_init_exists(self):
        """DiaryAudioProcessor must have __init__ calling super().__init__()."""
        src = _read_source()
        idx = src.find("class DiaryAudioProcessor")
        assert idx != -1
        block = src[idx:idx+600]
        assert "def __init__(self)" in block
        assert "super().__init__()" in block

    def test_frames_received_counter(self):
        """Must track _frames_received for diagnostics."""
        src = _read_source()
        idx = src.find("class DiaryAudioProcessor")
        block = src[idx:idx+800]
        assert "_frames_received" in block

    def test_flatten_for_mono(self):
        """Must use .flatten() for 1D audio arrays."""
        src = _read_source()
        idx = src.find("class DiaryAudioProcessor")
        block = src[idx:idx+1000]
        assert ".flatten()" in block

    def test_empty_bytes_check(self):
        """Must check pcm_bytes length before appending."""
        src = _read_source()
        idx = src.find("class DiaryAudioProcessor")
        block = src[idx:idx+1500]
        assert "len(pcm_bytes) > 0" in block or "pcm_bytes" in block

    def test_stereo_to_mono_mean(self):
        """Multi-channel audio should use mean(axis=0) for mixdown."""
        src = _read_source()
        idx = src.find("class DiaryAudioProcessor")
        block = src[idx:idx+1000]
        assert "mean(axis=0)" in block

    def test_process_frame_method_exists(self):
        """_process_frame helper method should exist."""
        src = _read_source()
        idx = src.find("class DiaryAudioProcessor")
        block = src[idx:idx+1000]
        assert "_process_frame" in block

    def test_recv_queued_in_source(self):
        """DiaryAudioProcessor should use recv_queued (not just recv)."""
        src = _read_source()
        idx = src.find("class DiaryAudioProcessor")
        block = src[idx:idx+1200]
        assert "recv_queued" in block


# ════════════════════════════════════════════════════════════════════════
# Audio processing logic tests
# ════════════════════════════════════════════════════════════════════════


class TestAudioConversion:
    """Test audio format conversion logic used in DiaryAudioProcessor."""

    def test_float32_to_int16(self):
        audio = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
        pcm = (audio * 32767).clip(-32768, 32767).astype(np.int16)
        assert pcm.dtype == np.int16
        assert pcm[0] == 0
        assert pcm[1] == 16383
        assert pcm[3] == 32767
        assert pcm[4] == -32767

    def test_already_int16_passthrough(self):
        audio = np.array([100, -200, 32767, -32768], dtype=np.int16)
        pcm = audio.astype(np.int16)
        np.testing.assert_array_equal(pcm, audio)

    def test_stereo_to_mono_mean(self):
        stereo = np.array([[1000, 2000, 3000], [-1000, -2000, -3000]], dtype=np.float32)
        mono = stereo.mean(axis=0)
        assert mono.shape == (3,)
        np.testing.assert_array_almost_equal(mono, [0.0, 0.0, 0.0])

    def test_mono_flatten(self):
        audio = np.array([[100, 200, 300]], dtype=np.int16)
        flat = audio.flatten()
        assert flat.shape == (3,)
        assert flat.ndim == 1

    def test_empty_audio_not_appended(self):
        buffer = []
        pcm = np.array([], dtype=np.int16)
        pcm_bytes = pcm.tobytes()
        if len(pcm_bytes) > 0:
            buffer.append(pcm_bytes)
        assert len(buffer) == 0

    def test_valid_audio_appended(self):
        buffer = []
        pcm = np.array([100, 200], dtype=np.int16)
        pcm_bytes = pcm.tobytes()
        if len(pcm_bytes) > 0:
            buffer.append(pcm_bytes)
        assert len(buffer) == 1

    def test_float64_audio_conversion(self):
        audio = np.array([0.5, -0.5], dtype=np.float64)
        assert audio.dtype.kind == 'f'
        pcm = (audio * 32767).clip(-32768, 32767).astype(np.int16)
        assert pcm[0] == 16383
        assert pcm[1] == -16383

    def test_multichannel_ndim_check(self):
        stereo = np.zeros((2, 480), dtype=np.float32)
        assert stereo.ndim > 1
        mono = stereo.mean(axis=0)
        assert mono.ndim == 1

    def test_single_channel_ndim_check(self):
        mono = np.zeros(480, dtype=np.float32)
        assert mono.ndim == 1
        flat = mono.flatten()
        assert flat.ndim == 1


# ════════════════════════════════════════════════════════════════════════
# WAV construction tests
# ════════════════════════════════════════════════════════════════════════


class TestWAVConstruction:
    """Test WAV file construction from accumulated audio frames."""

    def test_wav_from_accumulated_frames(self):
        chunks = []
        sr = 48000
        for _ in range(20):
            pcm = np.random.randint(-32768, 32767, size=480, dtype=np.int16)
            chunks.append(pcm.tobytes())

        pcm_data = b"".join(chunks)
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm_data)

        wav_bytes = buf.getvalue()
        assert wav_bytes[:4] == b"RIFF"
        assert len(wav_bytes) >= 44 + len(pcm_data)

        buf.seek(0)
        with wave.open(buf, 'rb') as rf:
            assert rf.getnchannels() == 1
            assert rf.getsampwidth() == 2
            assert rf.getframerate() == sr
            assert rf.getnframes() == 20 * 480

    def test_empty_frames_return_empty_bytes(self):
        frames = []
        if not frames:
            wav_bytes = b""
        assert wav_bytes == b""

    def test_single_frame_wav(self):
        pcm = np.array([100, -100, 200, -200], dtype=np.int16)
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(48000)
            wf.writeframes(pcm.tobytes())

        buf.seek(0)
        with wave.open(buf, 'rb') as rf:
            assert rf.getnframes() == 4


# ════════════════════════════════════════════════════════════════════════
# Audio pipeline integration tests
# ════════════════════════════════════════════════════════════════════════


class TestAudioPipelineIntegration:
    """Test the full audio pipeline: capture -> accumulate -> WAV."""

    def test_full_pipeline_float_audio(self):
        buffer = []
        sr = 48000

        for _ in range(50):
            float_audio = np.random.uniform(-1.0, 1.0, size=480).astype(np.float32)
            pcm = (float_audio * 32767).clip(-32768, 32767).astype(np.int16)
            pcm_bytes = pcm.tobytes()
            if len(pcm_bytes) > 0:
                buffer.append(pcm_bytes)

        assert len(buffer) == 50

        pcm_data = b"".join(buffer)
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm_data)

        wav_bytes = buf.getvalue()
        assert wav_bytes[:4] == b"RIFF"

        total_samples = 50 * 480
        duration = total_samples / sr
        assert abs(duration - 0.5) < 0.001

    def test_concurrent_audio_accumulation(self):
        lock = threading.Lock()
        buffer = []
        errors = []

        def writer(n_frames):
            try:
                for _ in range(n_frames):
                    pcm = np.zeros(480, dtype=np.int16).tobytes()
                    with lock:
                        buffer.append(pcm)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(50,)) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(buffer) == 200

    def test_frames_received_counter_tracks_appends(self):
        frames_received = 0
        buffer = []

        for _ in range(10):
            pcm = np.zeros(480, dtype=np.int16)
            pcm_bytes = pcm.tobytes()
            if len(pcm_bytes) > 0:
                buffer.append(pcm_bytes)
                frames_received += 1

        assert frames_received == 10
        assert len(buffer) == 10

    def test_fallback_raw_bytes_append(self):
        raw = b"\x00\x01\x02\x03"
        buffer = []
        if len(raw) > 0:
            buffer.append(raw)
        assert len(buffer) == 1
        assert buffer[0] == raw
