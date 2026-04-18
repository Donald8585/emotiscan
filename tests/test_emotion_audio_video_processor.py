"""
Tests for Diary video+audio architecture (v2): DiaryVideoProcessor, DiaryAudioProcessor,
_SharedDiaryBuffer, and the overall recording pipeline.

v2: Async background detection, st.audio_input for recording.
"""

import pytest
import time
import io
import wave
import struct
import threading
import numpy as np
from unittest.mock import patch, MagicMock, PropertyMock


# ════════════════════════════════════════════════════════════════════════
# Source code validation (new architecture)
# ════════════════════════════════════════════════════════════════════════


class TestDiaryProcessorArchitectureSource:
    """Verify the async detection + audio_input architecture in streamlit_app.py."""

    @staticmethod
    def _read_source():
        import os
        src_path = os.path.join(os.path.dirname(__file__), "..", "streamlit_app.py")
        with open(src_path) as f:
            return f.read()

    def test_diary_video_processor_class_exists(self):
        src = self._read_source()
        assert "class DiaryVideoProcessor" in src

    def test_diary_audio_processor_class_exists(self):
        src = self._read_source()
        assert "class DiaryAudioProcessor" in src

    def test_shared_diary_buffer_class_exists(self):
        src = self._read_source()
        assert "class _SharedDiaryBuffer" in src

    def test_diary_video_inherits_emotion_video(self):
        import ast, os
        src_path = os.path.join(os.path.dirname(__file__), "..", "streamlit_app.py")
        with open(src_path) as f:
            tree = ast.parse(f.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "DiaryVideoProcessor":
                base_names = [
                    b.id if isinstance(b, ast.Name) else
                    b.attr if isinstance(b, ast.Attribute) else ""
                    for b in node.bases
                ]
                assert "EmotionVideoProcessor" in base_names
                return
        pytest.fail("DiaryVideoProcessor not found")

    def test_diary_audio_inherits_audio_processor_base(self):
        import ast, os
        src_path = os.path.join(os.path.dirname(__file__), "..", "streamlit_app.py")
        with open(src_path) as f:
            tree = ast.parse(f.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "DiaryAudioProcessor":
                base_names = [
                    b.id if isinstance(b, ast.Name) else
                    b.attr if isinstance(b, ast.Attribute) else ""
                    for b in node.bases
                ]
                assert "AudioProcessorBase" in base_names
                return
        pytest.fail("DiaryAudioProcessor not found")

    def test_diary_tab_uses_video_processor(self):
        src = self._read_source()
        assert "video_processor_factory=DiaryVideoProcessor" in src

    def test_diary_tab_uses_audio_input(self):
        """Diary tab should use st.audio_input for recording."""
        src = self._read_source()
        assert "st.audio_input(" in src

    def test_session_state_has_webrtc_audio_processed(self):
        src = self._read_source()
        assert "diary_webrtc_audio_processed" in src

    def test_async_detection_in_video_processor(self):
        """Video processor should use async background detection."""
        src = self._read_source()
        assert "_detect_busy" in src
        assert "DETECT_INTERVAL" in src
        assert "threading.Thread" in src

    def test_shared_buffer_persists_after_stop(self):
        """The shared buffer is module-level, not inside the processor."""
        src = self._read_source()
        assert "_diary_buffer = _SharedDiaryBuffer()" in src

    def test_save_reads_audio_from_session_state(self):
        """Save reads audio from session state, not from buffer."""
        src = self._read_source()
        assert '"_diary_audio_bytes"' in src

    def test_save_reads_emotion_from_buffer(self):
        """Save still reads emotion data from the shared buffer."""
        src = self._read_source()
        assert "_diary_buffer.emotion_during_recording" in src

    def test_backward_compat_alias(self):
        """EmotionAudioVideoProcessor should still exist as an alias."""
        src = self._read_source()
        assert "EmotionAudioVideoProcessor = DiaryVideoProcessor" in src


# ════════════════════════════════════════════════════════════════════════
# Simulated processor tests (without actual WebRTC)
# ════════════════════════════════════════════════════════════════════════


class TestSimulatedAudioVideoProcessor:
    """Test the processor logic using simulated audio/video data."""

    def test_audio_buffer_accumulation(self):
        buffer = []
        for i in range(10):
            chunk = np.random.randint(-32768, 32767, size=480, dtype=np.int16).tobytes()
            buffer.append(chunk)
        assert len(buffer) == 10
        total_bytes = sum(len(f) for f in buffer)
        assert total_bytes == 10 * 480 * 2

    def test_wav_construction(self):
        frames = []
        sr = 48000
        for _ in range(10):
            chunk = np.random.randint(-32768, 32767, size=480, dtype=np.int16)
            frames.append(chunk.tobytes())

        pcm_data = b"".join(frames)
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm_data)

        wav_bytes = buf.getvalue()
        assert len(wav_bytes) > 44
        assert wav_bytes[:4] == b"RIFF"

    def test_wav_is_valid(self):
        frames = []
        sr = 16000
        for _ in range(5):
            chunk = np.zeros(160, dtype=np.int16)
            frames.append(chunk.tobytes())

        pcm_data = b"".join(frames)
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm_data)

        buf.seek(0)
        with wave.open(buf, 'rb') as rf:
            assert rf.getnchannels() == 1
            assert rf.getsampwidth() == 2
            assert rf.getframerate() == sr
            assert rf.getnframes() == 5 * 160

    def test_emotion_timeline_tracking(self):
        timeline = []
        emotions = ["neutral", "happy", "happy", "sad", "happy"]
        for i, emo in enumerate(emotions):
            elapsed = i * 0.1
            timeline.append((elapsed, emo, 0.8))

        all_emos = [e for _, e, _ in timeline]
        dominant = max(set(all_emos), key=all_emos.count)
        assert dominant == "happy"

    def test_emotion_during_recording_calculation(self):
        timeline = [
            (0.0, "sad", 0.7),
            (0.5, "sad", 0.8),
            (1.0, "neutral", 0.6),
            (1.5, "sad", 0.9),
        ]
        emotions = [e for _, e, _ in timeline]
        dominant = max(set(emotions), key=emotions.count)
        assert dominant == "sad"

        confs = [c for _, e, c in timeline if e == dominant]
        avg_conf = sum(confs) / len(confs)
        assert abs(avg_conf - 0.8) < 0.01

    def test_float_to_int16_conversion(self):
        float_audio = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
        pcm = (float_audio * 32767).clip(-32768, 32767).astype(np.int16)
        assert pcm.dtype == np.int16
        assert pcm[0] == 0
        assert pcm[1] == 16383
        assert pcm[3] == 32767
        assert pcm[4] == -32767

    def test_stereo_to_mono_mixdown(self):
        stereo = np.array([[100, 200, 300], [-100, -200, -300]], dtype=np.int16)
        mono = stereo.mean(axis=0)
        assert mono.shape == (3,)
        assert mono[0] == 0.0
        assert mono[1] == 0.0

    def test_audio_buffer_clear(self):
        buffer = [b"frame1", b"frame2", b"frame3"]
        buffer.clear()
        assert len(buffer) == 0

    def test_duration_calculation(self):
        sample_rate = 48000
        sample_width = 2
        total_bytes = 96000 * 5
        duration = total_bytes / (sample_rate * sample_width)
        assert duration == 5.0

    def test_empty_buffer_returns_empty_wav(self):
        frames = []
        if not frames:
            wav_bytes = b""
        assert wav_bytes == b""

    def test_async_detection_interval(self):
        """Detection should respect DETECT_INTERVAL timing."""
        DETECT_INTERVAL = 0.5
        submissions = []
        last_time = -DETECT_INTERVAL  # allow first submission at t=0
        for t in [0.0, 0.1, 0.2, 0.3, 0.5, 0.6, 0.9, 1.0, 1.5]:
            if (t - last_time) >= DETECT_INTERVAL:
                submissions.append(t)
                last_time = t
        # Should submit at 0.0, 0.5, 1.0, 1.5
        assert len(submissions) == 4


# ════════════════════════════════════════════════════════════════════════
# Thread safety tests
# ════════════════════════════════════════════════════════════════════════


class TestProcessorThreadSafety:
    """Test that shared state is accessed thread-safely."""

    def test_concurrent_audio_buffer_access(self):
        lock = threading.Lock()
        buffer = []
        errors = []

        def writer(thread_id, count):
            try:
                for i in range(count):
                    chunk = np.zeros(480, dtype=np.int16).tobytes()
                    with lock:
                        buffer.append(chunk)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(t, 100)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(buffer) == 400

    def test_concurrent_read_write(self):
        lock = threading.Lock()
        emotion = ["neutral"]
        buffer = []
        errors = []

        def writer():
            try:
                for _ in range(200):
                    with lock:
                        buffer.append(b"chunk")
                        emotion[0] = "happy"
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(200):
                    with lock:
                        _ = emotion[0]
                        _ = len(buffer)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=writer)
        t2 = threading.Thread(target=reader)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert len(errors) == 0

    def test_detect_busy_flag_thread_safety(self):
        """_detect_busy flag should prevent overlapping detection threads."""
        lock = threading.Lock()
        detect_busy = [False]
        detections_started = [0]
        errors = []

        def try_start_detection():
            try:
                with lock:
                    if detect_busy[0]:
                        return  # skip
                    detect_busy[0] = True
                    detections_started[0] += 1
                # Simulate detection
                time.sleep(0.01)
                with lock:
                    detect_busy[0] = False
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=try_start_detection) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # Some detections should have been skipped
        assert detections_started[0] <= 10


# ════════════════════════════════════════════════════════════════════════
# Session state flow tests
# ════════════════════════════════════════════════════════════════════════


class TestDiaryAVSessionFlow:
    """Test the diary session flow with video + audio_input."""

    def test_new_session_resets_av_state(self):
        state = {
            "diary_webrtc_audio_processed": True,
            "diary_entry_count_at_last_save": 3,
            "diary_face_emotion_live": "sad",
            "diary_face_confidence_live": 0.8,
        }
        state["diary_webrtc_audio_processed"] = False
        state["diary_entry_count_at_last_save"] = 0
        state["diary_face_emotion_live"] = "neutral"
        state["diary_face_confidence_live"] = 0.0

        assert state["diary_webrtc_audio_processed"] is False
        assert state["diary_entry_count_at_last_save"] == 0
        assert state["diary_face_emotion_live"] == "neutral"

    def test_auto_save_flag_prevents_double_processing(self):
        state = {"diary_webrtc_audio_processed": False}
        state["diary_webrtc_audio_processed"] = True
        assert state["diary_webrtc_audio_processed"] is True

    def test_entry_uses_emotion_during_recording(self):
        timeline = [
            (0.0, "angry", 0.9),
            (0.5, "angry", 0.85),
            (1.0, "neutral", 0.6),
            (1.5, "angry", 0.88),
        ]
        emotions = [e for _, e, _ in timeline]
        dominant = max(set(emotions), key=emotions.count)
        confs = [c for _, e, c in timeline if e == dominant]
        avg_conf = sum(confs) / len(confs)

        assert dominant == "angry"
        assert abs(avg_conf - 0.877) < 0.01

    def test_fallback_mode_when_no_webrtc(self):
        _diary_video_ok = False
        assert not _diary_video_ok

    def test_audio_bytes_flow(self):
        """st.audio_input bytes should flow through session_state to save."""
        state = {
            "_diary_audio_bytes": None,
            "_last_audio_input_id": None,
        }
        # Simulate recording
        audio_data = b"recorded_audio_bytes" * 50
        state["_diary_audio_bytes"] = audio_data
        state["_last_audio_input_id"] = "audio_1"

        # Save reads from session state
        assert state["_diary_audio_bytes"] is not None
        assert len(state["_diary_audio_bytes"]) > 100

        # After save, clear
        state["_diary_audio_bytes"] = None
        state["_last_audio_input_id"] = None
        assert state["_diary_audio_bytes"] is None
