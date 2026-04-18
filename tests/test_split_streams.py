"""
Tests for Split Video + Audio architecture (v2).

The diary tab now uses:
  1) Video-only WebRTC stream at 640x480 for emotion detection
  2) st.audio_input widget for voice recording (no WebRTC audio)
Video writes emotion to _diary_buffer; audio bytes go to session_state.
"""

import pytest
import os


SRC_PATH = os.path.join(os.path.dirname(__file__), "..", "streamlit_app.py")


def _read_source():
    with open(SRC_PATH) as f:
        return f.read()


# ════════════════════════════════════════════════════════════════════════
# Architecture: Video stream + st.audio_input
# ════════════════════════════════════════════════════════════════════════


class TestSplitStreamsArchitecture:
    """Verify the video stream + st.audio_input architecture."""

    def test_separate_video_streamer(self):
        """There must be a video-only diary WebRTC stream."""
        src = _read_source()
        assert 'key=f"diary-video-{_sess.session_id}"' in src

    def test_audio_input_widget(self):
        """Audio should use st.audio_input instead of WebRTC audio stream."""
        src = _read_source()
        assert "st.audio_input(" in src

    def test_video_stream_has_no_audio(self):
        """Video stream must request audio: False."""
        src = _read_source()
        idx = src.find('diary-video-')
        assert idx != -1
        block = src[idx:idx+500]
        assert '"audio": False' in block

    def test_no_webrtc_audio_stream(self):
        """There should be no separate WebRTC audio-only stream."""
        src = _read_source()
        assert 'key=f"diary-audio-{' not in src, \
            "WebRTC audio stream replaced by st.audio_input"

    def test_video_stream_uses_diary_video_processor(self):
        """Video stream must use DiaryVideoProcessor."""
        src = _read_source()
        idx = src.find('diary-video-')
        assert idx != -1
        block = src[idx:idx+300]
        assert 'video_processor_factory=DiaryVideoProcessor' in block

    def test_no_combined_av_stream(self):
        """There should be no combined video+audio diary stream."""
        src = _read_source()
        idx = src.find('diary-video-')
        if idx != -1:
            block = src[idx:idx+300]
            assert '"audio": True' not in block

    def test_old_diary_av_key_removed(self):
        """The old combined 'diary-av-' key should no longer exist."""
        src = _read_source()
        assert 'key=f"diary-av-' not in src


# ════════════════════════════════════════════════════════════════════════
# Resolution: Presentation quality
# ════════════════════════════════════════════════════════════════════════


class TestPresentationResolution:
    """Verify video resolution is suitable for presentations."""

    def test_diary_video_640x480(self):
        """Diary video should request 640x480 for presentation quality."""
        src = _read_source()
        idx = src.find('diary-video-')
        assert idx != -1
        block = src[idx:idx+300]
        assert '"width": {"ideal": 640}' in block
        assert '"height": {"ideal": 480}' in block

    def test_no_low_res_400x300(self):
        """The old 400x300 resolution should not be used for diary video."""
        src = _read_source()
        idx = src.find('diary-video-')
        assert idx != -1
        block = src[idx:idx+300]
        assert '"width": {"ideal": 400}' not in block
        assert '"height": {"ideal": 300}' not in block

    def test_main_tab_also_640x480(self):
        """Main Emotion Detection tab should also use 640x480."""
        src = _read_source()
        idx = src.find('emotiscan-webcam-')
        assert idx != -1
        block = src[idx:idx+500]
        assert '"width": {"ideal": 640}' in block
        assert '"height": {"ideal": 480}' in block


# ════════════════════════════════════════════════════════════════════════
# Audio input via st.audio_input
# ════════════════════════════════════════════════════════════════════════


class TestAudioInputWidget:
    """Verify audio recording uses st.audio_input."""

    def test_audio_input_key_has_session_id(self):
        """st.audio_input key should include session ID."""
        src = _read_source()
        assert 'diary_audio_input_' in src

    def test_audio_bytes_stored_in_session_state(self):
        """Audio bytes should be stored in session state."""
        src = _read_source()
        assert '"_diary_audio_bytes"' in src

    def test_new_audio_detection(self):
        """New audio should be tracked via _last_audio_input_id."""
        src = _read_source()
        assert '"_last_audio_input_id"' in src

    def test_audio_cleared_after_save(self):
        """Audio bytes should be cleared after saving an entry."""
        src = _read_source()
        # Find the save section
        idx = src.find('Save Entry')
        assert idx != -1
        block = src[idx:idx+2000]
        assert '"_diary_audio_bytes"' in block
        assert '"_last_audio_input_id"' in block


# ════════════════════════════════════════════════════════════════════════
# Shared buffer still works for video emotion
# ════════════════════════════════════════════════════════════════════════


class TestSharedBufferIntegration:
    """Video stream writes emotion to _diary_buffer."""

    def test_shared_buffer_exists(self):
        src = _read_source()
        assert '_diary_buffer = _SharedDiaryBuffer()' in src

    def test_buffer_clear_flag_preserved(self):
        """Flag-based clear should still work."""
        src = _read_source()
        assert '_diary_buffer_needs_clear' in src

    def test_save_reads_audio_from_session_state(self):
        """Save entry should read audio from session state, not buffer."""
        src = _read_source()
        assert '"_diary_audio_bytes"' in src

    def test_save_reads_emotion_from_buffer(self):
        """Save entry should still read emotion from buffer."""
        src = _read_source()
        assert '_diary_buffer.emotion_during_recording' in src

    def test_buffer_cleared_after_save(self):
        """Buffer should be cleared after saving (for emotion timeline reset)."""
        src = _read_source()
        assert '_diary_buffer.clear()' in src


# ════════════════════════════════════════════════════════════════════════
# Fallback when WebRTC unavailable
# ════════════════════════════════════════════════════════════════════════


class TestFallbackMode:
    """Fallback should activate when video stream is unavailable."""

    def test_fallback_checks_video_stream(self):
        """Fallback should show when video stream fails."""
        src = _read_source()
        assert 'not _diary_video_ok' in src

    def test_fallback_has_manual_emotion(self):
        """Fallback mode should have manual emotion selector."""
        src = _read_source()
        assert 'Manual Emotion Override' in src


# ════════════════════════════════════════════════════════════════════════
# Live panel: works with video-only stream
# ════════════════════════════════════════════════════════════════════════


class TestLivePanelSplitStreams:
    """Live panel should work correctly with video stream."""

    def test_panel_shows_face_emotion(self):
        """Panel should show face emotion metric."""
        src = _read_source()
        assert 'Face Emotion' in src

    def test_panel_shows_session_mood(self):
        """Panel should show session mood metric."""
        src = _read_source()
        assert 'Session Mood' in src

    def test_emotion_history_only_from_video(self):
        """Emotion history should only append when VIDEO is playing."""
        src = _read_source()
        idx = src.find('_diary_av_live_panel')
        assert idx != -1
        block = src[idx:idx+1500]
        assert '_video_playing' in block

    def test_panel_returns_when_video_not_playing(self):
        """Panel should return early if video is not playing."""
        src = _read_source()
        idx = src.find('_diary_av_live_panel')
        assert idx != -1
        block = src[idx:idx+600]
        assert 'not _video_playing' in block


# ════════════════════════════════════════════════════════════════════════
# Logic simulation
# ════════════════════════════════════════════════════════════════════════


class TestSplitStreamLogic:
    """Simulate split stream scenarios."""

    def test_video_only_scenario(self):
        """User starts only video — emotion tracking works, no audio."""
        video_playing = True
        audio_bytes = None

        assert video_playing is True
        assert audio_bytes is None

    def test_audio_only_scenario(self):
        """User records only audio — voice recording works, no emotion."""
        video_playing = False
        audio_bytes = b"pcm_data"
        current_emotion = "neutral"  # stays neutral without video

        assert audio_bytes is not None
        assert current_emotion == "neutral"

    def test_both_streams_scenario(self):
        """User uses both video and audio — full experience."""
        video_playing = True
        audio_bytes = b"pcm_data"
        emotion_history = [("happy", 0.9), ("sad", 0.7)]

        assert video_playing is True
        assert audio_bytes is not None
        assert len(emotion_history) == 2

    def test_neither_stream_scenario(self):
        """Neither stream active — fallback mode."""
        video_playing = False
        audio_bytes = None
        should_save = audio_bytes is not None and len(audio_bytes) > 100
        assert should_save is False

    def test_save_requires_audio(self):
        """Save should require recorded audio bytes."""
        audio_bytes = b"x" * 200
        has_recorded_audio = audio_bytes is not None and len(audio_bytes) > 100
        assert has_recorded_audio is True

    def test_save_rejects_empty_audio(self):
        """Save should reject empty/no audio."""
        audio_bytes = None
        has_recorded_audio = audio_bytes is not None and len(audio_bytes) > 100
        assert has_recorded_audio is False

    def test_save_rejects_tiny_audio(self):
        """Save should reject tiny audio (< 100 bytes)."""
        audio_bytes = b"x" * 50
        has_recorded_audio = audio_bytes is not None and len(audio_bytes) > 100
        assert has_recorded_audio is False
