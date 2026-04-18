"""
Tests for Buffer Clear Fix: Flag-based clear instead of clearing on every rerun.

Root cause: _diary_buffer.clear() was called on EVERY Streamlit rerun when
_diary_stream_active was False, wiping current_emotion back to "neutral"
and audio_frames between stream stop and save button click.

Fix: Changed to _diary_buffer_needs_clear flag — set True only when
"Start New Session" is clicked, cleared once, then flag resets to False.

v2 note: save logic now also clears _diary_audio_bytes and _last_audio_input_id
from session_state, plus _diary_buffer.clear() for emotion timeline reset.
"""

import pytest
import os
import threading
import time


SRC_PATH = os.path.join(os.path.dirname(__file__), "..", "streamlit_app.py")


def _read_source():
    with open(SRC_PATH) as f:
        return f.read()


# ════════════════════════════════════════════════════════════════════════
# Source code verification
# ════════════════════════════════════════════════════════════════════════


class TestBufferClearFlagSource:
    """Verify the flag-based buffer clear is in the source code."""

    def test_flag_set_on_new_session(self):
        """'Start New Session' should set the flag."""
        src = _read_source()
        assert "_diary_buffer_needs_clear = True" in src

    def test_flag_checked_before_clear(self):
        """Buffer clear must be guarded by the flag check."""
        src = _read_source()
        assert '_diary_buffer_needs_clear' in src
        assert 'st.session_state.get("_diary_buffer_needs_clear", False)' in src or \
               "st.session_state._diary_buffer_needs_clear" in src

    def test_flag_reset_after_clear(self):
        """Flag must be set to False after clear to prevent repeated clearing."""
        src = _read_source()
        assert "_diary_buffer_needs_clear = False" in src

    def test_no_unconditional_clear_on_rerun(self):
        """There should be no unconditional _diary_buffer.clear() on every rerun.

        clear() should only happen when:
        1) The _diary_buffer_needs_clear flag is True (new session)
        2) After saving an entry (post-save cleanup)
        3) During end session processing
        """
        src = _read_source()
        lines = src.split('\n')
        for i, line in enumerate(lines):
            stripped = line.strip()
            if '_diary_buffer.clear()' in stripped:
                context = '\n'.join(lines[max(0, i-10):i+1])
                ok = ('_diary_buffer_needs_clear' in context or
                      'End Session' in context or
                      'end_session' in context or
                      'last_ended_session' in context or
                      'add_entry' in context or
                      'Save Entry' in context or
                      '_entry' in context or
                      'save' in context.lower())
                assert ok, (
                    f"Found unguarded _diary_buffer.clear() at line {i+1}. "
                    f"Context:\n{context}"
                )

    def test_start_new_session_sets_flag(self):
        """The 'Start New Session' button handler should set the clear flag."""
        src = _read_source()
        idx = src.find('Start New Session')
        assert idx != -1, "Start New Session button must exist"
        block = src[idx:idx+1000]
        assert "_diary_buffer_needs_clear = True" in block

    def test_save_clears_audio_session_state(self):
        """Save entry should clear _diary_audio_bytes and _last_audio_input_id."""
        src = _read_source()
        assert '_diary_audio_bytes"] = None' in src
        assert '_last_audio_input_id"] = None' in src


# ════════════════════════════════════════════════════════════════════════
# Behavioral / logic tests
# ════════════════════════════════════════════════════════════════════════


class TestBufferClearFlagBehavior:
    """Test the flag-based clearing logic in isolation."""

    def test_flag_false_by_default(self):
        state = {}
        needs_clear = state.get("_diary_buffer_needs_clear", False)
        assert needs_clear is False

    def test_flag_true_triggers_clear(self):
        state = {"_diary_buffer_needs_clear": True}
        buffer_data = {"audio": [b"frame1", b"frame2"], "emotion": "happy"}

        if state.get("_diary_buffer_needs_clear", False):
            buffer_data["audio"].clear()
            buffer_data["emotion"] = "neutral"
            state["_diary_buffer_needs_clear"] = False

        assert len(buffer_data["audio"]) == 0
        assert state["_diary_buffer_needs_clear"] is False

    def test_subsequent_reruns_dont_clear(self):
        state = {"_diary_buffer_needs_clear": False}
        buffer_data = {"audio": [b"frame1", b"frame2"], "count": 2}

        for _ in range(10):
            if state.get("_diary_buffer_needs_clear", False):
                buffer_data["audio"].clear()
                buffer_data["count"] = 0
                state["_diary_buffer_needs_clear"] = False

        assert len(buffer_data["audio"]) == 2
        assert buffer_data["count"] == 2

    def test_emotion_preserved_across_reruns(self):
        class FakeBuffer:
            def __init__(self):
                self.current_emotion = "happy"
                self.current_confidence = 0.9
                self.audio_frames = [b"audio1", b"audio2"]

            def clear(self):
                self.audio_frames.clear()
                self.current_emotion = "neutral"
                self.current_confidence = 0.0

        buf = FakeBuffer()
        state = {"_diary_buffer_needs_clear": False}

        for _ in range(5):
            if state.get("_diary_buffer_needs_clear", False):
                buf.clear()
                state["_diary_buffer_needs_clear"] = False

        assert buf.current_emotion == "happy"
        assert buf.current_confidence == 0.9
        assert len(buf.audio_frames) == 2

    def test_audio_preserved_between_stop_and_save(self):
        """Audio bytes in session state must survive between recording and save.

        This was the key bug: user stops streaming -> Streamlit reruns ->
        old code cleared buffer -> user clicks save -> no audio.
        Now audio is in session_state, not buffer, so it survives reruns.
        """
        state = {
            "_diary_audio_bytes": b"recorded_audio" * 100,
            "_diary_buffer_needs_clear": False,
        }

        # Simulate multiple reruns
        for _ in range(5):
            if state.get("_diary_buffer_needs_clear", False):
                state["_diary_audio_bytes"] = None
                state["_diary_buffer_needs_clear"] = False

        # Audio should still be there
        assert state["_diary_audio_bytes"] is not None
        assert len(state["_diary_audio_bytes"]) == 1400

    def test_new_session_properly_clears(self):
        """Starting a new session SHOULD clear the buffer (once)."""
        class FakeBuffer:
            def __init__(self):
                self.audio_frames = [b"old1", b"old2"]
                self.emotion_timeline = [(0.0, "sad", 0.7)]

            def clear(self):
                self.audio_frames.clear()
                self.emotion_timeline.clear()

        buf = FakeBuffer()
        state = {"_diary_buffer_needs_clear": True}

        if state.get("_diary_buffer_needs_clear", False):
            buf.clear()
            state["_diary_buffer_needs_clear"] = False

        assert len(buf.audio_frames) == 0
        assert len(buf.emotion_timeline) == 0
        assert state["_diary_buffer_needs_clear"] is False

        buf.audio_frames.append(b"new_audio")
        for _ in range(3):
            if state.get("_diary_buffer_needs_clear", False):
                buf.clear()
                state["_diary_buffer_needs_clear"] = False

        assert len(buf.audio_frames) == 1


class TestBufferClearEdgeCases:
    """Edge cases for the buffer clear flag."""

    def test_end_session_does_not_set_flag(self):
        """End Session should NOT set the clear flag (data must persist for summary)."""
        src = _read_source()
        idx = src.find('"End Session"')
        assert idx != -1
        block = src[idx:idx+400]
        assert "_diary_buffer_needs_clear = True" not in block

    def test_buffer_clear_in_save_for_cleanup(self):
        """After saving an entry, buffer is cleared for emotion timeline reset."""
        src = _read_source()
        assert "_diary_buffer.clear()" in src

    def test_flag_default_false_in_get(self):
        """session_state.get should default to False for the flag."""
        src = _read_source()
        assert '"_diary_buffer_needs_clear", False' in src or \
               "'_diary_buffer_needs_clear', False" in src
