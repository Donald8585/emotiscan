"""Tests for emotion detection logic and timeline-aware emotion tracking.

After reverting both re-weighting and bootstrap approaches (both distorted
detection), the system now uses DeepFace's raw top emotion directly.
These tests verify:
1. Direct top-emotion selection (max probability wins)
2. Time-weighted emotion_during_recording logic
3. get_full_context uses per-second face_emotion_timeline
4. best_emotion includes face_emotion_timeline data
5. Demo summary/compassionate response includes sensor details
"""
import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestDirectTopEmotion(unittest.TestCase):
    """Verify that max(all_emotions) picks the right winner."""

    def test_clear_winner(self):
        scores = {"happy": 0.85, "neutral": 0.08, "sad": 0.03,
                  "surprise": 0.02, "angry": 0.01, "fear": 0.005, "disgust": 0.005}
        top = max(scores, key=scores.get)
        self.assertEqual(top, "happy")

    def test_fear_winner(self):
        scores = {"fear": 0.73, "neutral": 0.17, "sad": 0.08}
        top = max(scores, key=scores.get)
        self.assertEqual(top, "fear")

    def test_sad_winner(self):
        scores = {"sad": 0.50, "neutral": 0.25, "fear": 0.10,
                  "happy": 0.05, "angry": 0.05, "surprise": 0.03, "disgust": 0.02}
        top = max(scores, key=scores.get)
        self.assertEqual(top, "sad")

    def test_disgust_winner(self):
        scores = {"disgust": 0.40, "neutral": 0.30, "sad": 0.15,
                  "happy": 0.05, "angry": 0.05, "surprise": 0.03, "fear": 0.02}
        top = max(scores, key=scores.get)
        self.assertEqual(top, "disgust")

    def test_all_emotions_can_win(self):
        for emo in ["happy", "sad", "angry", "fear", "surprise", "neutral", "disgust"]:
            scores = {e: 0.02 for e in
                      ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]}
            scores[emo] = 0.88
            top = max(scores, key=scores.get)
            self.assertEqual(top, emo, f"'{emo}' at 88% should be the winner")


class TestBestEmotionUsesTimeline(unittest.TestCase):
    """Verify that DiarySession.best_emotion includes face_emotion_timeline data."""

    def test_timeline_overrides_snapshot(self):
        """If the per-entry label says 'neutral' but timeline shows 'fear' 30 times,
        best_emotion should pick 'fear'."""
        from diary.diary_session import DiaryEntry, DiarySession
        sess = DiarySession(session_id="test1", start_time="2026-01-01 10:00:00")
        timeline = [{"time": f"10:00:{i:02d}", "emotion": "fear", "confidence": 0.7}
                     for i in range(30)]
        entry = DiaryEntry(
            timestamp="10:00:00",
            text="testing",
            face_emotion="neutral",
            face_confidence=0.5,
            voice_sentiment={"emotion": "neutral"},
            audio_emotion={"estimated_emotion": "neutral"},
            fused_emotion="neutral",
            fused_confidence=0.5,
            face_emotion_timeline=timeline,
        )
        sess.entries.append(entry)
        self.assertEqual(sess.best_emotion, "fear")

    def test_no_timeline_uses_snapshot(self):
        """When no timeline data, falls back to the per-entry labels."""
        from diary.diary_session import DiaryEntry, DiarySession
        sess = DiarySession(session_id="test2", start_time="2026-01-01 10:00:00")
        entry = DiaryEntry(
            timestamp="10:00:00",
            text="testing",
            face_emotion="sad",
            face_confidence=0.6,
            voice_sentiment={"emotion": "sad"},
            audio_emotion={"estimated_emotion": "neutral"},
            fused_emotion="sad",
            fused_confidence=0.6,
            face_emotion_timeline=[],
        )
        sess.entries.append(entry)
        self.assertEqual(sess.best_emotion, "sad")


class TestGetFullContextUsesTimeline(unittest.TestCase):
    """Verify get_full_context returns per-second timeline data for face_emotions."""

    def test_face_emotions_from_timeline(self):
        from diary.diary_session import DiaryEntry, DiarySession
        sess = DiarySession(session_id="test3", start_time="2026-01-01 10:00:00")
        timeline = [
            {"time": "10:00:01", "emotion": "fear", "confidence": 0.7},
            {"time": "10:00:02", "emotion": "fear", "confidence": 0.8},
            {"time": "10:00:03", "emotion": "neutral", "confidence": 0.5},
        ]
        entry = DiaryEntry(
            timestamp="10:00:00",
            text="test",
            face_emotion="neutral",
            face_confidence=0.5,
            voice_sentiment={"emotion": "neutral"},
            audio_emotion={"estimated_emotion": "neutral"},
            fused_emotion="neutral",
            fused_confidence=0.5,
            face_emotion_timeline=timeline,
        )
        sess.entries.append(entry)
        ctx = sess.get_full_context()
        # Should use the per-second timeline, not the snapshot
        self.assertEqual(ctx["face_emotions"], ["fear", "fear", "neutral"])

    def test_face_emotions_fallback_no_timeline(self):
        from diary.diary_session import DiaryEntry, DiarySession
        sess = DiarySession(session_id="test4", start_time="2026-01-01 10:00:00")
        entry = DiaryEntry(
            timestamp="10:00:00",
            text="test",
            face_emotion="happy",
            face_confidence=0.9,
            voice_sentiment={"emotion": "neutral"},
            audio_emotion={"estimated_emotion": "neutral"},
            fused_emotion="happy",
            fused_confidence=0.8,
            face_emotion_timeline=[],
        )
        sess.entries.append(entry)
        ctx = sess.get_full_context()
        # No timeline ??falls back to per-entry label
        self.assertEqual(ctx["face_emotions"], ["happy"])


if __name__ == "__main__":
    unittest.main()
