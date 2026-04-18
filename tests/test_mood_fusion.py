"""Tests for mood_fusion.py — emotion vector math, weighted fusion, polarity mapping."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.mood_fusion import MoodFusion
from config import EMOTION_LABELS


class TestWeights:
    def test_weights_sum_to_one(self):
        fusion = MoodFusion()
        assert abs(fusion.weights_sum() - 1.0) < 0.001

    def test_all_weight_keys_present(self):
        assert "face" in MoodFusion.WEIGHTS
        assert "text_sentiment" in MoodFusion.WEIGHTS
        assert "audio" in MoodFusion.WEIGHTS


class TestEmotionToVector:
    def test_known_emotion_high_confidence(self):
        vec = MoodFusion.emotion_to_vector("happy", 0.9)
        assert vec["happy"] == 0.9
        # Remainder spread equally among 6 other emotions
        others = [v for k, v in vec.items() if k != "happy"]
        assert all(abs(v - 0.1 / 6) < 0.001 for v in others)

    def test_known_emotion_zero_confidence(self):
        vec = MoodFusion.emotion_to_vector("sad", 0.0)
        # Primary emotion gets 0.0, remainder (1.0) spread among 6 others
        assert vec["sad"] == 0.0
        for emo in EMOTION_LABELS:
            if emo != "sad":
                assert abs(vec[emo] - 1.0 / 6) < 0.01

    def test_known_emotion_full_confidence(self):
        vec = MoodFusion.emotion_to_vector("angry", 1.0)
        assert vec["angry"] == 1.0
        for emo in EMOTION_LABELS:
            if emo != "angry":
                assert vec[emo] == 0.0

    def test_unknown_emotion_defaults_to_neutral(self):
        vec = MoodFusion.emotion_to_vector("confused", 0.8)
        assert vec["neutral"] == 0.8

    def test_none_emotion_defaults_to_neutral(self):
        vec = MoodFusion.emotion_to_vector(None, 0.5)
        assert vec["neutral"] == 0.5

    def test_empty_string_defaults_to_neutral(self):
        vec = MoodFusion.emotion_to_vector("", 0.5)
        assert vec["neutral"] == 0.5

    def test_confidence_clamped_high(self):
        vec = MoodFusion.emotion_to_vector("happy", 1.5)
        assert vec["happy"] == 1.0

    def test_confidence_clamped_low(self):
        vec = MoodFusion.emotion_to_vector("happy", -0.5)
        assert vec["happy"] == 0.0

    def test_all_emotions_present(self):
        vec = MoodFusion.emotion_to_vector("fear", 0.7)
        for emo in EMOTION_LABELS:
            assert emo in vec

    def test_vector_sums_to_one(self):
        vec = MoodFusion.emotion_to_vector("surprise", 0.6)
        total = sum(vec.values())
        assert abs(total - 1.0) < 0.01


class TestFuse:
    def test_basic_fusion(self):
        fusion = MoodFusion()
        result = fusion.fuse(
            face_emotion="happy",
            face_confidence=0.9,
            text_sentiment={"polarity": 0.5, "emotion": "happy", "confidence": 0.7},
            audio_emotion={"estimated_emotion": "happy", "confidence": 0.5},
        )
        assert result["emotion"] == "happy"
        assert 0 <= result["confidence"] <= 1.0
        assert "breakdown" in result
        assert "all_emotions" in result

    def test_mixed_emotions(self):
        fusion = MoodFusion()
        result = fusion.fuse(
            face_emotion="sad",
            face_confidence=0.8,
            text_sentiment={"emotion": "happy", "confidence": 0.7},
            audio_emotion={"estimated_emotion": "angry", "confidence": 0.6},
        )
        # Face has highest weight (0.5) with high confidence, so sad should win
        assert result["emotion"] == "sad"

    def test_none_sentiments_handled(self):
        fusion = MoodFusion()
        result = fusion.fuse(
            face_emotion="neutral",
            face_confidence=0.5,
            text_sentiment=None,
            audio_emotion=None,
        )
        assert "emotion" in result
        assert result["emotion"] in EMOTION_LABELS

    def test_empty_dict_sentiments(self):
        fusion = MoodFusion()
        result = fusion.fuse(
            face_emotion="neutral",
            face_confidence=0.5,
            text_sentiment={},
            audio_emotion={},
        )
        assert result["emotion"] in EMOTION_LABELS

    def test_breakdown_structure(self):
        fusion = MoodFusion()
        result = fusion.fuse(
            face_emotion="happy",
            face_confidence=0.8,
            text_sentiment={"emotion": "happy", "confidence": 0.6},
            audio_emotion={"estimated_emotion": "neutral", "confidence": 0.4},
        )
        bd = result["breakdown"]
        assert bd["face"]["emotion"] == "happy"
        assert bd["face"]["confidence"] == 0.8
        assert bd["text"]["emotion"] == "happy"
        assert bd["audio"]["emotion"] == "neutral"

    def test_all_emotions_normalized(self):
        fusion = MoodFusion()
        result = fusion.fuse(
            face_emotion="fear",
            face_confidence=0.6,
            text_sentiment={"emotion": "sad", "confidence": 0.5},
            audio_emotion={"estimated_emotion": "neutral", "confidence": 0.4},
        )
        total = sum(result["all_emotions"].values())
        assert abs(total - 1.0) < 0.01

    def test_all_emotions_has_all_labels(self):
        fusion = MoodFusion()
        result = fusion.fuse(
            face_emotion="neutral",
            face_confidence=0.5,
            text_sentiment={"emotion": "neutral", "confidence": 0.5},
            audio_emotion={"estimated_emotion": "neutral", "confidence": 0.5},
        )
        for emo in EMOTION_LABELS:
            assert emo in result["all_emotions"]

    def test_face_dominates_with_high_confidence(self):
        """Face has weight 0.5, so with very high confidence it should dominate."""
        fusion = MoodFusion()
        result = fusion.fuse(
            face_emotion="disgust",
            face_confidence=0.99,
            text_sentiment={"emotion": "happy", "confidence": 0.3},
            audio_emotion={"estimated_emotion": "happy", "confidence": 0.3},
        )
        assert result["emotion"] == "disgust"


class TestPolarityToEmotion:
    def test_very_negative(self):
        assert MoodFusion.polarity_to_emotion(-0.8) == "angry"

    def test_moderately_negative(self):
        assert MoodFusion.polarity_to_emotion(-0.3) == "sad"

    def test_neutral(self):
        assert MoodFusion.polarity_to_emotion(0.0) == "neutral"

    def test_positive(self):
        assert MoodFusion.polarity_to_emotion(0.4) == "happy"

    def test_very_positive(self):
        assert MoodFusion.polarity_to_emotion(0.8) == "surprise"

    def test_exact_one(self):
        # Edge case: polarity = 1.0 exactly
        assert MoodFusion.polarity_to_emotion(1.0) == "surprise"

    def test_exact_negative_one(self):
        assert MoodFusion.polarity_to_emotion(-1.0) == "angry"

    def test_out_of_range_high(self):
        # Should be clamped to 1.0
        assert MoodFusion.polarity_to_emotion(5.0) == "surprise"

    def test_out_of_range_low(self):
        assert MoodFusion.polarity_to_emotion(-5.0) == "angry"

    def test_boundary_neg_05(self):
        result = MoodFusion.polarity_to_emotion(-0.5)
        assert result in ("angry", "sad")  # boundary

    def test_boundary_neg_02(self):
        result = MoodFusion.polarity_to_emotion(-0.2)
        assert result in ("sad", "neutral")  # boundary

    def test_boundary_02(self):
        result = MoodFusion.polarity_to_emotion(0.2)
        assert result in ("neutral", "happy")  # boundary
