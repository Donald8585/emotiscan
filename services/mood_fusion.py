"""
EmotiScan: Weighted fusion of face, voice, and text emotion signals.
Combines multiple modalities into a unified emotion score.
"""

from config import (
    EMOTION_LABELS,
    FACE_EMOTION_WEIGHT,
    TEXT_SENTIMENT_WEIGHT,
    AUDIO_EMOTION_WEIGHT,
)


class MoodFusion:
    """Combine face, voice, and text emotions into a unified score."""

    WEIGHTS = {
        "face": FACE_EMOTION_WEIGHT,
        "text_sentiment": TEXT_SENTIMENT_WEIGHT,
        "audio": AUDIO_EMOTION_WEIGHT,
    }

    # Map sentiment polarity ranges to emotion labels
    POLARITY_TO_EMOTION = {
        (-1.0, -0.5): "angry",
        (-0.5, -0.2): "sad",
        (-0.2, 0.2): "neutral",
        (0.2, 0.6): "happy",
        (0.6, 1.01): "surprise",  # 1.01 to include 1.0 exactly
    }

    def fuse(
        self,
        face_emotion: str,
        face_confidence: float,
        text_sentiment: dict,
        audio_emotion: dict,
    ) -> dict:
        """
        Fuse face, text, and audio emotion signals.

        Args:
            face_emotion: Emotion label from face detection
            face_confidence: Confidence from face detection (0-1)
            text_sentiment: {"polarity": float, "emotion": str, "confidence": float, ...}
            audio_emotion: {"estimated_emotion": str, "confidence": float, ...}

        Returns:
            {"emotion": str, "confidence": float, "breakdown": dict}
        """
        text_sentiment = text_sentiment or {}
        audio_emotion = audio_emotion or {}

        # Convert each signal to a score vector over all 7 emotions
        face_vec = self.emotion_to_vector(face_emotion, face_confidence)

        text_emo = text_sentiment.get("emotion", "neutral")
        text_conf = text_sentiment.get("confidence", 0.5)
        text_vec = self.emotion_to_vector(text_emo, text_conf)

        audio_emo = audio_emotion.get("estimated_emotion", "neutral")
        audio_conf = audio_emotion.get("confidence", 0.3)
        audio_vec = self.emotion_to_vector(audio_emo, audio_conf)

        # Weighted combination
        fused = {}
        for emo in EMOTION_LABELS:
            fused[emo] = (
                self.WEIGHTS["face"] * face_vec.get(emo, 0.0)
                + self.WEIGHTS["text_sentiment"] * text_vec.get(emo, 0.0)
                + self.WEIGHTS["audio"] * audio_vec.get(emo, 0.0)
            )

        # Normalize
        total = sum(fused.values())
        if total > 0:
            fused = {k: round(v / total, 4) for k, v in fused.items()}
        else:
            fused = {k: round(1.0 / len(EMOTION_LABELS), 4) for k in EMOTION_LABELS}

        top_emotion = max(fused, key=fused.get)
        top_confidence = fused[top_emotion]

        return {
            "emotion": top_emotion,
            "confidence": round(top_confidence, 4),
            "breakdown": {
                "face": {"emotion": face_emotion, "confidence": round(face_confidence, 3)},
                "text": {"emotion": text_emo, "confidence": round(text_conf, 3)},
                "audio": {"emotion": audio_emo, "confidence": round(audio_conf, 3)},
            },
            "all_emotions": fused,
        }

    @staticmethod
    def emotion_to_vector(emotion: str, confidence: float) -> dict:
        """
        Convert an emotion label + confidence to a score vector over all 7 emotions.

        The primary emotion gets `confidence` score, the rest share `1 - confidence`.
        """
        emotion = emotion.lower().strip() if emotion else "neutral"
        confidence = max(0.0, min(1.0, confidence))

        if emotion not in EMOTION_LABELS:
            emotion = "neutral"

        vector = {}
        remainder = 1.0 - confidence
        for emo in EMOTION_LABELS:
            if emo == emotion:
                vector[emo] = confidence
            else:
                vector[emo] = remainder / max(len(EMOTION_LABELS) - 1, 1)

        return vector

    @classmethod
    def polarity_to_emotion(cls, polarity: float) -> str:
        """Map a sentiment polarity value (-1 to 1) to an emotion label."""
        polarity = max(-1.0, min(1.0, polarity))
        for (low, high), emotion in cls.POLARITY_TO_EMOTION.items():
            if low <= polarity < high:
                return emotion
        # Edge case: exactly 1.0
        return "surprise"

    @classmethod
    def weights_sum(cls) -> float:
        """Return the sum of fusion weights (should be 1.0)."""
        return sum(cls.WEIGHTS.values())
