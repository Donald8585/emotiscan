"""
EmotiScan: Speech-to-text and text sentiment analysis.
Uses faster-whisper for STT, TextBlob for sentiment, numpy for audio heuristics.
"""

import io
import struct
import logging
import numpy as np

from config import WHISPER_MODEL, STT_SAMPLE_RATE, STT_MAX_DURATION, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE

logger = logging.getLogger(__name__)


class SpeechService:
    """STT + text sentiment + audio emotion heuristics."""

    def __init__(self, whisper_model: str = None, sample_rate: int = None):
        self._whisper_model_name = whisper_model or WHISPER_MODEL
        self._sample_rate = sample_rate or STT_SAMPLE_RATE
        self._whisper = None
        self._whisper_loaded = False
        self._whisper_error = None

    def _load_whisper(self):
        """Lazy-load faster-whisper model."""
        if self._whisper_loaded:
            return
        self._whisper_loaded = True
        try:
            from faster_whisper import WhisperModel
            self._whisper = WhisperModel(
                self._whisper_model_name,
                device=WHISPER_DEVICE,
                compute_type=WHISPER_COMPUTE_TYPE,
            )
            logger.info("SpeechService: faster-whisper loaded (model=%s)", self._whisper_model_name)
        except Exception as e:
            self._whisper_error = str(e)
            logger.warning("SpeechService: faster-whisper unavailable: %s", e)

    @property
    def is_whisper_available(self) -> bool:
        self._load_whisper()
        return self._whisper is not None

    def transcribe_audio(self, audio_bytes: bytes, format: str = "wav") -> dict:
        """
        Transcribe audio bytes to text.

        Returns:
            {"text": str, "language": str, "duration": float}
        """
        if not audio_bytes:
            return {"text": "", "language": "unknown", "duration": 0.0}

        self._load_whisper()

        if self._whisper is not None:
            try:
                return self._transcribe_whisper(audio_bytes)
            except Exception as e:
                logger.warning("Whisper transcription failed: %s", e)

        return {"text": "", "language": "unknown", "duration": 0.0}

    def _transcribe_whisper(self, audio_bytes: bytes) -> dict:
        """Transcribe using faster-whisper."""
        audio_file = io.BytesIO(audio_bytes)
        segments, info = self._whisper.transcribe(
            audio_file,
            beam_size=5,
            language=None,  # auto-detect
        )
        text_parts = []
        for segment in segments:
            text_parts.append(segment.text)

        full_text = " ".join(text_parts).strip()
        return {
            "text": full_text,
            "language": info.language if info else "unknown",
            "duration": info.duration if info else 0.0,
        }

    def analyze_sentiment(self, text: str) -> dict:
        """
        Analyze text sentiment.

        Returns:
            {"polarity": float[-1,1], "subjectivity": float[0,1],
             "emotion": str, "confidence": float}
        """
        if not text or not text.strip():
            return {
                "polarity": 0.0,
                "subjectivity": 0.0,
                "emotion": "neutral",
                "confidence": 0.5,
            }

        try:
            from textblob import TextBlob
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
        except ImportError:
            logger.warning("TextBlob not available, using heuristic sentiment")
            polarity, subjectivity = self._heuristic_sentiment(text)
        except Exception as e:
            logger.warning("Sentiment analysis error: %s", e)
            polarity, subjectivity = self._heuristic_sentiment(text)

        emotion = self._polarity_to_emotion(polarity)
        confidence = min(abs(polarity) * 1.5 + 0.3, 1.0)

        return {
            "polarity": round(polarity, 3),
            "subjectivity": round(subjectivity, 3),
            "emotion": emotion,
            "confidence": round(confidence, 3),
        }

    @staticmethod
    def _heuristic_sentiment(text: str) -> tuple:
        """Basic keyword-based sentiment when TextBlob isn't available."""
        text_lower = text.lower()
        positive_words = {"happy", "great", "good", "love", "wonderful", "amazing",
                          "excellent", "joy", "grateful", "motivated", "excited", "glad"}
        negative_words = {"sad", "bad", "angry", "hate", "terrible", "awful",
                          "depressed", "anxious", "fear", "worried", "overwhelmed", "stressed"}

        words = set(text_lower.split())
        pos_count = len(words & positive_words)
        neg_count = len(words & negative_words)
        total = pos_count + neg_count

        if total == 0:
            return 0.0, 0.3

        polarity = (pos_count - neg_count) / max(total, 1)
        subjectivity = min(total / max(len(words), 1) * 2, 1.0)
        return polarity, subjectivity

    @staticmethod
    def _polarity_to_emotion(polarity: float) -> str:
        """Map sentiment polarity to an emotion label."""
        if polarity < -0.5:
            return "angry"
        elif polarity < -0.2:
            return "sad"
        elif polarity < 0.2:
            return "neutral"
        elif polarity < 0.6:
            return "happy"
        else:
            return "surprise"

    def analyze_audio_emotion(self, audio_bytes: bytes) -> dict:
        """
        Extract audio-level emotion cues from pitch, energy, speech rate.

        Returns:
            {"energy": float, "pitch_var": float,
             "estimated_emotion": str, "confidence": float}
        """
        if not audio_bytes or len(audio_bytes) < 100:
            return {
                "energy": 0.0,
                "pitch_var": 0.0,
                "estimated_emotion": "neutral",
                "confidence": 0.3,
            }

        try:
            samples = self._bytes_to_samples(audio_bytes)
            if samples is None or len(samples) < 100:
                return {"energy": 0.0, "pitch_var": 0.0, "estimated_emotion": "neutral", "confidence": 0.3}

            energy = float(np.sqrt(np.mean(samples.astype(float) ** 2)))
            energy_norm = min(energy / 8000.0, 1.0)

            # Estimate pitch variation using zero-crossing rate
            zero_crossings = np.sum(np.abs(np.diff(np.sign(samples))) > 0)
            zcr = zero_crossings / max(len(samples), 1)
            pitch_var = min(zcr * 100, 1.0)

            # Heuristic emotion from energy + pitch
            if energy_norm > 0.7 and pitch_var > 0.4:
                emotion = "angry"
                confidence = 0.6
            elif energy_norm > 0.5 and pitch_var > 0.5:
                emotion = "surprise"
                confidence = 0.5
            elif energy_norm > 0.4:
                emotion = "happy"
                confidence = 0.5
            elif energy_norm < 0.15:
                emotion = "sad"
                confidence = 0.5
            else:
                emotion = "neutral"
                confidence = 0.4

            return {
                "energy": round(energy_norm, 3),
                "pitch_var": round(pitch_var, 3),
                "estimated_emotion": emotion,
                "confidence": round(confidence, 3),
            }
        except Exception as e:
            logger.warning("Audio emotion analysis failed: %s", e)
            return {"energy": 0.0, "pitch_var": 0.0, "estimated_emotion": "neutral", "confidence": 0.3}

    @staticmethod
    def _bytes_to_samples(audio_bytes: bytes) -> np.ndarray:
        """Convert raw audio bytes to numpy int16 samples.
        Tries to parse as WAV first, then falls back to raw PCM."""
        try:
            # Try WAV header parse
            if audio_bytes[:4] == b'RIFF':
                # Skip 44-byte WAV header
                raw = audio_bytes[44:]
            else:
                raw = audio_bytes

            # Parse as 16-bit signed little-endian PCM
            n_samples = len(raw) // 2
            if n_samples == 0:
                return None
            samples = np.array(
                struct.unpack(f"<{n_samples}h", raw[:n_samples * 2]),
                dtype=np.int16,
            )
            return samples
        except Exception:
            # Last resort: treat bytes as uint8, center around 0
            arr = np.frombuffer(audio_bytes, dtype=np.uint8).astype(np.int16) - 128
            return arr * 256  # scale to int16 range
