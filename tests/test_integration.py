"""Integration tests - test module interactions (emotion → art, emotion → research)."""

import pytest
import sys
import os
import numpy as np
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import EMOTION_LABELS, NEGATIVE_EMOTIONS, POSITIVE_EMOTIONS
from ui.ascii_art_generator import get_emotion_art, get_mood_booster, get_topic_art
from services.llm_service import LLMService
from services.web_search_service import search_emotion_articles
from mcp.mcp_client import rank_results


class TestEmotionToArtPipeline:
    """Test that every detected emotion produces valid ASCII art."""

    def test_all_emotions_produce_art(self):
        for emotion in EMOTION_LABELS:
            art = get_emotion_art(emotion)
            assert isinstance(art, str)
            assert len(art) > 10  # Should be non-trivial art

    def test_all_emotions_produce_boosters(self):
        for emotion in EMOTION_LABELS:
            booster = get_mood_booster(emotion)
            assert isinstance(booster, str)
            assert len(booster) > 5

    def test_negative_emotions_get_special_boosters(self):
        for emotion in NEGATIVE_EMOTIONS:
            booster = get_mood_booster(emotion)
            assert isinstance(booster, str)
            # Negative emotion boosters should be uplifting
            assert len(booster) > 10


class TestEmotionDetectorToArt:
    """Test mock detection results flowing to art generation."""

    def test_detection_result_to_art(self):
        # Simulate what happens when emotion detector returns a result
        mock_results = [
            {"bbox": [100, 100, 200, 200], "emotion": "happy", "confidence": 0.9,
             "all_emotions": {"happy": 0.9, "neutral": 0.1}},
        ]

        for result in mock_results:
            emotion = result["emotion"]
            art = get_emotion_art(emotion)
            assert isinstance(art, str)

            booster = get_mood_booster(emotion)
            assert isinstance(booster, str)

    def test_multiple_faces_all_get_art(self):
        emotions = ["happy", "sad", "angry"]
        for emo in emotions:
            art = get_emotion_art(emo)
            assert isinstance(art, str)
            assert len(art) > 0

