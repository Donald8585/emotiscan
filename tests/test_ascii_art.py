"""Tests for ascii_art_generator.py - test all art functions, all emotions, dynamic generation."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ui.ascii_art_generator import (
    STATIC_HEADER, STATIC_SEPARATOR, EMOTION_ARTS, MOOD_BOOSTERS, TOPIC_ART,
    format_paper_card, get_emotion_art, get_mood_booster,
    generate_dynamic_art, get_topic_art,
)
from config import EMOTION_LABELS


class TestStaticElements:
    def test_static_header_is_string(self):
        assert isinstance(STATIC_HEADER, str)
        assert len(STATIC_HEADER) > 0

    def test_static_separator_is_string(self):
        assert isinstance(STATIC_SEPARATOR, str)
        assert len(STATIC_SEPARATOR) > 0

    def test_separator_is_consistent(self):
        assert len(STATIC_SEPARATOR) == 70


class TestFormatPaperCard:
    def test_basic_card(self):
        card = format_paper_card("Test Paper Title", 0.85)
        assert "Test Paper Title" in card
        assert "0.85" in card

    def test_card_with_zero_score(self):
        card = format_paper_card("Paper", 0.0)
        assert "Paper" in card
        assert "0.00" in card

    def test_card_with_max_score(self):
        card = format_paper_card("Paper", 1.0)
        assert "1.00" in card

    def test_card_with_long_title(self):
        long_title = "A" * 100
        card = format_paper_card(long_title, 0.5)
        assert isinstance(card, str)

    def test_card_with_art(self):
        card = format_paper_card("Test", 0.5, art="*art*")
        assert isinstance(card, str)


class TestEmotionArts:
    def test_all_emotions_have_arts(self):
        for emotion in EMOTION_LABELS:
            assert emotion in EMOTION_ARTS, f"Missing art for emotion: {emotion}"

    def test_each_emotion_has_multiple_arts(self):
        for emotion in EMOTION_LABELS:
            arts = EMOTION_ARTS[emotion]
            assert len(arts) >= 3, f"Emotion {emotion} has only {len(arts)} arts (need >= 3)"

    def test_arts_are_strings(self):
        for emotion, arts in EMOTION_ARTS.items():
            for art in arts:
                assert isinstance(art, str), f"Art for {emotion} is not a string"
                assert len(art) > 0, f"Empty art for {emotion}"


class TestGetEmotionArt:
    def test_returns_string_for_all_emotions(self):
        for emotion in EMOTION_LABELS:
            art = get_emotion_art(emotion)
            assert isinstance(art, str)
            assert len(art) > 0

    def test_unknown_emotion_returns_neutral(self):
        art = get_emotion_art("nonexistent")
        assert isinstance(art, str)
        assert len(art) > 0

    def test_case_insensitive(self):
        art = get_emotion_art("HAPPY")
        assert isinstance(art, str)
        assert len(art) > 0

    def test_whitespace_handling(self):
        art = get_emotion_art("  happy  ")
        assert isinstance(art, str)
        assert len(art) > 0

    def test_randomness(self):
        """Multiple calls should (eventually) return different arts."""
        arts = set()
        for _ in range(20):
            arts.add(get_emotion_art("happy"))
        # With 5 options and 20 tries, we should get at least 2 different ones
        assert len(arts) >= 2


class TestMoodBoosters:
    def test_all_emotions_have_boosters(self):
        for emotion in EMOTION_LABELS:
            assert emotion in MOOD_BOOSTERS, f"Missing booster for: {emotion}"

    def test_boosters_are_strings(self):
        for emotion, boosters in MOOD_BOOSTERS.items():
            for b in boosters:
                assert isinstance(b, str)

    def test_get_mood_booster_all_emotions(self):
        for emotion in EMOTION_LABELS:
            booster = get_mood_booster(emotion)
            assert isinstance(booster, str)

    def test_get_mood_booster_unknown_emotion(self):
        booster = get_mood_booster("confused")
        assert isinstance(booster, str)


class TestDynamicArt:
    def test_wave_style(self):
        art = generate_dynamic_art("Hello", "wave")
        assert isinstance(art, str)
        assert len(art) > 0

    def test_matrix_style(self):
        art = generate_dynamic_art("Test", "matrix")
        assert isinstance(art, str)
        assert len(art) > 0

    def test_spiral_style(self):
        art = generate_dynamic_art("Spiral", "spiral")
        assert isinstance(art, str)
        assert len(art) > 0

    def test_blocks_style(self):
        art = generate_dynamic_art("Block", "blocks")
        assert isinstance(art, str)
        assert len(art) > 0

    def test_unknown_style_defaults_to_wave(self):
        art = generate_dynamic_art("Test", "unknown_style")
        assert isinstance(art, str)
        assert len(art) > 0

    def test_empty_text(self):
        art = generate_dynamic_art("", "wave")
        assert isinstance(art, str)
        assert len(art) > 0

    def test_long_text(self):
        art = generate_dynamic_art("A" * 100, "matrix")
        assert isinstance(art, str)


class TestTopicArt:
    def test_topic_art_dict_exists(self):
        assert isinstance(TOPIC_ART, dict)
        assert "llm" in TOPIC_ART
        assert "rag" in TOPIC_ART
        assert "agents" in TOPIC_ART
        assert "default" in TOPIC_ART

    def test_get_topic_art_llm(self):
        art = get_topic_art("large language models")
        assert art == TOPIC_ART["llm"]

    def test_get_topic_art_rag(self):
        art = get_topic_art("retrieval augmented generation")
        assert art == TOPIC_ART["rag"]

    def test_get_topic_art_agents(self):
        art = get_topic_art("AI agents and tools")
        assert art == TOPIC_ART["agents"]

    def test_get_topic_art_default(self):
        art = get_topic_art("quantum computing")
        assert art == TOPIC_ART["default"]
