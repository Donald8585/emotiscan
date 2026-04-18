"""Tests for config.py - verify all config values exist and are valid."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (
    OLLAMA_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT, OLLAMA_NUM_CTX, OLLAMA_TEMPERATURE,
    MCP_RANKER_URL, MCP_RANKER_HOST, MCP_RANKER_PORT,
    YOLO_MODEL, YOLO_FACE_CONF, YOLO_FACE_IOU, YOLO_DEVICE, EMOTION_LABELS,
    NEGATIVE_EMOTIONS, POSITIVE_EMOTIONS, NEUTRAL_EMOTIONS,
    GPU_LAYERS, WEB_SEARCH_MAX_RESULTS, WEB_SEARCH_CACHE_TTL,
    WEBRTC_STUN_SERVER,
    PAGE_TITLE, PAGE_ICON, LAYOUT,
    EMOTION_RESEARCH_TOPICS, QUERY_EXPANSIONS,
)


class TestOllamaConfig:
    def test_ollama_url_is_string(self):
        assert isinstance(OLLAMA_URL, str)
        assert OLLAMA_URL.startswith("http")

    def test_ollama_model_is_string(self):
        assert isinstance(OLLAMA_MODEL, str)
        assert len(OLLAMA_MODEL) > 0

    def test_ollama_timeout_positive(self):
        assert isinstance(OLLAMA_TIMEOUT, (int, float))
        assert OLLAMA_TIMEOUT > 0

    def test_ollama_num_ctx_positive(self):
        assert isinstance(OLLAMA_NUM_CTX, int)
        assert OLLAMA_NUM_CTX > 0

    def test_ollama_temperature_range(self):
        assert 0 <= OLLAMA_TEMPERATURE <= 2.0


class TestMCPConfig:
    def test_mcp_ranker_url(self):
        assert isinstance(MCP_RANKER_URL, str)
        assert "localhost" in MCP_RANKER_URL or "127.0.0.1" in MCP_RANKER_URL

    def test_mcp_ranker_port(self):
        assert isinstance(MCP_RANKER_PORT, int)
        assert 1 <= MCP_RANKER_PORT <= 65535


class TestEmotionConfig:
    def test_emotion_labels_exist(self):
        assert isinstance(EMOTION_LABELS, list)
        assert len(EMOTION_LABELS) == 7

    def test_required_emotions_present(self):
        required = {"happy", "sad", "angry", "surprise", "fear", "disgust", "neutral"}
        assert set(EMOTION_LABELS) == required

    def test_emotion_sets_are_subsets(self):
        all_emo = set(EMOTION_LABELS)
        assert NEGATIVE_EMOTIONS.issubset(all_emo)
        assert POSITIVE_EMOTIONS.issubset(all_emo)
        assert NEUTRAL_EMOTIONS.issubset(all_emo)

    def test_emotion_research_topics_cover_all(self):
        for label in EMOTION_LABELS:
            assert label in EMOTION_RESEARCH_TOPICS


class TestYOLOConfig:
    def test_yolo_model_string(self):
        assert isinstance(YOLO_MODEL, str)
        assert YOLO_MODEL.endswith(".pt")

    def test_yolo_face_conf_range(self):
        assert 0 < YOLO_FACE_CONF <= 1.0

    def test_yolo_face_iou_range(self):
        assert 0 < YOLO_FACE_IOU <= 1.0

    def test_yolo_device_valid(self):
        assert YOLO_DEVICE in ("cuda", "cpu")


class TestGeneralConfig:
    def test_gpu_layers_range(self):
        assert isinstance(GPU_LAYERS, int)
        assert 0 <= GPU_LAYERS <= 99

    def test_web_search_defaults(self):
        assert isinstance(WEB_SEARCH_MAX_RESULTS, int)
        assert WEB_SEARCH_MAX_RESULTS > 0
        assert isinstance(WEB_SEARCH_CACHE_TTL, int)
        assert WEB_SEARCH_CACHE_TTL > 0

    def test_page_config(self):
        assert isinstance(PAGE_TITLE, str)
        assert len(PAGE_TITLE) > 0
        assert isinstance(LAYOUT, str)

    def test_webrtc_stun(self):
        assert isinstance(WEBRTC_STUN_SERVER, str)
        assert "stun" in WEBRTC_STUN_SERVER

    def test_query_expansions(self):
        assert isinstance(QUERY_EXPANSIONS, dict)
        assert len(QUERY_EXPANSIONS) > 0
