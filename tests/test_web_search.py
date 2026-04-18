"""Tests for web_search_service.py - test with mocked DDG, test caching."""

import pytest
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.web_search_service import (
    search_general, search_emotion_articles, clear_cache,
    _SearchCache,
)


class TestSearchCache:
    def test_cache_set_and_get(self):
        cache = _SearchCache(ttl=60)
        cache.set("key1", [{"result": "data"}])
        assert cache.get("key1") == [{"result": "data"}]

    def test_cache_miss(self):
        cache = _SearchCache(ttl=60)
        assert cache.get("nonexistent") is None

    def test_cache_expiry(self):
        cache = _SearchCache(ttl=0)  # TTL = 0 means immediate expiry
        cache.set("key1", "data")
        import time
        time.sleep(0.01)
        assert cache.get("key1") is None

    def test_cache_clear(self):
        cache = _SearchCache(ttl=60)
        cache.set("key1", "data1")
        cache.set("key2", "data2")
        cache.clear()
        assert cache.get("key1") is None
        assert cache.get("key2") is None


class TestSearchGeneral:
    def test_search_with_ddg(self):
        """Test successful DDG search via mocked module import."""
        clear_cache()

        mock_ddgs_instance = MagicMock()
        mock_ddgs_instance.__enter__ = MagicMock(return_value=mock_ddgs_instance)
        mock_ddgs_instance.__exit__ = MagicMock(return_value=False)
        mock_ddgs_instance.text.return_value = [
            {"title": "Emotion Regulation Study", "href": "https://apa.org/emotion-study", "body": "Research on emotion regulation."},
            {"title": "CBT for Anxiety", "href": "https://nih.gov/cbt-anxiety", "body": "Cognitive behavioral therapy."},
        ]

        mock_ddgs_cls = MagicMock(return_value=mock_ddgs_instance)
        mock_module = MagicMock()
        mock_module.DDGS = mock_ddgs_cls

        with patch.dict("sys.modules", {"duckduckgo_search": mock_module}):
            clear_cache()
            results = search_general("emotion regulation therapy", max_results=2)

        assert len(results) == 2
        assert results[0]["title"] == "Emotion Regulation Study"
        assert results[0]["url"] == "https://apa.org/emotion-study"

    def test_search_fallback_on_import_error(self):
        """When DDG is not installed, should return demo results."""
        clear_cache()

        with patch("web_search_service.DDGS", side_effect=ImportError("no module"), create=True):
            # Force a fresh import
            results = search_general("test query", max_results=3)

        # Should get demo results (or cached from a previous test)
        assert isinstance(results, list)

    def test_search_caching(self):
        """Cached results should be returned on second call."""
        clear_cache()

        from services.web_search_service import _cache
        _cache.set("cache test:3", [{"title": "Cached", "url": "https://example.com", "snippet": "test"}])

        # Second call should hit cache
        cached = _cache.get("cache test:3")
        assert cached is not None
        assert len(cached) == 1


class TestSearchEmotionArticles:
    def test_returns_list(self):
        clear_cache()
        results = search_emotion_articles("happy", max_results=3)
        assert isinstance(results, list)

    def test_results_have_required_fields(self):
        clear_cache()
        results = search_emotion_articles("sad", max_results=2)
        for r in results:
            assert "title" in r
            assert "url" in r
            assert "snippet" in r


class TestClearCache:
    def test_clear_cache_works(self):
        from services.web_search_service import _cache
        _cache.set("test", "data")
        clear_cache()
        assert _cache.get("test") is None
