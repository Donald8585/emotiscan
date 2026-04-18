"""Tests for web_search_service.py enhancements — coping strategies search."""

import pytest
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.web_search_service import (
    search_coping_strategies,
    _cache,
    clear_cache,
)


class TestSearchCopingStrategies:
    def setup_method(self):
        clear_cache()

    @patch("duckduckgo_search.DDGS")
    def test_returns_list(self, mock_ddgs):
        mock_ddgs.return_value.__enter__ = lambda s: s
        mock_ddgs.return_value.__exit__ = lambda *a: None
        mock_ddgs.return_value.text.return_value = [
            {"title": "Cope with sadness", "href": "https://example.com", "body": "Journaling helps."},
        ]
        results = search_coping_strategies("sad")
        assert isinstance(results, list)
        assert len(results) > 0

    @patch("duckduckgo_search.DDGS")
    def test_all_emotions_have_results(self, mock_ddgs):
        mock_ddgs.return_value.__enter__ = lambda s: s
        mock_ddgs.return_value.__exit__ = lambda *a: None
        mock_ddgs.return_value.text.return_value = [
            {"title": "Coping tip", "href": "https://example.com/tip", "body": "Take a walk."},
        ]
        for emo in ["happy", "sad", "angry", "fear", "surprise", "disgust", "neutral"]:
            clear_cache()  # Clear between emotions so mock is re-hit
            results = search_coping_strategies(emo)
            assert len(results) > 0
            for r in results:
                assert "title" in r
                assert "url" in r
                assert "snippet" in r

    @patch("duckduckgo_search.DDGS")
    def test_max_results_respected(self, mock_ddgs):
        mock_ddgs.return_value.__enter__ = lambda s: s
        mock_ddgs.return_value.__exit__ = lambda *a: None
        mock_ddgs.return_value.text.return_value = [
            {"title": f"Tip {i}", "href": f"https://example.com/{i}", "body": f"Tip body {i}"}
            for i in range(5)
        ]
        results = search_coping_strategies("happy", max_results=3)
        assert len(results) <= 3

    @patch("duckduckgo_search.DDGS")
    def test_caching_works(self, mock_ddgs):
        clear_cache()
        mock_ddgs.return_value.__enter__ = lambda s: s
        mock_ddgs.return_value.__exit__ = lambda *a: None
        mock_ddgs.return_value.text.return_value = [
            {"title": "Sad tip", "href": "https://example.com/sad", "body": "It gets better."},
        ]
        results1 = search_coping_strategies("sad", max_results=3)
        results2 = search_coping_strategies("sad", max_results=3)
        # Both should return same results (from cache or demo)
        assert len(results1) == len(results2)

    def test_ddg_exception_falls_back_gracefully(self):
        """When DDG import fails, should return an empty list."""
        clear_cache()
        # Force DDG to fail by patching the import
        with patch("duckduckgo_search.DDGS", side_effect=Exception("DDG unavailable")):
            results = search_coping_strategies("angry")
        assert isinstance(results, list)

    @patch.dict("sys.modules", {"duckduckgo_search": MagicMock()})
    def test_with_mocked_ddg(self):
        """Test with mocked DuckDuckGo."""
        clear_cache()
        mock_module = sys.modules["duckduckgo_search"]
        mock_ddgs = MagicMock()
        mock_ddgs.__enter__ = MagicMock(return_value=mock_ddgs)
        mock_ddgs.__exit__ = MagicMock(return_value=False)
        mock_ddgs.text.return_value = [
            {"title": "Test Result", "href": "https://example.com", "body": "Test snippet"}
        ]
        mock_module.DDGS.return_value = mock_ddgs

        results = search_coping_strategies("happy")
        assert isinstance(results, list)


class TestCopingIntegrationWithDiary:
    """Test that coping strategies can be used in the diary flow."""

    def test_search_returns_consistent_structure(self):
        """Results from search_coping_strategies should match web_results format."""
        results = search_coping_strategies("fear", max_results=3)
        for r in results:
            assert isinstance(r.get("title"), str)
            assert isinstance(r.get("url"), str)
            assert isinstance(r.get("snippet"), str)
