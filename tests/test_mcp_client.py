"""Tests for mcp_client.py - test with mocked MCP servers, test fallbacks."""

import pytest
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mcp.mcp_client import (
    search_papers, rank_results, _rank_local, _check_port,
)


class TestCheckPort:
    @patch("mcp_client.socket.create_connection")
    def test_port_open(self, mock_conn):
        mock_socket = MagicMock()
        mock_conn.return_value = mock_socket
        assert _check_port("localhost", 8002) is True
        mock_socket.close.assert_called_once()

    @patch("mcp_client.socket.create_connection")
    def test_port_closed(self, mock_conn):
        mock_conn.side_effect = ConnectionRefusedError()
        assert _check_port("localhost", 9999) is False


class TestSearchPapers:
    @patch("mcp_client._search_arxiv_direct")
    def test_search_papers_direct(self, mock_search):
        mock_search.return_value = [
            {"title": "Test Paper", "abstract": "Test abstract", "source": "arxiv"},
        ]
        results = search_papers("test query")
        assert len(results) == 1
        assert results[0]["title"] == "Test Paper"

    @patch("mcp_client._search_arxiv_direct")
    def test_search_papers_fallback_on_error(self, mock_search):
        mock_search.side_effect = Exception("API error")
        results = search_papers("test query")
        # Should return empty list when ArXiv fails
        assert isinstance(results, list)
        assert len(results) == 0


class TestRankResults:
    def test_rank_empty_list(self):
        result = rank_results("test interests", [])
        assert result == []

    def test_rank_local_fallback(self):
        articles = [
            {"title": "Machine Learning Paper", "abstract": "Deep learning neural networks"},
            {"title": "Cooking Recipe", "abstract": "How to make pasta with tomato sauce"},
            {"title": "AI Research", "abstract": "Artificial intelligence machine learning model"},
        ]
        ranked = rank_results("machine learning AI", articles)
        assert len(ranked) == 3
        # All should have relevance_score
        for a in ranked:
            assert "relevance_score" in a
        # ML-related articles should rank higher than cooking
        scores = {a["title"]: a["relevance_score"] for a in ranked}
        assert scores["Machine Learning Paper"] > scores["Cooking Recipe"]

    @patch("mcp_client._check_port", return_value=True)
    @patch("mcp_client._rank_via_mcp")
    def test_rank_via_mcp(self, mock_mcp_rank, mock_port):
        articles = [{"title": "Test", "abstract": "Test"}]
        mock_mcp_rank.return_value = [{"title": "Test", "abstract": "Test", "relevance_score": 0.9}]

        result = rank_results("test", articles)
        assert result[0]["relevance_score"] == 0.9

    @patch("mcp_client._check_port", return_value=True)
    @patch("mcp_client._rank_via_mcp")
    def test_rank_falls_back_when_mcp_fails(self, mock_mcp_rank, mock_port):
        mock_mcp_rank.side_effect = Exception("MCP error")
        articles = [
            {"title": "Paper A", "abstract": "machine learning"},
            {"title": "Paper B", "abstract": "cooking recipes"},
        ]
        result = rank_results("machine learning", articles)
        assert len(result) == 2
        for a in result:
            assert "relevance_score" in a


class TestRankLocal:
    def test_basic_ranking(self):
        articles = [
            {"title": "Deep Learning", "abstract": "Neural networks and deep learning"},
            {"title": "Gardening Tips", "abstract": "How to grow tomatoes in your garden"},
        ]
        ranked = _rank_local("deep learning neural networks", articles)
        assert ranked[0]["title"] == "Deep Learning"

    def test_empty_articles(self):
        assert _rank_local("test", []) == []

    def test_scores_between_0_and_1(self):
        articles = [
            {"title": "Test Paper", "abstract": "Some content here"},
        ]
        ranked = _rank_local("test content", articles)
        for a in ranked:
            assert 0 <= a["relevance_score"] <= 1.0
