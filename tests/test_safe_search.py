"""
Tests for safe search filtering and content safety in web_search_service.
"""

import pytest
from unittest.mock import patch, MagicMock

# ── import module under test ────────────────────────────────────────────
import services.web_search_service as ws


# ════════════════════════════════════════════════════════════════════════
# _is_blocked
# ════════════════════════════════════════════════════════════════════════


class TestIsBlocked:
    """Tests for the _is_blocked content filter."""

    def test_blocks_adult_domain(self):
        r = {"title": "Something", "url": "https://xhamster.com/search/foo", "snippet": "..."}
        assert ws._is_blocked(r) is True

    def test_blocks_pornhub(self):
        r = {"title": "Foo", "url": "https://pornhub.com/video/123", "snippet": "..."}
        assert ws._is_blocked(r) is True

    def test_blocks_xnxx(self):
        r = {"title": "Foo", "url": "https://www.xnxx.com/search/bar", "snippet": "..."}
        assert ws._is_blocked(r) is True

    def test_blocks_baidu_zhidao(self):
        r = {"title": "surprise和...", "url": "https://zhidao.baidu.com/question/123", "snippet": "..."}
        assert ws._is_blocked(r) is True

    def test_blocks_adult_keyword_in_title(self):
        r = {"title": "Bokep Porn Videos", "url": "https://example.com", "snippet": "..."}
        assert ws._is_blocked(r) is True

    def test_blocks_adult_keyword_in_snippet(self):
        r = {"title": "Something", "url": "https://example.com", "snippet": "Watch xxx content here."}
        assert ws._is_blocked(r) is True

    def test_allows_psychology_today(self):
        r = {"title": "Understanding Emotions", "url": "https://www.psychologytoday.com/article/123", "snippet": "Research on emotion regulation."}
        assert ws._is_blocked(r) is False

    def test_allows_apa(self):
        r = {"title": "APA Study", "url": "https://www.apa.org/study", "snippet": "Behavioral research findings."}
        assert ws._is_blocked(r) is False

    def test_allows_nih(self):
        r = {"title": "NIH Study", "url": "https://www.nih.gov/article", "snippet": "Evidence-based mental health."}
        assert ws._is_blocked(r) is False

    def test_handles_none_values(self):
        r = {"title": None, "url": None, "snippet": None}
        assert ws._is_blocked(r) is False

    def test_handles_empty_result(self):
        r = {}
        assert ws._is_blocked(r) is False

    def test_blocks_gambling_domain(self):
        r = {"title": "Bet", "url": "https://bet365.com/odds", "snippet": "..."}
        assert ws._is_blocked(r) is True


# ════════════════════════════════════════════════════════════════════════
# _is_relevant
# ════════════════════════════════════════════════════════════════════════


class TestIsRelevant:
    """Tests for the _is_relevant topic filter."""

    def test_relevant_psychology_article(self):
        r = {"title": "Understanding Emotions", "url": "https://www.psychologytoday.com/", "snippet": "Research on emotion regulation."}
        assert ws._is_relevant(r) is True

    def test_relevant_nih_mental_health(self):
        r = {"title": "Mental Health Tips", "url": "https://www.nih.gov/tips", "snippet": "Evidence-based coping strategies."}
        assert ws._is_relevant(r) is True

    def test_relevant_by_keyword_therapy(self):
        r = {"title": "Therapy Options", "url": "https://example.com", "snippet": "Cognitive behavioral therapy helps."}
        assert ws._is_relevant(r) is True

    def test_relevant_by_keyword_mindfulness(self):
        r = {"title": "Mindfulness Guide", "url": "https://example.com", "snippet": "Mindfulness meditation practice."}
        assert ws._is_relevant(r) is True

    def test_relevant_by_keyword_anxiety(self):
        r = {"title": "Dealing with Anxiety", "url": "https://example.com", "snippet": "Anxiety relief techniques."}
        assert ws._is_relevant(r) is True

    def test_relevant_by_keyword_mood(self):
        r = {"title": "Understanding Mood Swings", "url": "https://example.com", "snippet": "Managing mood changes."}
        assert ws._is_relevant(r) is True

    def test_irrelevant_random_tech(self):
        r = {"title": "How to Build a PC", "url": "https://techsite.com", "snippet": "Latest GPU benchmarks and CPU reviews."}
        assert ws._is_relevant(r) is False

    def test_irrelevant_cooking(self):
        r = {"title": "Best Pasta Recipe", "url": "https://recipes.com", "snippet": "Italian pasta with tomato sauce."}
        assert ws._is_relevant(r) is False

    def test_irrelevant_sports(self):
        r = {"title": "NBA Scores", "url": "https://espn.com", "snippet": "Lakers vs Warriors tonight."}
        assert ws._is_relevant(r) is False

    def test_handles_none_values(self):
        r = {"title": None, "url": None, "snippet": None}
        assert ws._is_relevant(r) is False

    def test_handles_empty_result(self):
        r = {}
        assert ws._is_relevant(r) is False

    def test_relevant_by_url_pubmed(self):
        r = {"title": "Study Results", "url": "https://pubmed.ncbi.nlm.nih.gov/12345", "snippet": "A new study on treatment outcomes."}
        assert ws._is_relevant(r) is True

    def test_relevant_by_url_sciencedirect(self):
        r = {"title": "Paper", "url": "https://www.sciencedirect.com/article", "snippet": "Novel findings in therapy."}
        assert ws._is_relevant(r) is True


# ════════════════════════════════════════════════════════════════════════
# _filter_results
# ════════════════════════════════════════════════════════════════════════


class TestFilterResults:
    """Tests for the combined filter pipeline."""

    def test_filters_blocked_content(self):
        results = [
            {"title": "Porn Video", "url": "https://pornhub.com/v/1", "snippet": "..."},
            {"title": "Coping with Anxiety", "url": "https://apa.org/anxiety", "snippet": "Evidence-based therapy."},
        ]
        filtered = ws._filter_results(results, enforce_relevance=True)
        assert len(filtered) == 1
        assert filtered[0]["title"] == "Coping with Anxiety"

    def test_filters_irrelevant_when_enforced(self):
        results = [
            {"title": "How to Cook Pasta", "url": "https://food.com", "snippet": "Recipe for pasta."},
            {"title": "Anxiety Therapy", "url": "https://apa.org/therapy", "snippet": "CBT for anxiety."},
        ]
        filtered = ws._filter_results(results, enforce_relevance=True)
        assert len(filtered) == 1
        assert "Anxiety" in filtered[0]["title"]

    def test_allows_irrelevant_when_not_enforced(self):
        results = [
            {"title": "How to Cook Pasta", "url": "https://food.com", "snippet": "Recipe for pasta."},
        ]
        filtered = ws._filter_results(results, enforce_relevance=False)
        assert len(filtered) == 1

    def test_still_blocks_unsafe_when_not_enforcing_relevance(self):
        results = [
            {"title": "Porn Video", "url": "https://pornhub.com/v/1", "snippet": "..."},
            {"title": "How to Cook Pasta", "url": "https://food.com", "snippet": "Recipe for pasta."},
        ]
        filtered = ws._filter_results(results, enforce_relevance=False)
        assert len(filtered) == 1
        assert filtered[0]["title"] == "How to Cook Pasta"

    def test_empty_input(self):
        assert ws._filter_results([], enforce_relevance=True) == []

    def test_all_blocked(self):
        results = [
            {"title": "Porn", "url": "https://xvideos.com/x", "snippet": "xxx"},
            {"title": "Bokep", "url": "https://xnxx.com/b", "snippet": "adult video"},
        ]
        assert ws._filter_results(results, enforce_relevance=True) == []

    def test_mixed_results_preserves_order(self):
        results = [
            {"title": "Bad", "url": "https://xnxx.com/1", "snippet": "..."},
            {"title": "Good A - Psychology", "url": "https://apa.org/a", "snippet": "Research on emotion."},
            {"title": "Irrelevant", "url": "https://food.com", "snippet": "Pasta recipe."},
            {"title": "Good B - Mental Health", "url": "https://nih.gov/b", "snippet": "Mental health tips."},
        ]
        filtered = ws._filter_results(results, enforce_relevance=True)
        assert len(filtered) == 2
        assert filtered[0]["title"] == "Good A - Psychology"
        assert filtered[1]["title"] == "Good B - Mental Health"


# ════════════════════════════════════════════════════════════════════════
# _safe_query
# ════════════════════════════════════════════════════════════════════════


class TestSafeQuery:
    """Tests for the _safe_query function."""

    def test_already_has_emotion_keyword(self):
        q = "emotion regulation techniques"
        assert ws._safe_query(q) == q  # unchanged

    def test_already_has_psychology(self):
        q = "psychology of happiness"
        assert ws._safe_query(q) == q

    def test_already_has_anxiety(self):
        q = "anxiety reduction strategies"
        assert ws._safe_query(q) == q

    def test_already_has_mental_health(self):
        q = "mental health awareness"
        assert ws._safe_query(q) == q

    def test_generic_query_gets_appended(self):
        q = "surprise response brain"
        result = ws._safe_query(q)
        # Should append scoping terms when no safe keyword is present
        assert "coping strategies" in result.lower()
        assert q in result

    def test_random_query_gets_scoped(self):
        q = "how to deal with problems"
        result = ws._safe_query(q)
        # 'deal' doesn't match safe terms, so scoping is appended
        assert q in result
        assert len(result) > len(q)  # something was appended

    def test_case_insensitive(self):
        q = "EMOTION regulation"
        assert ws._safe_query(q) == q

    def test_coping_keyword_passes(self):
        q = "coping with loss"
        assert ws._safe_query(q) == q


# ════════════════════════════════════════════════════════════════════════
# search_general with safe search
# ════════════════════════════════════════════════════════════════════════


class TestSearchGeneralSafeSearch:
    """Tests for search_general safe-search integration."""

    def test_search_general_calls_ddgs_with_safesearch(self):
        """Verify that safe-search is enforced in search_general."""
        ws.clear_cache()
        mock_ddgs_instance = MagicMock()
        mock_ddgs_instance.text.return_value = [
            {"title": "CBT for Anxiety", "href": "https://apa.org/cbt", "body": "Cognitive behavioral therapy research."},
        ]
        mock_ddgs_cm = MagicMock()
        mock_ddgs_cm.__enter__ = MagicMock(return_value=mock_ddgs_instance)
        mock_ddgs_cm.__exit__ = MagicMock(return_value=False)

        with patch("duckduckgo_search.DDGS", return_value=mock_ddgs_cm):
            results = ws.search_general("anxiety therapy", max_results=5, enforce_relevance=True)

        # Check safesearch parameter was passed
        mock_ddgs_instance.text.assert_called_once()
        call_kwargs = mock_ddgs_instance.text.call_args
        assert call_kwargs[1].get("safesearch") == "on"

    def test_search_general_returns_empty_on_ddg_error(self):
        """When DDG is unavailable, search_general returns empty list."""
        ws.clear_cache()
        with patch("duckduckgo_search.DDGS", side_effect=ImportError("no module")):
            results = ws.search_general("surprise", max_results=5)

        assert results == []

    def test_search_emotion_articles_safe(self):
        """search_emotion_articles should produce safe results."""
        ws.clear_cache()
        with patch("duckduckgo_search.DDGS", side_effect=ImportError("no module")):
            results = ws.search_emotion_articles("surprise", max_results=5)
        for r in results:
            assert not ws._is_blocked(r)

    def test_search_coping_strategies_safe(self):
        """search_coping_strategies should produce safe results."""
        ws.clear_cache()
        with patch("duckduckgo_search.DDGS", side_effect=ImportError("no module")):
            results = ws.search_coping_strategies("sad", max_results=5)
        for r in results:
            assert not ws._is_blocked(r)
            # Demo coping results should all be relevant
            assert ws._is_relevant(r), f"Coping result should be relevant: {r['title']}"


# ════════════════════════════════════════════════════════════════════════
# search_general filters out unsafe live results
# ════════════════════════════════════════════════════════════════════════


class TestSearchGeneralFiltering:
    """Tests that search_general properly filters live DDG results."""

    def test_filters_out_porn_from_live_results(self):
        """Unsafe results from DDG should be filtered out."""
        ws.clear_cache()
        mock_ddgs_instance = MagicMock()
        mock_ddgs_instance.text.return_value = [
            {"title": "Bokep Porn Videos", "href": "https://xhamster.com/search/bokep", "body": "..."},
            {"title": "surprise和suprise", "href": "https://zhidao.baidu.com/question/123", "body": "surprise"},
            {"title": "CBT for Anxiety - APA", "href": "https://www.apa.org/cbt", "body": "Cognitive behavioral therapy."},
            {"title": "Emotion Regulation Research", "href": "https://www.nature.com/articles/123", "body": "Brain research on emotion."},
        ]
        mock_ddgs_cm = MagicMock()
        mock_ddgs_cm.__enter__ = MagicMock(return_value=mock_ddgs_instance)
        mock_ddgs_cm.__exit__ = MagicMock(return_value=False)

        with patch("duckduckgo_search.DDGS", return_value=mock_ddgs_cm):
            results = ws.search_general("surprise emotion", max_results=5, enforce_relevance=True)

        titles = [r["title"] for r in results]
        assert "Bokep Porn Videos" not in titles
        assert "surprise和suprise" not in titles
        assert "CBT for Anxiety - APA" in titles
        assert "Emotion Regulation Research" in titles

    def test_filters_preserves_relevant_results(self):
        """All relevant, safe results should pass through."""
        ws.clear_cache()
        mock_ddgs_instance = MagicMock()
        mock_ddgs_instance.text.return_value = [
            {"title": "Mindfulness for Stress", "href": "https://apa.org/mindfulness", "body": "Mindfulness meditation."},
            {"title": "Depression Treatment Options", "href": "https://nih.gov/depression", "body": "Evidence-based mental health."},
        ]
        mock_ddgs_cm = MagicMock()
        mock_ddgs_cm.__enter__ = MagicMock(return_value=mock_ddgs_instance)
        mock_ddgs_cm.__exit__ = MagicMock(return_value=False)

        with patch("duckduckgo_search.DDGS", return_value=mock_ddgs_cm):
            results = ws.search_general("mental health", max_results=5, enforce_relevance=True)

        assert len(results) == 2

