"""Tests for ArXiv safe-query scoping and paper relevance filtering.

These tests verify that:
1. _arxiv_safe_query() adds 'emotion psychology' prefix when needed
2. _is_relevant_paper() rejects particle physics / astrophysics papers
3. _is_relevant_paper() accepts psychology / emotion papers
4. _search_arxiv() integrates both filters correctly
"""
import unittest
from unittest.mock import patch, MagicMock

from diary.diary_session import DiarySessionManager


class TestArxivSafeQuery(unittest.TestCase):
    """Test _arxiv_safe_query scopes queries to psychology/emotion domain."""

    def test_no_prefix_when_emotion_present(self):
        """Queries already containing 'emotion' should stay unchanged."""
        q = "emotion regulation strategies"
        self.assertEqual(DiarySessionManager._arxiv_safe_query(q), q)

    def test_no_prefix_when_psychology_present(self):
        q = "developmental psychology adolescents"
        self.assertEqual(DiarySessionManager._arxiv_safe_query(q), q)

    def test_no_prefix_when_anxiety_present(self):
        q = "anxiety reduction CBT"
        self.assertEqual(DiarySessionManager._arxiv_safe_query(q), q)

    def test_no_prefix_when_therapy_present(self):
        q = "exposure therapy fear conditioning"
        self.assertEqual(DiarySessionManager._arxiv_safe_query(q), q)

    def test_no_prefix_when_depression_present(self):
        q = "treatment resistant depression"
        self.assertEqual(DiarySessionManager._arxiv_safe_query(q), q)

    def test_no_prefix_when_mindfulness_present(self):
        q = "mindfulness based stress reduction"
        self.assertEqual(DiarySessionManager._arxiv_safe_query(q), q)

    def test_no_prefix_when_neuroscience_present(self):
        q = "affective neuroscience patterns"
        self.assertEqual(DiarySessionManager._arxiv_safe_query(q), q)

    def test_prefix_added_for_generic_query(self):
        """Vague queries need 'emotion psychology' prefix to avoid physics results."""
        q = "ways to feel better after a bad day"
        result = DiarySessionManager._arxiv_safe_query(q)
        self.assertTrue(result.startswith("emotion psychology "))
        self.assertIn("ways to feel better after a bad day", result)

    def test_prefix_added_for_bare_query(self):
        q = "how to relax after work"
        result = DiarySessionManager._arxiv_safe_query(q)
        self.assertTrue(result.startswith("emotion psychology "))

    def test_coping_is_recognised(self):
        """'coping' is an academic term — should NOT get prefix."""
        q = "coping mechanisms adults"
        self.assertEqual(DiarySessionManager._arxiv_safe_query(q), q)

    def test_case_insensitive(self):
        """Academic term matching should be case-insensitive."""
        q = "DEPRESSION treatment fMRI"
        self.assertEqual(DiarySessionManager._arxiv_safe_query(q), q)


class TestIsRelevantPaper(unittest.TestCase):
    """Test _is_relevant_paper rejects physics/astro and accepts psych papers."""

    # ── Particle physics papers (MUST be rejected) ──

    def test_rejects_atlas_detector(self):
        paper = {"title": "Search for new phenomena with the ATLAS detector at LHC",
                 "abstract": "A search for new particles using proton-proton collisions."}
        self.assertFalse(DiarySessionManager._is_relevant_paper(paper))

    def test_rejects_cms_experiment(self):
        paper = {"title": "CMS measurement of Higgs boson production",
                 "abstract": "Precision measurement of the Higgs boson."}
        self.assertFalse(DiarySessionManager._is_relevant_paper(paper))

    def test_rejects_lhcb_paper(self):
        paper = {"title": "LHCb observation of new meson decay",
                 "abstract": "Study of b-quark meson decays at the LHC."}
        self.assertFalse(DiarySessionManager._is_relevant_paper(paper))

    def test_rejects_dark_matter(self):
        paper = {"title": "Constraints on dark matter annihilation",
                 "abstract": "We study dark matter candidates using galaxy surveys."}
        self.assertFalse(DiarySessionManager._is_relevant_paper(paper))

    def test_rejects_gravitational_waves(self):
        paper = {"title": "Detection of gravitational wave signals",
                 "abstract": "Analysis of LIGO gravitational wave events."}
        self.assertFalse(DiarySessionManager._is_relevant_paper(paper))

    def test_rejects_cosmology(self):
        paper = {"title": "Cosmological constraints from CMB",
                 "abstract": "We use cosmology data to constrain parameters."}
        self.assertFalse(DiarySessionManager._is_relevant_paper(paper))

    def test_rejects_stellar_astrophysics(self):
        paper = {"title": "Stellar evolution in globular clusters",
                 "abstract": "We study stellar populations in astrophysical environments."}
        self.assertFalse(DiarySessionManager._is_relevant_paper(paper))

    # ── Psychology / emotion papers (MUST be accepted) ──

    def test_accepts_emotion_regulation(self):
        paper = {"title": "Emotion regulation in adolescents",
                 "abstract": "We studied cognitive reappraisal in emotional regulation."}
        self.assertTrue(DiarySessionManager._is_relevant_paper(paper))

    def test_accepts_anxiety_study(self):
        paper = {"title": "Social anxiety and cognitive bias",
                 "abstract": "This study examines anxiety disorders and their cognitive correlates."}
        self.assertTrue(DiarySessionManager._is_relevant_paper(paper))

    def test_accepts_depression_research(self):
        paper = {"title": "Neural correlates of depression",
                 "abstract": "Brain imaging reveals patterns in depression and mood disorders."}
        self.assertTrue(DiarySessionManager._is_relevant_paper(paper))

    def test_accepts_facial_expression(self):
        paper = {"title": "Facial expression recognition in the wild",
                 "abstract": "Deep learning for facial expression and sentiment analysis."}
        self.assertTrue(DiarySessionManager._is_relevant_paper(paper))

    def test_accepts_mindfulness(self):
        paper = {"title": "Effects of mindfulness meditation on stress",
                 "abstract": "Mindfulness-based intervention reduced stress and improved well-being."}
        self.assertTrue(DiarySessionManager._is_relevant_paper(paper))

    def test_accepts_therapy_paper(self):
        paper = {"title": "CBT for trauma survivors",
                 "abstract": "Cognitive behavioral therapy improved mental health outcomes."}
        self.assertTrue(DiarySessionManager._is_relevant_paper(paper))

    # ── Edge cases ──

    def test_rejects_ambiguous_no_indicators(self):
        """Paper with no psych or physics terms → reject (safe default)."""
        paper = {"title": "A novel algorithm for optimization",
                 "abstract": "We present a new method for combinatorial optimization."}
        self.assertFalse(DiarySessionManager._is_relevant_paper(paper))

    def test_rejects_physics_even_with_emotion_in_abstract(self):
        """If a physics paper mentions 'emotion' vaguely, reject_terms should still win."""
        paper = {"title": "Measurement of muon neutrino interactions",
                 "abstract": "We detected neutrino events. This work may emotion the community."}
        self.assertFalse(DiarySessionManager._is_relevant_paper(paper))

    def test_handles_missing_fields(self):
        """Should not crash on missing title/abstract."""
        self.assertFalse(DiarySessionManager._is_relevant_paper({}))
        self.assertFalse(DiarySessionManager._is_relevant_paper({"title": None, "abstract": None}))

    def test_handles_empty_strings(self):
        paper = {"title": "", "abstract": ""}
        self.assertFalse(DiarySessionManager._is_relevant_paper(paper))


class TestSearchArxivIntegration(unittest.TestCase):
    """Test _search_arxiv integrates safe_query + relevance filter."""

    @patch("mcp_client.search_papers")
    def test_filters_out_physics_papers(self, mock_search):
        """Physics papers from ArXiv should be filtered out of results."""
        mock_search.return_value = [
            {"title": "ATLAS detector at LHC", "url": "https://arxiv.org/abs/1", "abstract": "proton collider"},
            {"title": "Emotion regulation in adults", "url": "https://arxiv.org/abs/2", "abstract": "psychology emotion study"},
            {"title": "CMS Higgs boson", "url": "https://arxiv.org/abs/3", "abstract": "particle boson"},
        ]
        results = DiarySessionManager._search_arxiv(["coping with stress"])
        # Only the emotion paper should survive
        self.assertEqual(len(results), 1)
        self.assertIn("Emotion regulation", results[0]["title"])

    @patch("mcp_client.search_papers")
    def test_deduplicates_by_url(self, mock_search):
        """Same paper from different queries should appear only once."""
        paper = {"title": "Mood study", "url": "https://arxiv.org/abs/99",
                 "abstract": "A study of mood and emotion regulation."}
        mock_search.return_value = [paper]
        results = DiarySessionManager._search_arxiv(["mood", "emotion"], max_per_query=5)
        urls = [r["url"] for r in results]
        self.assertEqual(len(set(urls)), len(urls))

    @patch("mcp_client.search_papers")
    def test_respects_max_per_query(self, mock_search):
        """Should not return more than max_per_query results per query."""
        papers = [
            {"title": f"Emotion study {i}", "url": f"https://arxiv.org/abs/{i}",
             "abstract": "psychology emotion affective"}
            for i in range(10)
        ]
        mock_search.return_value = papers
        results = DiarySessionManager._search_arxiv(["emotions"], max_per_query=2)
        self.assertLessEqual(len(results), 2)

    def test_returns_empty_on_import_error(self):
        """Should gracefully return [] if mcp_client is unavailable."""
        with patch.dict("sys.modules", {"mcp_client": None}):
            results = DiarySessionManager._search_arxiv(["test"])
            self.assertEqual(results, [])

    @patch("mcp_client.search_papers", side_effect=Exception("network error"))
    def test_handles_search_exception(self, mock_search):
        """Network errors should be caught, not crash the session."""
        results = DiarySessionManager._search_arxiv(["test query"])
        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
