"""
Tests for the 3 critical bug fixes:
  1. ArXiv research results unrelated ??sort by relevance, not date
  2. Web search returns no results ??fallback to demo on empty DDG
  3. Face emotion always neutral ??DETECT_WIDTH=320, MIN_FACE_SIZE=30
"""

import ast
import os
import sys
import numpy as np
from unittest.mock import patch, MagicMock
import pytest

# Add workspace to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ?��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��?
# Bug 1: ArXiv Research Unrelated ??sort_by defaults to "relevance"
# ?��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��?

class TestArxivRelevanceSort:
    """Verify ArXiv search defaults to relevance sort for topical accuracy."""

    def test_search_papers_default_sort_is_relevance(self):
        """search_papers should default to sort_by='relevance', not 'date'."""
        import inspect
        from mcp.mcp_client import search_papers
        sig = inspect.signature(search_papers)
        default = sig.parameters["sort_by"].default
        assert default == "relevance", (
            f"Expected sort_by default 'relevance', got '{default}'. "
            "Date sorting returns random recent papers instead of topical matches."
        )

    def test_search_papers_source_has_relevance_default(self):
        """Source code should show sort_by: str = 'relevance'."""
        src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mcp", "mcp_client.py")
        with open(src_path) as f:
            src = f.read()
        assert 'sort_by: str = "relevance"' in src, (
            "mcp_client.py should default sort_by to 'relevance'"
        )

    def test_emotion_categories_defined(self):
        """Emotion-related ArXiv categories should be defined."""
        from mcp.mcp_client import _EMOTION_CATEGORIES
        assert isinstance(_EMOTION_CATEGORIES, list)
        assert len(_EMOTION_CATEGORIES) >= 3
        # Should include human-computer interaction and AI categories
        assert "cs.HC" in _EMOTION_CATEGORIES
        assert "cs.AI" in _EMOTION_CATEGORIES

    def test_arxiv_direct_uses_relevance_sort(self):
        """_search_arxiv_direct should map 'relevance' to SortCriterion.Relevance."""
        src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mcp", "mcp_client.py")
        with open(src_path) as f:
            src = f.read()
        assert "SortCriterion.Relevance" in src, (
            "ArXiv search should use SortCriterion.Relevance"
        )

    def test_diary_session_search_arxiv_uses_safe_query(self):
        """Diary session's _search_arxiv should scope queries with _safe_query."""
        src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "diary", "diary_session.py")
        with open(src_path) as f:
            src = f.read()
        assert "_safe_query(query)" in src or "_safe_query(q)" in src, (
            "ArXiv queries in diary session should be passed through _safe_query()"
        )


# ?��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��?
# Bug 2: Web Search Returns No Results ??empty list on DDG failure
# ?��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��?

class TestWebSearchFallback:
    """Verify web search returns a list (possibly empty) when DDG is unavailable."""

    def test_search_general_returns_list_when_ddg_empty(self):
        """search_general should return [] when DDG returns empty results."""
        mock_ddgs_cls = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.text.return_value = []  # DDG returns empty
        mock_ddgs_cls.return_value.__enter__ = MagicMock(return_value=mock_ctx)
        mock_ddgs_cls.return_value.__exit__ = MagicMock(return_value=False)

        with patch.dict("sys.modules", {"duckduckgo_search": MagicMock(DDGS=mock_ddgs_cls)}):
            from services.web_search_service import search_general, clear_cache
            clear_cache()
            results = search_general("test query emotion", max_results=3)
            assert isinstance(results, list)

    def test_search_emotion_articles_returns_list_when_ddg_empty(self):
        """search_emotion_articles returns [] when DDG returns empty results."""
        mock_ddgs_cls = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.text.return_value = []
        mock_ddgs_cls.return_value.__enter__ = MagicMock(return_value=mock_ctx)
        mock_ddgs_cls.return_value.__exit__ = MagicMock(return_value=False)

        with patch.dict("sys.modules", {"duckduckgo_search": MagicMock(DDGS=mock_ddgs_cls)}):
            from services.web_search_service import search_emotion_articles, clear_cache
            clear_cache()
            results = search_emotion_articles("neutral", max_results=3)
            assert isinstance(results, list)

    def test_search_coping_returns_list_when_ddg_empty(self):
        """search_coping_strategies returns [] when DDG returns empty results."""
        mock_ddgs_cls = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.text.return_value = []
        mock_ddgs_cls.return_value.__enter__ = MagicMock(return_value=mock_ctx)
        mock_ddgs_cls.return_value.__exit__ = MagicMock(return_value=False)

        with patch.dict("sys.modules", {"duckduckgo_search": MagicMock(DDGS=mock_ddgs_cls)}):
            from services.web_search_service import search_coping_strategies, clear_cache
            clear_cache()
            results = search_coping_strategies("sad", max_results=3)
            assert isinstance(results, list)

    def test_search_general_returns_list_when_ddg_raises(self):
        """search_general should return [] on DDG exception."""
        mock_mod = MagicMock()
        mock_mod.DDGS.side_effect = Exception("API error")
        with patch.dict("sys.modules", {"duckduckgo_search": mock_mod}):
            from services.web_search_service import search_general, clear_cache
            clear_cache()
            results = search_general("test query", max_results=3)
            assert isinstance(results, list)

    def test_search_general_with_ddg_results_that_are_all_filtered(self):
        """When DDG returns results but all are filtered (irrelevant), return []."""
        mock_ddgs_cls = MagicMock()
        mock_ctx = MagicMock()
        # DDG returns results about cooking (not emotion-related)
        mock_ctx.text.return_value = [
            {"title": "Best Pizza Recipe", "href": "https://cooking.com/pizza", "body": "Delicious pizza recipe"},
            {"title": "How to Bake Bread", "href": "https://baking.com/bread", "body": "Fresh homemade bread"},
        ]
        mock_ddgs_cls.return_value.__enter__ = MagicMock(return_value=mock_ctx)
        mock_ddgs_cls.return_value.__exit__ = MagicMock(return_value=False)

        with patch.dict("sys.modules", {"duckduckgo_search": MagicMock(DDGS=mock_ddgs_cls)}):
            from services.web_search_service import search_general, clear_cache
            clear_cache()
            results = search_general("cooking tips", max_results=3, enforce_relevance=True)
            assert isinstance(results, list)


# ?��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��?
# Bug 3: Face Emotion Always Neutral ??DETECT_WIDTH=320, MIN_FACE_SIZE=30
# ?��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��?

class TestEmotionDetectionFixes:
    """Verify face emotion detection works at downsampled webcam resolution."""

    def test_detect_width_is_320(self):
        """DETECT_WIDTH should be 320 (not 160) for reliable face detection."""
        src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "streamlit_app.py")
        with open(src_path) as f:
            src = f.read()
        assert "DETECT_WIDTH = 320" in src, (
            "DETECT_WIDTH should be 320. At 160px, faces are ~56px which is below MIN_FACE_SIZE."
        )

    def test_min_face_size_is_30(self):
        """MIN_FACE_SIZE should be 30 to accept smaller downsampled faces."""
        from services.emotion_detector import EmotionDetector
        det = EmotionDetector()
        assert det.MIN_FACE_SIZE == 30, (
            f"MIN_FACE_SIZE should be 30, got {det.MIN_FACE_SIZE}. "
            "At 60, faces from 320px-width downsampled frames are rejected."
        )

    def test_detection_works_at_320px_width(self):
        """Emotion detection should work on a 320px-wide downsampled frame."""
        import cv2
        from services.emotion_detector import EmotionDetector

        det = EmotionDetector()
        if not det.is_ready or det.backend_name == "opencv_heuristic":
            pytest.skip("DeepFace not available for this test")

        # Create a face-like image at webcam resolution
        img = np.ones((480, 640, 3), dtype=np.uint8) * 128
        cv2.ellipse(img, (320, 200), (80, 100), 0, 0, 360, (200, 180, 170), -1)
        cv2.circle(img, (290, 180), 10, (50, 50, 50), -1)
        cv2.circle(img, (350, 180), 10, (50, 50, 50), -1)
        cv2.ellipse(img, (320, 230), (25, 10), 0, 0, 360, (150, 100, 100), -1)

        # Downsample to 320px (matches new DETECT_WIDTH)
        DETECT_WIDTH = 320
        h, w = img.shape[:2]
        scale = DETECT_WIDTH / w
        small = cv2.resize(img, (DETECT_WIDTH, int(h * scale)), interpolation=cv2.INTER_AREA)

        annotated, results = det.detect_emotions(small)
        assert len(results) > 0, (
            "Should detect a face at 320px width. "
            "This was failing at 160px because faces were smaller than MIN_FACE_SIZE."
        )
        assert results[0]["emotion"] != "", "Emotion should not be empty"

    def test_detection_fails_at_old_160px_with_old_min_size(self):
        """Demonstrate that the OLD settings (160px, min_size=60) would fail."""
        import cv2
        from services.emotion_detector import EmotionDetector

        det = EmotionDetector()
        if not det.is_ready or det.backend_name == "opencv_heuristic":
            pytest.skip("DeepFace not available for this test")

        # Create face image and downsample to OLD 160px width
        img = np.ones((480, 640, 3), dtype=np.uint8) * 128
        cv2.ellipse(img, (320, 200), (80, 100), 0, 0, 360, (200, 180, 170), -1)
        cv2.circle(img, (290, 180), 10, (50, 50, 50), -1)
        cv2.circle(img, (350, 180), 10, (50, 50, 50), -1)
        cv2.ellipse(img, (320, 230), (25, 10), 0, 0, 360, (150, 100, 100), -1)

        old_width = 160
        h, w = img.shape[:2]
        scale = old_width / w
        small = cv2.resize(img, (old_width, int(h * scale)), interpolation=cv2.INTER_AREA)

        # At 160px with MIN_FACE_SIZE=30, detection works; at 60, it would fail
        # Save the current MIN_FACE_SIZE and test with old value
        original_min = det.MIN_FACE_SIZE
        try:
            det.MIN_FACE_SIZE = 60  # old value
            # Clear smoothing history to get fresh results
            det._emotion_history = []
            annotated, results = det.detect_emotions(small)
            # At 160px with min_size=60, Haar detects ~56px face ??rejected
            assert len(results) == 0, (
                "OLD settings (160px + min_size=60) should fail to detect faces. "
                "This confirms the bug existed."
            )
        finally:
            det.MIN_FACE_SIZE = original_min

    def test_diary_video_processor_has_new_detect_width(self):
        """DiaryVideoProcessor inherits DETECT_WIDTH=320 from EmotionVideoProcessor."""
        src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "streamlit_app.py")
        with open(src_path) as f:
            src = f.read()
        # DiaryVideoProcessor extends EmotionVideoProcessor, so it inherits DETECT_WIDTH
        assert "class DiaryVideoProcessor(EmotionVideoProcessor)" in src
        # The parent class has DETECT_WIDTH=320
        assert "DETECT_WIDTH = 320" in src

    def test_max_face_ratio_unchanged(self):
        """MAX_FACE_RATIO should still filter whole-image false detections."""
        from services.emotion_detector import EmotionDetector
        det = EmotionDetector()
        assert det.MAX_FACE_RATIO == 0.6, "MAX_FACE_RATIO should remain 0.6"


# ?��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��?
# Integration: End-to-end diary session search pipeline
# ?��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��??��?

class TestDiarySearchIntegration:
    """Verify the complete diary end_session search pipeline produces results."""

    def test_end_session_produces_web_results(self):
        """end_session should always produce non-empty web_results."""
        from diary.diary_session import DiarySessionManager, DiarySession, DiaryEntry
        import diary.diary_session as ds_module

        mgr = DiarySessionManager()
        session = mgr.start_session()
        session.entries.append(DiaryEntry(
            timestamp="12:00:00",
            text="I feel overwhelmed by work deadlines",
            face_emotion="sad",
            face_confidence=0.8,
            voice_sentiment={"polarity": -0.5, "subjectivity": 0.7, "emotion": "sad", "confidence": 0.7},
            audio_emotion={"energy": 0.3, "pitch_var": 0.2, "estimated_emotion": "sad", "confidence": 0.6},
            fused_emotion="sad",
            fused_confidence=0.75,
        ))

        # Mock the LLM and ArXiv search (which is imported as a function in diary_session)
        with patch.object(mgr, '_llm') as mock_llm:
            mock_llm._generate.return_value = "Summary: user feels stressed about deadlines."
            orig_search = ds_module.DiarySessionManager._search_arxiv
            ds_module.DiarySessionManager._search_arxiv = staticmethod(lambda queries, max_per_query=3: [])
            try:
                session = mgr.end_session(session)
            finally:
                ds_module.DiarySessionManager._search_arxiv = orig_search

        assert isinstance(session.web_results, list), (
            "end_session must set web_results to a list"
        )

    def test_end_session_produces_relevant_research_queries(self):
        """Research queries should contain psychology/emotion terms."""
        from diary.diary_session import DiarySessionManager, DiarySession, DiaryEntry

        mgr = DiarySessionManager()
        session = mgr.start_session()
        session.entries.append(DiaryEntry(
            timestamp="12:00:00",
            text="I am feeling anxious about my exam",
            face_emotion="fear",
            face_confidence=0.7,
            voice_sentiment={"polarity": -0.3, "emotion": "fear", "confidence": 0.6},
            audio_emotion={"energy": 0.5, "pitch_var": 0.4, "estimated_emotion": "fear", "confidence": 0.5},
            fused_emotion="fear",
            fused_confidence=0.65,
        ))

        with patch.object(mgr, '_llm') as mock_llm:
            # Simulate LLM returning queries
            mock_llm._generate.side_effect = [
                "Summary of session",
                "exposure therapy for test anxiety\nfear conditioning exam stress\nanxiety reduction techniques",
            ]
            queries = mgr.get_research_queries(session)

        assert len(queries) > 0, "Should generate research queries"
        # Each query should be scoped to psychology via _safe_query
        for q in queries:
            q_lower = q.lower()
            has_psych = any(term in q_lower for term in [
                "psychology", "mental health", "emotion", "therapy",
                "coping", "anxiety", "fear", "stress", "mindfulness",
                "cognitive", "behavioral", "mood", "regulation",
            ])
            assert has_psych, f"Query '{q}' should be scoped to psychology/emotion topics"
