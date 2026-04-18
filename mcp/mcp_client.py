"""
EmotiScan: MCP client for ArXiv search and relevance ranking.
Falls back to direct API/local TF-IDF when MCP servers are down.
"""

import socket
import logging

from config import MCP_RANKER_HOST, MCP_RANKER_PORT, MCP_ARXIV_HOST, MCP_ARXIV_PORT

logger = logging.getLogger(__name__)


def _check_port(host: str, port: int, timeout: float = 1.0) -> bool:
    """Check if a TCP port is open."""
    try:
        s = socket.create_connection((host, port), timeout=timeout)
        s.close()
        return True
    except Exception:
        return False


# ArXiv categories focused on mental health and emotion research
_EMOTION_CATEGORIES = ["cs.HC", "cs.CL", "cs.CV", "eess.AS", "q-bio.NC", "q-bio.QM"]


def search_papers(query: str, max_results: int = 5, sort_by: str = "relevance",
                  date_from: str = None, date_to: str = None,
                  categories: list = None) -> list:
    """
    Search ArXiv for papers. Tries MCP server first, falls back to direct API.

    Args:
        sort_by: "relevance" (default, best for topical queries) or "date"
        categories: ArXiv categories to filter by. Defaults to emotion/psych-related.

    Returns:
        list of dicts with title, abstract, authors, url, published, etc.
    """
    # Try arxiv MCP server first
    if _check_port(MCP_ARXIV_HOST, MCP_ARXIV_PORT):
        try:
            return _search_via_mcp(query, max_results, sort_by, date_from, date_to, categories)
        except Exception as e:
            logger.warning(f"ArXiv MCP search failed: {e}")

    # Fallback to direct arxiv API
    try:
        return _search_arxiv_direct(query, max_results, sort_by, date_from, date_to, categories)
    except Exception as e:
        logger.warning(f"ArXiv direct search failed: {e}")
        return []


def rank_results(interests: str, articles: list) -> list:
    """
    Rank articles by relevance. Tries MCP ranker, falls back to local TF-IDF.

    Returns:
        Sorted list of articles with relevance_score added.
    """
    if not articles:
        return articles

    # Try MCP ranker server
    if _check_port(MCP_RANKER_HOST, MCP_RANKER_PORT):
        try:
            return _rank_via_mcp(interests, articles)
        except Exception as e:
            logger.warning(f"MCP ranker failed: {e}")

    # Fallback to local TF-IDF
    return _rank_local(interests, articles)


def _search_via_mcp(query: str, max_results: int = 5, sort_by: str = "relevance",
                    date_from: str = None, date_to: str = None,
                    categories: list = None) -> list:
    """Search ArXiv via the MCP arxiv server HTTP endpoint."""
    import httpx
    import json

    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "search_arxiv",
            "arguments": {
                "query": query,
                "max_results": max_results,
                "sort_by": sort_by,
                "date_from": date_from,
                "date_to": date_to,
                "categories": categories,
            },
        },
    }
    resp = httpx.post(
        f"http://{MCP_ARXIV_HOST}:{MCP_ARXIV_PORT}/mcp",
        json=payload,
        timeout=30.0,
    )
    resp.raise_for_status()
    result = resp.json()
    if "result" in result:
        content = result["result"]
        if isinstance(content, list):
            return content
        if isinstance(content, dict) and "content" in content:
            for item in content["content"]:
                if item.get("type") == "text":
                    return json.loads(item["text"])
    raise ValueError("Unexpected MCP response format")


def _search_arxiv_direct(query: str, max_results: int = 5, sort_by: str = "date",
                         date_from: str = None, date_to: str = None,
                         categories: list = None) -> list:
    """Direct ArXiv API search."""
    import arxiv
    from datetime import datetime, timezone

    full_query = query
    if categories:
        cat_filter = " OR ".join(f"cat:{c}" for c in categories)
        full_query = f"({query}) AND ({cat_filter})"

    sort_map = {
        "date": arxiv.SortCriterion.SubmittedDate,
        "relevance": arxiv.SortCriterion.Relevance,
        "lastUpdated": arxiv.SortCriterion.LastUpdatedDate,
    }
    client = arxiv.Client(page_size=min(max_results, 25), delay_seconds=1.0, num_retries=3)
    search = arxiv.Search(
        query=full_query,
        max_results=max_results,
        sort_by=sort_map.get(sort_by, arxiv.SortCriterion.SubmittedDate),
    )

    date_from_dt = None
    date_to_dt = None
    if date_from:
        date_from_dt = datetime.strptime(date_from, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    if date_to:
        date_to_dt = datetime.strptime(date_to, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    results = []
    for paper in client.results(search):
        if date_from_dt and paper.published < date_from_dt:
            continue
        if date_to_dt and paper.published > date_to_dt:
            continue
        results.append({
            "title": paper.title,
            "abstract": paper.summary[:800],
            "authors": [a.name for a in paper.authors[:5]],
            "url": paper.entry_id,
            "pdf_url": paper.pdf_url,
            "published": str(paper.published.date()),
            "categories": paper.categories,
            "source": "arxiv",
        })
    return results


def _rank_via_mcp(interests: str, articles: list) -> list:
    """Rank via the MCP ranker HTTP endpoint."""
    import httpx

    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "rank_articles",
            "arguments": {
                "user_interests": interests,
                "articles": articles,
            },
        },
    }
    resp = httpx.post(
        f"http://{MCP_RANKER_HOST}:{MCP_RANKER_PORT}/mcp",
        json=payload,
        timeout=10.0,
    )
    resp.raise_for_status()
    result = resp.json()
    if "result" in result:
        content = result["result"]
        if isinstance(content, list):
            return content
        # Handle MCP response format
        if isinstance(content, dict) and "content" in content:
            import json
            for item in content["content"]:
                if item.get("type") == "text":
                    return json.loads(item["text"])
    # If MCP response format is unexpected, fall back
    raise ValueError("Unexpected MCP response format")


def _rank_local(interests: str, articles: list) -> list:
    """Local TF-IDF ranking fallback."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    if not articles:
        return articles

    corpus = [interests] + [
        a.get("title", "") + " " + a.get("abstract", a.get("content", a.get("title", "")))
        for a in articles
    ]
    vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True)
    tfidf_matrix = vectorizer.fit_transform(corpus)
    scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    for i, article in enumerate(articles):
        article["relevance_score"] = round(float(scores[i]), 4)
    return sorted(articles, key=lambda x: x["relevance_score"], reverse=True)
