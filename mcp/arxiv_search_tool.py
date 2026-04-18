"""
MCP Tool 1: arxiv_search_tool (Enhanced)
=========================================
Features: category filtering, date range, retry with backoff.
"""
from fastmcp import FastMCP
import arxiv
import os
import time
import logging
from datetime import datetime, timezone
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_ARXIV_PORT = int(os.environ.get("MCP_ARXIV_PORT", "8001"))
mcp = FastMCP("ArxivSearchTool")

# ArXiv categories focused on mental health and emotion research
ARXIV_CATEGORIES = {
    "cs.HC": "Human-Computer Interaction (emotion-aware systems, mental health apps)",
    "cs.CL": "Computation & Language (sentiment and emotion analysis in text)",
    "cs.CV": "Computer Vision (facial emotion recognition)",
    "eess.AS": "Audio and Speech Processing (speech emotion recognition)",
    "q-bio.NC": "Neurons and Cognition (neuroscience of emotion and mental health)",
    "q-bio.QM": "Quantitative Methods (quantitative mental health modelling)",
}


def fetch_with_retry(client: arxiv.Client, search: arxiv.Search, max_retries: int = 3) -> list:
    """Retry ArXiv API calls with exponential backoff (1s, 2s, 4s)."""
    for attempt in range(max_retries):
        try:
            results = list(client.results(search))
            return results
        except Exception as e:
            wait = 2 ** attempt  # 1s, 2s, 4s
            if attempt < max_retries - 1:
                logger.warning(f"ArXiv fetch failed (attempt {attempt+1}): {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                logger.error(f"ArXiv fetch failed after {max_retries} attempts: {e}")
                raise


@mcp.tool()
def search_arxiv(
    query: str,
    max_results: int = 5,
    sort_by: str = "date",
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    categories: Optional[list[str]] = None,
) -> list[dict]:
    """Search ArXiv for academic papers matching the query.

    Args:
        query: Search query string (supports AND/OR/NOT operators)
        max_results: Number of results to return (5-50)
        sort_by: Sort order - "date", "relevance", or "lastUpdated"
        date_from: Start date filter (YYYY-MM-DD)
        date_to: End date filter (YYYY-MM-DD)
        categories: ArXiv category filter, e.g. ["cs.AI", "cs.LG"]

    Returns:
        List of dicts with keys: title, abstract, authors, url, published, categories, pdf_url
    """
    max_results = min(max_results, 50)

    # Build query with optional category filter
    if categories:
        cat_filter = " OR ".join(f"cat:{c}" for c in categories)
        full_query = f"({query}) AND ({cat_filter})"
    else:
        full_query = query

    # Map sort option
    sort_map = {
        "date": arxiv.SortCriterion.SubmittedDate,
        "relevance": arxiv.SortCriterion.Relevance,
        "lastUpdated": arxiv.SortCriterion.LastUpdatedDate,
    }
    sort_criterion = sort_map.get(sort_by, arxiv.SortCriterion.SubmittedDate)

    client = arxiv.Client(
        page_size=min(max_results, 25),
        delay_seconds=1.0,  # ArXiv rate limit
        num_retries=3,
    )
    search = arxiv.Search(
        query=full_query,
        max_results=max_results,
        sort_by=sort_criterion,
    )

    try:
        raw_results = fetch_with_retry(client, search)
    except Exception as e:
        logger.error(f"search_arxiv failed: {e}")
        return [{"error": str(e), "query": query}]

    # Apply date filters (ArXiv API date filtering is unreliable, so we filter locally)
    date_from_dt = None
    date_to_dt = None
    if date_from:
        date_from_dt = datetime.strptime(date_from, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    if date_to:
        date_to_dt = datetime.strptime(date_to, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    results = []
    for paper in raw_results:
        pub_date = paper.published
        if date_from_dt and pub_date < date_from_dt:
            continue
        if date_to_dt and pub_date > date_to_dt:
            continue
        results.append({
            "title": paper.title,
            "abstract": paper.summary[:800],
            "authors": [a.name for a in paper.authors[:5]],  # Top 5 authors
            "url": paper.entry_id,
            "pdf_url": paper.pdf_url,
            "published": str(paper.published.date()),
            "categories": paper.categories,
            "primary_category": paper.primary_category,
        })

    logger.info(f"search_arxiv: query={query}, returned {len(results)} results")
    return results


@mcp.tool()
def list_arxiv_categories() -> dict:
    """List all supported ArXiv categories with descriptions."""
    return ARXIV_CATEGORIES


if __name__ == "__main__":
    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=_ARXIV_PORT,
        path="/mcp",
    )
