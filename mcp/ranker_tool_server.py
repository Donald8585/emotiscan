"""
Tool 3 - Relevance Ranker MCP Server (Streamable HTTP)
========================================================
Run:  python ranker_tool_server.py
Test: MCP Inspector -> http://localhost:8000/mcp  (port overridable via MCP_RANKER_PORT env var)
"""
import os
from fastmcp import FastMCP
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

mcp = FastMCP("RelevanceRankerTool")


@mcp.tool()
def rank_articles(user_interests: str, articles: list[dict]) -> list[dict]:
    """Rank articles by cosine similarity to user interest profile.
    Each article dict must have 'title' and 'abstract' or 'content'.
    """
    if not articles:
        return []
    corpus = [user_interests] + [
        a.get("title", "") + " " + a.get("abstract", a.get("content", a.get("title", "")))
        for a in articles
    ]
    vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True)
    tfidf_matrix = vectorizer.fit_transform(corpus)
    scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    for i, article in enumerate(articles):
        article["relevance_score"] = round(float(scores[i]), 4)
    return sorted(articles, key=lambda x: x["relevance_score"], reverse=True)


@mcp.tool()
def rank_with_keywords(keywords: list[str], articles: list[dict], top_k: int = 5) -> list[dict]:
    """Rank articles using keyword matching + TF-IDF hybrid scoring (70/30 blend)."""
    if not articles:
        return []
    interest_text = " ".join(keywords)
    corpus = [interest_text] + [
        a.get("title", "") + " " + a.get("abstract", a.get("content", ""))
        for a in articles
    ]
    vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True)
    tfidf_matrix = vectorizer.fit_transform(corpus)
    tfidf_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    for i, article in enumerate(articles):
        text = (article.get("title", "") + " " + article.get("abstract", "") + " " + article.get("content", "")).lower()
        keyword_hits = sum(1 for kw in keywords if kw.lower() in text)
        keyword_bonus = keyword_hits / max(len(keywords), 1) * 0.3
        combined = round(float(tfidf_scores[i]) * 0.7 + keyword_bonus, 4)
        article["relevance_score"] = combined
        article["keyword_matches"] = keyword_hits
    ranked = sorted(articles, key=lambda x: x["relevance_score"], reverse=True)
    return ranked[:top_k]


if __name__ == "__main__":
    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=int(os.environ.get("MCP_RANKER_PORT", "8000")),
        path="/mcp",
    )
