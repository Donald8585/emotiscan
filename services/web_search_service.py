"""
EmotiScan: DuckDuckGo web search with safe-search enforcement and content filtering.
Zero-config, no API key needed.
"""

import re
import time
import logging
from config import WEB_SEARCH_MAX_RESULTS, WEB_SEARCH_CACHE_TTL

logger = logging.getLogger(__name__)

# ── Content Safety ─────────────────────────────────────────────────
# Blocked URL domains (adult/gambling/dictionary/translation/unrelated content)
BLOCKED_DOMAINS = frozenset({
    "xhamster.com", "pornhub.com", "xnxx.com", "xvideos.com",
    "redtube.com", "youporn.com", "tube8.com", "spankbang.com",
    "chaturbate.com", "onlyfans.com", "livejasmin.com",
    "bet365.com", "draftkings.com", "fanduel.com",
    "baidu.com", "zhidao.baidu.com",
    # Dictionary / translation sites — return word definitions, not coping advice
    "dictionary.cambridge.org", "cambridge.org/dictionary",
    "iciba.com", "dict.cn", "youdao.com", "dict.youdao.com",
    "merriam-webster.com", "dictionary.com", "thefreedictionary.com",
    "collinsdictionary.com", "ldoceonline.com", "macmillandictionary.com",
    "oxfordlearnersdictionaries.com", "wordreference.com",
    "translate.google.com", "deepl.com/translator",
    "global.bing.com/dict", "bing.com/dict",
    # Wikipedia — too generic, not actionable advice
    "wikipedia.org", "wiktionary.org",
    # Generic Q&A / forums that rarely have evidence-based content
    "quora.com", "answers.yahoo.com",
})

# Blocked keywords in titles/URLs/snippets (case-insensitive match)
BLOCKED_KEYWORDS = frozenset({
    "porn", "xxx", "sex", "nude", "naked", "hentai", "erotic",
    "adult video", "adult film", "bokep", "camgirl", "onlyfans",
    "gambling", "casino", "betting odds",
    # Dictionary / translation page markers
    "翻譯", "翻译", "词典", "詞典", "辞典", "dictionary search",
    "word definition", "pronunciation", "音标", "读音", "用法",
    "在线词典", "/dict/search",
})

# Required relevance keywords — at least one must appear in the result
# for it to pass the relevance filter (case-insensitive)
#
# NOTE: Removed overly generic words that match dictionary/wiki pages:
#   - Single emotion labels ("anger", "fear", "disgust") — every dictionary page has these
#   - "strategy", "exercise", "technique" — too generic on their own
#   - "research", "study", "journal" — match academic noise
# Kept multi-word or domain-specific terms that strongly signal mental-health content.
RELEVANCE_KEYWORDS = frozenset({
    # ── Core mental health / psychology terms ────────────────────────
    "mental health", "mental wellness", "mental illness", "mental disorder",
    "psychology", "psychological", "psychotherapy", "psychiatric", "psychiatry",
    "counseling", "counselling", "therapist", "therapy",
    "cognitive behavioral", "cbt", "dbt", "dialectical behavior",
    "acceptance and commitment", "act therapy", "emdr", "psychoanalysis",
    "group therapy", "talk therapy", "art therapy", "music therapy",
    "somatic therapy", "neurofeedback",
    # ── Coping & regulation ──────────────────────────────────────────
    "coping", "coping strategy", "coping mechanism", "coping skill",
    "emotion regulation", "emotional regulation", "affect regulation",
    "stress management", "anger management", "frustration management",
    "behavioral activation", "exposure therapy",
    "cognitive reappraisal", "cognitive restructuring",
    "progressive muscle", "body scan", "grounding technique",
    "breathing exercise", "deep breathing", "relaxation technique",
    "mindfulness", "meditation", "yoga therapy", "nature therapy",
    "journaling", "expressive writing", "gratitude", "gratitude practice",
    "how to cope", "how to deal", "how to manage", "how to overcome",
    "tips for", "strategies for", "ways to", "guide to",
    "evidence-based", "evidence based",
    # ── Wellbeing & self-care ────────────────────────────────────────
    "well-being", "wellbeing", "wellness", "self-care", "self care",
    "self-compassion", "self compassion", "self-esteem", "self-worth",
    "self-love", "self-acceptance", "self-efficacy",
    "resilience", "resilience building", "grit", "optimism", "hope",
    "positive psychology", "emotional health", "emotional wellbeing",
    "emotional intelligence", "emotion awareness", "emotional literacy",
    "psychological wellbeing", "psychological health", "psychological safety",
    "sleep hygiene", "sleep health", "digital detox",
    "social support", "peer support", "community support",
    # ── Positive emotions ────────────────────────────────────────────
    "happiness", "happy", "happiness research", "science of happiness",
    "joy", "joyfulness", "positive emotion", "positive emotions",
    "contentment", "satisfaction", "life satisfaction",
    "excitement", "elation", "enthusiasm", "passion",
    "serenity", "tranquility", "peace of mind", "inner peace",
    "hope", "hopeful", "hopefulness",
    "love", "compassion", "kindness", "affection",
    "gratitude", "awe", "wonder", "inspiration",
    "confidence", "empowerment", "flourishing", "thriving",
    # ── Negative emotions ────────────────────────────────────────────
    "sadness", "grief", "grief counseling", "grief therapy", "bereavement",
    "sorrow", "mourning", "loss", "heartbreak",
    "anger", "rage", "frustration", "irritability",
    "fear", "fear response", "phobia", "panic attack",
    "disgust", "shame", "guilt", "embarrassment",
    "loneliness", "isolation", "social isolation",
    "jealousy", "envy", "resentment",
    "disappointment", "regret", "remorse",
    "overwhelmed", "helpless", "hopelessness", "despair",
    "worry", "nervousness", "apprehension", "dread",
    "emotional pain", "emotional suffering", "emotional distress",
    "hurt", "heartbroken",
    # ── Clinical / disorder terms ────────────────────────────────────
    "anxiety", "anxiety disorder", "social anxiety", "generalized anxiety",
    "depression", "clinical depression", "major depressive",
    "burnout", "exhaustion", "compassion fatigue",
    "trauma", "trauma therapy", "trauma recovery", "ptsd", "post-traumatic",
    "bipolar", "bipolar disorder", "mood disorder",
    "ocd", "obsessive compulsive",
    "eating disorder", "anorexia", "bulimia", "binge eating",
    "panic disorder", "agoraphobia",
    "adhd", "attention deficit",
    "autism", "autistic",
    "borderline personality", "bpd",
    "schizophrenia", "psychosis",
    "emotional abuse", "emotional neglect",
    "crisis intervention", "mental health crisis", "suicidal ideation",
    # ── Neuroscience / research ──────────────────────────────────────
    "neuroscience", "affective neuroscience", "neuroscience of emotion",
    "cognitive neuroscience", "neuroplasticity",
    "mood", "affect", "affective disorder",
    # ── Trusted domains ──────────────────────────────────────────────
    "apa.org", "nih.gov", "psychologytoday.com",
    "sciencedirect.com", "pubmed", "ncbi",
    "frontiersin.org", "springer.com",
    "helpguide.org", "mind.org.uk", "nami.org",
    "betterhelp.com", "talkspace.com",
    "verywellmind.com", "healthline.com/health",
})


def _is_blocked(result: dict) -> bool:
    """Check if a search result should be blocked (unsafe/irrelevant content)."""
    title = (result.get("title", "") or "").lower()
    url = (result.get("url", "") or "").lower()
    snippet = (result.get("snippet", "") or "").lower()
    combined = f"{title} {url} {snippet}"

    # Check blocked domains
    for domain in BLOCKED_DOMAINS:
        if domain in url:
            return True

    # Check blocked keywords
    for kw in BLOCKED_KEYWORDS:
        if kw in combined:
            return True

    return False


def _is_relevant(result: dict) -> bool:
    """Check if a search result is relevant to emotion/mental-health topics."""
    title = (result.get("title", "") or "").lower()
    url = (result.get("url", "") or "").lower()
    snippet = (result.get("snippet", "") or "").lower()
    combined = f"{title} {url} {snippet}"

    for kw in RELEVANCE_KEYWORDS:
        if kw in combined:
            return True
    return False


def _filter_results(results: list, enforce_relevance: bool = True) -> list:
    """Filter out blocked content and optionally enforce topic relevance."""
    filtered = []
    for r in results:
        if _is_blocked(r):
            logger.debug("Blocked result: %s", r.get("title", "")[:60])
            continue
        if enforce_relevance and not _is_relevant(r):
            logger.debug("Irrelevant result dropped: %s", r.get("title", "")[:60])
            continue
        filtered.append(r)
    return filtered


class _SearchCache:
    """Simple TTL cache for search results."""

    def __init__(self, ttl: int = WEB_SEARCH_CACHE_TTL):
        self._cache: dict = {}
        self._ttl = ttl

    def get(self, key: str):
        entry = self._cache.get(key)
        if entry is None:
            return None
        if time.time() - entry["ts"] > self._ttl:
            del self._cache[key]
            return None
        return entry["data"]

    def set(self, key: str, data):
        self._cache[key] = {"data": data, "ts": time.time()}

    def clear(self):
        self._cache.clear()


_cache = _SearchCache()


def _safe_query(query: str) -> str:
    """Return the query unchanged — no topic scoping applied."""
    return query


def search_emotion_articles(emotion: str, max_results: int = WEB_SEARCH_MAX_RESULTS) -> list:
    """Search for articles about an emotion (safe-search enforced)."""
    query = f"how to deal with {emotion} emotion mental health tips"
    return search_general(query, max_results=max_results, enforce_relevance=True)


def search_general(
    query: str,
    max_results: int = WEB_SEARCH_MAX_RESULTS,
    enforce_relevance: bool = False,
) -> list:
    """
    Search DuckDuckGo for articles with safe-search ON and content filtering.

    Args:
        query: Search query string
        max_results: Maximum results to return
        enforce_relevance: If True, filter out results not related to emotion/psychology

    Returns:
        list of {title, url, snippet}
    """
    safe_q = _safe_query(query) if enforce_relevance else query
    cache_key = f"{safe_q}:{max_results}:{enforce_relevance}"
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            # safesearch="on" is the strictest mode (off, moderate, on)
            raw = list(ddgs.text(safe_q, region="en-us", max_results=max_results + 15, safesearch="on"))
        if raw:
            results = [
                {
                    "title": r.get("title", ""),
                    "url": r.get("href", r.get("link", "")),
                    "snippet": r.get("body", r.get("snippet", "")),
                }
                for r in raw
            ]
            # Apply content safety + relevance filters
            results = _filter_results(results, enforce_relevance=enforce_relevance)
            results = results[:max_results]
            if results:
                _cache.set(cache_key, results)
                return results
        logger.info("DuckDuckGo returned no usable results for '%s'", safe_q)
        return []
    except Exception:
        return []


def search_coping_strategies(emotion: str, max_results: int = WEB_SEARCH_MAX_RESULTS,
                             context: str = "") -> list:
    """Search for evidence-based coping strategies.

    Args:
        emotion: The detected dominant emotion (e.g. 'disgust', 'sad').
        max_results: Max results to return.
        context: Optional transcript/summary text describing the user's actual
                 situation.  When provided, the search query is built from
                 concrete problem keywords instead of the generic emotion label.
    """
    if context and len(context.strip()) > 20:
        # Extract key phrases from the context for a targeted query
        # Take the first ~120 chars of context as the core problem
        short_ctx = context.strip()[:120].rsplit(" ", 1)[0]
        query = f"{short_ctx} coping strategies mental health advice"
    else:
        query = f"how to cope with {emotion} feelings evidence-based strategies"
    cache_key = f"coping:{query}:{max_results}"
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            raw = list(ddgs.text(query, region="en-us", max_results=max_results + 15, safesearch="on"))
        if raw:
            results = [
                {
                    "title": r.get("title", ""),
                    "url": r.get("href", r.get("link", "")),
                    "snippet": r.get("body", r.get("snippet", "")),
                }
                for r in raw
            ]
            # Block unsafe content but don't enforce strict relevance —
            # the query itself is already scoped (either by transcript context
            # or by the emotion+coping keywords)
            results = _filter_results(results, enforce_relevance=False)
            results = results[:max_results]
            if results:
                _cache.set(cache_key, results)
                return results
    except Exception:
        pass

    return []


def clear_cache():
    """Clear the search cache."""
    _cache.clear()
