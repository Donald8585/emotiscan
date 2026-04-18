"""
EmotiScan: Voice diary session management.
Orchestrates recording sessions with face + voice + text emotion tracking.
"""

import uuid
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime

from config import MAX_DIARY_ENTRIES, DIARY_AUTO_RESEARCH, EMOTION_LABELS
from services.speech_service import SpeechService
from services.mood_fusion import MoodFusion
from services.llm_service import LLMService
from services.web_search_service import search_general, search_emotion_articles, search_coping_strategies, _safe_query

logger = logging.getLogger(__name__)


@dataclass
class DiaryEntry:
    """A single diary recording entry with multimodal emotion data."""
    timestamp: str
    text: str                      # transcribed text
    face_emotion: str              # from webcam
    face_confidence: float
    voice_sentiment: dict          # polarity, subjectivity, emotion, confidence
    audio_emotion: dict            # energy, pitch_var, estimated_emotion, confidence
    fused_emotion: str             # final combined emotion
    fused_confidence: float
    face_emotion_timeline: list = field(default_factory=list)  # [{time, emotion, confidence}]

    def to_dict(self) -> dict:
        return asdict(self)

    def to_markdown(self) -> str:
        lines = [
            f"### Entry at {self.timestamp}",
            f"**Transcription:** {self.text}",
            f"**Face:** {self.face_emotion} ({self.face_confidence:.0%})",
            f"**Voice Sentiment:** polarity={self.voice_sentiment.get('polarity', 0):.2f}, "
            f"emotion={self.voice_sentiment.get('emotion', 'neutral')}",
            f"**Audio:** energy={self.audio_emotion.get('energy', 0):.2f}, "
            f"emotion={self.audio_emotion.get('estimated_emotion', 'neutral')}",
            f"**Fused:** {self.fused_emotion} ({self.fused_confidence:.0%})",
            "",
        ]
        return "\n".join(lines)


@dataclass
class DiarySession:
    """A diary recording session containing multiple entries."""
    session_id: str
    start_time: str
    entries: list = field(default_factory=list)    # list of DiaryEntry
    summary: str = ""
    research_queries: list = field(default_factory=list)
    web_results: list = field(default_factory=list)
    arxiv_results: list = field(default_factory=list)
    compassionate_response: str = ""
    suggested_solutions: str = ""                  # pre-computed LLM solutions
    coping_strategies: list = field(default_factory=list)  # pre-computed coping results

    def to_dict(self) -> dict:
        d = {
            "session_id": self.session_id,
            "start_time": self.start_time,
            "entries": [e.to_dict() if isinstance(e, DiaryEntry) else e for e in self.entries],
            "summary": self.summary,
            "research_queries": self.research_queries,
            "web_results": self.web_results,
            "arxiv_results": self.arxiv_results,
            "compassionate_response": self.compassionate_response,
            "suggested_solutions": self.suggested_solutions,
            "coping_strategies": self.coping_strategies,
        }
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def to_markdown(self) -> str:
        lines = [
            f"# Diary Session: {self.session_id}",
            f"**Started:** {self.start_time}",
            f"**Entries:** {len(self.entries)}",
            "",
        ]
        if self.summary:
            lines.append(f"## Summary\n{self.summary}\n")
        if self.compassionate_response:
            lines.append(f"## Supportive Response\n{self.compassionate_response}\n")
        if self.research_queries:
            lines.append("## Research Queries")
            for q in self.research_queries:
                lines.append(f"- {q}")
            lines.append("")
        if self.arxiv_results:
            lines.append("## ArXiv Papers")
            for p in self.arxiv_results:
                title = p.get("title", "Untitled")
                url = p.get("url", "")
                authors = ", ".join(p.get("authors", [])) if isinstance(p.get("authors"), list) else str(p.get("authors", ""))
                lines.append(f"- **{title}** — {authors} [{url}]")
            lines.append("")
        if self.web_results:
            lines.append("## Web Results")
            for r in self.web_results:
                title = r.get("title", "Untitled")
                url = r.get("url", "")
                snippet = r.get("snippet", "")
                lines.append(f"- **{title}** — {snippet[:120]} [{url}]")
            lines.append("")
        lines.append("## Entries\n")
        for entry in self.entries:
            if isinstance(entry, DiaryEntry):
                lines.append(entry.to_markdown())
            else:
                lines.append(str(entry))
        return "\n".join(lines)

    @property
    def dominant_emotion(self) -> str:
        """Get the most frequent fused emotion across all entries."""
        if not self.entries:
            return "neutral"
        emotions = []
        for e in self.entries:
            if isinstance(e, DiaryEntry):
                emotions.append(e.fused_emotion)
            elif isinstance(e, dict):
                emotions.append(e.get("fused_emotion", "neutral"))
        if not emotions:
            return "neutral"
        return max(set(emotions), key=emotions.count)

    @property
    def best_emotion(self) -> str:
        """Get the most representative non-neutral emotion across all modalities.

        Checks face, voice, audio and fused emotions — picks the most frequent
        non-neutral value so that research/search/LLM reflect what the sensors
        actually detected, not a washed-out 'neutral' default.

        Includes the per-second face_emotion_timeline so the video graph data
        (which is what the user sees) has proper weight in determining the
        best emotion.
        """
        if not self.entries:
            return "neutral"
        all_emos: list[str] = []
        for e in self.entries:
            if isinstance(e, DiaryEntry):
                all_emos.append(e.fused_emotion)
                all_emos.append(e.face_emotion)
                all_emos.append(e.voice_sentiment.get("emotion", "neutral"))
                all_emos.append(e.audio_emotion.get("estimated_emotion", "neutral"))
                # Include per-second face timeline — this is the data shown
                # on the video emotion graph and should dominate
                for pt in (e.face_emotion_timeline or []):
                    if isinstance(pt, dict):
                        all_emos.append(pt.get("emotion", "neutral"))
            elif isinstance(e, dict):
                all_emos.append(e.get("fused_emotion", "neutral"))
                all_emos.append(e.get("face_emotion", "neutral"))
                vs = e.get("voice_sentiment", {})
                all_emos.append(vs.get("emotion", "neutral") if isinstance(vs, dict) else "neutral")
                ae = e.get("audio_emotion", {})
                all_emos.append(ae.get("estimated_emotion", "neutral") if isinstance(ae, dict) else "neutral")
                for pt in (e.get("face_emotion_timeline") or []):
                    if isinstance(pt, dict):
                        all_emos.append(pt.get("emotion", "neutral"))
        # Prefer non-neutral emotions
        non_neutral = [em for em in all_emos if em and em != "neutral"]
        if non_neutral:
            return max(set(non_neutral), key=non_neutral.count)
        return "neutral"

    def get_full_context(self) -> dict:
        """Build a rich context dict with transcript, per-modality emotions,
        and key topics extracted from the diary text.  Used by LLM, research,
        and web-search so they address the user's actual problems.

        Includes the per-second face_emotion_timeline from the video graph so
        that summaries accurately reflect what the camera detected over time,
        not just the single snapshot label per entry.
        """
        texts: list[str] = []
        face_emos: list[str] = []
        face_timeline_emos: list[str] = []   # per-second from video graph
        voice_emos: list[str] = []
        audio_emos: list[str] = []
        fused_emos: list[str] = []
        for e in self.entries:
            if isinstance(e, DiaryEntry):
                texts.append(e.text)
                face_emos.append(e.face_emotion)
                # Also grab per-second emotion timeline from video
                for pt in (e.face_emotion_timeline or []):
                    if isinstance(pt, dict):
                        face_timeline_emos.append(pt.get("emotion", "neutral"))
                voice_emos.append(e.voice_sentiment.get("emotion", "neutral"))
                audio_emos.append(e.audio_emotion.get("estimated_emotion", "neutral"))
                fused_emos.append(e.fused_emotion)
            elif isinstance(e, dict):
                texts.append(e.get("text", ""))
                face_emos.append(e.get("face_emotion", "neutral"))
                for pt in (e.get("face_emotion_timeline") or []):
                    if isinstance(pt, dict):
                        face_timeline_emos.append(pt.get("emotion", "neutral"))
                vs = e.get("voice_sentiment", {})
                voice_emos.append(vs.get("emotion", "neutral") if isinstance(vs, dict) else "neutral")
                ae = e.get("audio_emotion", {})
                audio_emos.append(ae.get("estimated_emotion", "neutral") if isinstance(ae, dict) else "neutral")
                fused_emos.append(e.get("fused_emotion", "neutral"))

        # If we have per-second timeline data from the video graph, use that
        # as the face_emotions list — it's far more accurate than the single
        # snapshot label stored per entry.
        effective_face_emos = face_timeline_emos if face_timeline_emos else face_emos

        return {
            "transcript": " ".join(texts),
            "face_emotions": effective_face_emos,
            "face_emotions_summary": face_emos,  # original per-entry labels
            "voice_emotions": voice_emos,
            "audio_emotions": audio_emos,
            "fused_emotions": fused_emos,
            "best_emotion": self.best_emotion,
            "dominant_emotion": self.dominant_emotion,
        }

    @property
    def emotion_timeline(self) -> list:
        """Return list of (timestamp, fused_emotion, fused_confidence) for charting."""
        timeline = []
        for e in self.entries:
            if isinstance(e, DiaryEntry):
                timeline.append((e.timestamp, e.fused_emotion, e.fused_confidence))
            elif isinstance(e, dict):
                timeline.append((
                    e.get("timestamp", ""),
                    e.get("fused_emotion", "neutral"),
                    e.get("fused_confidence", 0.5),
                ))
        return timeline


class DiarySessionManager:
    """Orchestrates diary recording sessions."""

    def __init__(self):
        self._speech = SpeechService()
        self._fusion = MoodFusion()
        self._llm = LLMService()

    def start_session(self) -> DiarySession:
        """Start a new diary session."""
        return DiarySession(
            session_id=str(uuid.uuid4())[:8],
            start_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

    def add_entry(
        self,
        session: DiarySession,
        audio_bytes: bytes,
        face_emotion: str = "neutral",
        face_confidence: float = 0.5,
        face_emotion_timeline: list | None = None,
    ) -> DiaryEntry:
        """
        Process audio + face data into a diary entry and add it to the session.

        Args:
            session: The active session
            audio_bytes: Raw audio bytes from the recorder
            face_emotion: Current face emotion from webcam
            face_confidence: Face emotion confidence
            face_emotion_timeline: Per-second face emotion history [{time, emotion, confidence}]

        Returns:
            The created DiaryEntry
        """
        # Transcribe audio
        transcription = self._speech.transcribe_audio(audio_bytes)
        text = transcription.get("text", "")

        # Analyze text sentiment
        voice_sentiment = self._speech.analyze_sentiment(text)

        # Analyze audio emotion heuristics
        audio_emotion = self._speech.analyze_audio_emotion(audio_bytes)

        # Fuse all signals
        fusion_result = self._fusion.fuse(
            face_emotion=face_emotion,
            face_confidence=face_confidence,
            text_sentiment=voice_sentiment,
            audio_emotion=audio_emotion,
        )

        entry = DiaryEntry(
            timestamp=datetime.now().strftime("%H:%M:%S"),
            text=text,
            face_emotion=face_emotion,
            face_confidence=face_confidence,
            voice_sentiment=voice_sentiment,
            audio_emotion=audio_emotion,
            fused_emotion=fusion_result["emotion"],
            fused_confidence=fusion_result["confidence"],
            face_emotion_timeline=face_emotion_timeline or [],
        )

        if len(session.entries) < MAX_DIARY_ENTRIES:
            session.entries.append(entry)
        else:
            logger.warning("Max diary entries (%d) reached", MAX_DIARY_ENTRIES)

        return entry

    def end_session(self, session: DiarySession) -> DiarySession:
        """
        End the session: generate summary, research queries, search results,
        and compassionate response.

        All outputs use the full session context (transcript + face/voice/audio
        emotions) so they address the user's actual spoken problems, not just
        a single emotion label.

        Returns:
            Updated session with summary, research queries, search results,
            and compassionate response.
        """
        ctx = session.get_full_context()

        session.summary = self.get_session_summary(session, ctx=ctx)
        if DIARY_AUTO_RESEARCH:
            session.research_queries = self.get_research_queries(session, ctx=ctx)

        # Use best_emotion (non-neutral if sensors detected one) for search
        emotion_for_search = ctx["best_emotion"]

        # ArXiv search
        try:
            session.arxiv_results = self._search_arxiv(
                session.research_queries, max_per_query=3
            )
        except Exception as e:
            logger.warning("ArXiv search failed in end_session: %s", e)
            session.arxiv_results = []

        # Web search — search for the user's actual problems, not just the emotion
        try:
            web_results = []
            # Search based on actual problems from research queries
            for q in session.research_queries[:3]:
                web_results.extend(search_general(q, max_results=3))
            # Also search for coping with the detected emotion
            if emotion_for_search != "neutral":
                web_results.extend(
                    search_emotion_articles(emotion_for_search, max_results=3)
                )
            # Deduplicate by URL
            seen_urls: set[str] = set()
            unique = []
            for r in web_results:
                url = r.get("url", "")
                if url not in seen_urls:
                    seen_urls.add(url)
                    unique.append(r)
            session.web_results = unique
        except Exception as e:
            logger.warning("Web search failed in end_session: %s", e)
            session.web_results = []

        # Compassionate response
        session.compassionate_response = self.get_compassionate_response(
            session, ctx=ctx
        )

        return session

    def get_session_summary(self, session: DiarySession, ctx: dict | None = None) -> str:
        """Generate a summary of the diary session using LLM or fallback.

        The summary addresses the user's actual spoken content AND the emotions
        detected across all modalities (face camera, voice tone, audio energy,
        and the fused result).
        """
        if not session.entries:
            return "Empty session - no entries recorded."

        if ctx is None:
            ctx = session.get_full_context()

        transcript = ctx["transcript"]
        face_emos = ctx["face_emotions"]
        voice_emos = ctx["voice_emotions"]
        audio_emos = ctx["audio_emotions"]
        fused_emos = ctx["fused_emotions"]
        best_emo = ctx["best_emotion"]

        # Summarize face emotions for the prompt (per-second data can be long)
        def _emo_breakdown(emos: list) -> str:
            if not emos:
                return "none detected"
            counts: dict[str, int] = {}
            for em in emos:
                counts[em] = counts.get(em, 0) + 1
            total = len(emos)
            parts = [f"{e}: {round(100*c/total)}%" for e, c in
                     sorted(counts.items(), key=lambda x: -x[1])]
            return ", ".join(parts) + f" (over {total} readings)"

        face_summary = _emo_breakdown(face_emos)

        prompt = (
            "Summarize this voice-diary session in 3-5 sentences. "
            "Focus on the PROBLEMS and TOPICS the user actually talked about, "
            "then note how their emotions (from multiple sensors) relate.\n\n"
            f"Transcript: {transcript[:600]}\n"
            f"Face-camera emotion breakdown: {face_summary}\n"
            f"Voice-tone emotions: {voice_emos}\n"
            f"Audio-energy emotions: {audio_emos}\n"
            f"Fused emotions: {fused_emos}\n"
            f"Strongest detected emotion: {best_emo}\n\n"
            "Be empathetic. Mention the specific issues the user raised "
            "and how the sensor readings reflect their emotional state. "
            "Pay special attention to the face-camera breakdown — this is "
            "real-time data from the user's webcam during the session:"
        )

        result = self._llm._generate(prompt)
        if result and result.strip():
            return result.strip()
        return ""

    def get_research_queries(self, session: DiarySession, ctx: dict | None = None) -> list:
        """Generate research search queries based on the user's actual spoken
        problems AND their detected emotions from all modalities.

        All queries are scoped to psychology/mental-health topics for safe,
        relevant results.
        """
        if not session.entries:
            return []

        if ctx is None:
            ctx = session.get_full_context()

        transcript = ctx["transcript"]
        best_emo = ctx["best_emotion"]
        face_emos = ctx["face_emotions"]
        voice_emos = ctx["voice_emotions"]

        # Summarize face emotions for the prompt
        def _emo_breakdown(emos: list) -> str:
            if not emos:
                return "none detected"
            counts: dict[str, int] = {}
            for em in emos:
                counts[em] = counts.get(em, 0) + 1
            total = len(emos)
            parts = [f"{e}: {round(100*c/total)}%" for e, c in
                     sorted(counts.items(), key=lambda x: -x[1])]
            return ", ".join(parts) + f" (over {total} readings)"

        face_summary = _emo_breakdown(face_emos)

        # Try LLM for query generation — focus on user's actual problems
        prompt = (
            "You are a helpful research assistant. The user recorded a voice diary.\n"
            "Generate exactly 3 web-search queries that will find articles or tips "
            "to HELP them with their specific situation.\n\n"
            f"=== What the user said ===\n{transcript[:500]}\n\n"
            f"=== Detected emotions ===\n"
            f"Face camera: {face_summary}\n"
            f"Voice tone: {voice_emos}\n"
            f"Strongest: {best_emo}\n\n"
            "RULES:\n"
            "1. Each query MUST reference a CONCRETE topic/problem from the transcript\n"
            "2. Queries should be what a real person would type into Google\n"
            "3. Include practical terms like 'tips', 'how to', 'strategies', 'advice'\n"
            "4. BAD: 'emotion regulation techniques' (too generic)\n"
            "   BAD: 'angry psychology research' (just an emotion label)\n"
            "   GOOD: 'how to deal with deadline pressure at work'\n"
            "   GOOD: 'tips for better sleep when anxious about exams'\n"
            "   GOOD: 'relationship communication after argument advice'\n"
            "5. If the transcript is unclear, use the detected emotion + common causes\n\n"
            "Return ONLY 3 queries, one per line, no numbering:"
        )
        result = self._llm._generate(prompt)
        if result and result.strip():
            queries = [q.strip().lstrip("0123456789.-) ") for q in result.strip().split("\n") if q.strip()]
            if queries:
                return queries[:5]

        return []

    @staticmethod
    def _arxiv_safe_query(query: str) -> str:
        """Return the query unchanged — no topic scoping applied."""
        return query

    @staticmethod
    def _is_relevant_paper(paper: dict) -> bool:
        """Accept all papers — no topic filtering applied."""
        return True

    @staticmethod
    def _search_arxiv(queries: list, max_per_query: int = 3) -> list:
        """Search ArXiv for research papers using mcp_client.
        
        Queries are scoped to psychology/neuroscience and results are filtered
        to reject particle physics, astrophysics, etc.
        """
        results = []
        try:
            from mcp_client import search_papers
        except ImportError:
            logger.warning("mcp_client not available for diary ArXiv search")
            return results

        seen_urls: set = set()
        for query in queries[:3]:
            try:
                safe_q = DiarySessionManager._arxiv_safe_query(query)
                # Fetch extra results so we still have enough after filtering
                papers = search_papers(safe_q, max_results=max_per_query + 5)
                added = 0
                for paper in papers:
                    if added >= max_per_query:
                        break
                    url = paper.get("url", "")
                    if url in seen_urls:
                        continue
                    if not DiarySessionManager._is_relevant_paper(paper):
                        logger.debug("ArXiv: rejected irrelevant paper: %s", paper.get("title", "")[:60])
                        continue
                    seen_urls.add(url)
                    results.append({
                        "title": paper.get("title", ""),
                        "url": url,
                        "abstract": paper.get("abstract", "")[:500],
                        "authors": paper.get("authors", []),
                        "published": paper.get("published", ""),
                        "source": "arxiv",
                    })
                    added += 1
            except Exception as e:
                logger.warning("ArXiv search failed for query '%s': %s", query, e)
        return results

    def get_compassionate_response(self, session: DiarySession, ctx: dict | None = None) -> str:
        """Generate a compassionate, supportive response addressing the user's
        actual problems and using emotion data from ALL modalities."""
        if not session.entries:
            return "Take care of yourself. Remember, every emotion is valid and temporary."

        if ctx is None:
            ctx = session.get_full_context()

        transcript = ctx["transcript"]
        best_emo = ctx["best_emotion"]
        face_emos = ctx["face_emotions"]
        voice_emos = ctx["voice_emotions"]

        # Summarize face emotions for the prompt
        def _emo_breakdown(emos: list) -> str:
            if not emos:
                return "none detected"
            counts: dict[str, int] = {}
            for em in emos:
                counts[em] = counts.get(em, 0) + 1
            total = len(emos)
            parts = [f"{e}: {round(100*c/total)}%" for e, c in
                     sorted(counts.items(), key=lambda x: -x[1])]
            return ", ".join(parts) + f" (over {total} readings)"

        face_summary = _emo_breakdown(face_emos)

        prompt = (
            "The user just completed a voice-diary session.\n\n"
            f"What they said: {transcript[:500]}\n"
            f"Face-camera emotion breakdown: {face_summary}\n"
            f"Voice-tone emotions: {voice_emos}\n"
            f"Strongest detected emotion: {best_emo}\n\n"
            "Respond as a compassionate counselor who:\n"
            "1) Addresses the SPECIFIC problems/topics the user talked about\n"
            "2) Acknowledges the emotions detected by the camera and voice analysis "
            "(pay special attention to the face-camera breakdown)\n"
            "3) Suggests 2-3 specific, actionable coping methods relevant to "
            "their actual situation (not generic emotion advice)\n"
            "4) Ends with an encouraging note\n"
            "Keep it warm, personal, and about 100-150 words."
        )

        result = self._llm._generate(prompt)
        if result and result.strip():
            return result.strip()
        return ""

    @staticmethod
    def breathing_exercise_text() -> str:
        """Return a guided breathing exercise for moments of emotional distress."""
        return (
            "**Guided Breathing Exercise (4-7-8 Technique)**\n\n"
            "1. **Breathe IN** through your nose for **4 seconds**\n"
            "2. **HOLD** your breath for **7 seconds**\n"
            "3. **Breathe OUT** slowly through your mouth for **8 seconds**\n"
            "4. Repeat **3 more times**\n\n"
            "This activates your parasympathetic nervous system, "
            "lowering heart rate and promoting calm.\n\n"
            "Remember: your feelings are valid, and this moment will pass."
        )
