"""
EmotiScan: Persistent memory store using SQLite.

Stores diary sessions (video emotion timelines, audio transcriptions, chat
history, research results) across server restarts.  Also stores the general
chat history and per-session emotion detection history.

All data lives in a single SQLite file under DATA_DIR (default: ./emotiscan_data).
Audio/video blobs are saved as files on disk (not in SQLite) to keep the DB lean.
"""

import json
import os
import sqlite3
import logging
import threading
import wave
import struct
from datetime import datetime
from pathlib import Path
from typing import Optional

from config import DATA_DIR, DB_FILE

logger = logging.getLogger(__name__)

# ── Schema version (bump when tables change) ─────────────────────
_SCHEMA_VERSION = 1


def _get_db(db_path: str | None = None) -> sqlite3.Connection:
    """Open a SQLite connection with WAL mode for concurrent reads."""
    path = db_path or DB_FILE
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str | None = None) -> sqlite3.Connection:
    """Create tables if they don't exist and return the connection."""
    conn = _get_db(db_path)
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS schema_version (
        version INTEGER PRIMARY KEY
    );

    -- Diary sessions
    CREATE TABLE IF NOT EXISTS diary_sessions (
        session_id   TEXT PRIMARY KEY,
        start_time   TEXT NOT NULL,
        end_time     TEXT,
        summary      TEXT DEFAULT '',
        compassionate_response TEXT DEFAULT '',
        research_queries TEXT DEFAULT '[]',   -- JSON array
        web_results  TEXT DEFAULT '[]',        -- JSON array
        arxiv_results TEXT DEFAULT '[]',       -- JSON array
        created_at   TEXT DEFAULT (datetime('now'))
    );

    -- Diary entries (belong to a session)
    CREATE TABLE IF NOT EXISTS diary_entries (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id      TEXT NOT NULL REFERENCES diary_sessions(session_id) ON DELETE CASCADE,
        entry_index     INTEGER NOT NULL,
        timestamp       TEXT NOT NULL,
        text            TEXT DEFAULT '',
        face_emotion    TEXT DEFAULT 'neutral',
        face_confidence REAL DEFAULT 0.0,
        voice_sentiment TEXT DEFAULT '{}',   -- JSON
        audio_emotion   TEXT DEFAULT '{}',   -- JSON
        fused_emotion   TEXT DEFAULT 'neutral',
        fused_confidence REAL DEFAULT 0.0,
        face_emotion_timeline TEXT DEFAULT '[]',  -- JSON array
        audio_file_path TEXT DEFAULT ''
    );

    -- Chat messages (post-diary supportive chat + general chat)
    CREATE TABLE IF NOT EXISTS chat_messages (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id  TEXT DEFAULT '',          -- '' for general chat
        role        TEXT NOT NULL,            -- 'user' or 'assistant'
        content     TEXT NOT NULL,
        created_at  TEXT DEFAULT (datetime('now'))
    );

    -- Emotion detection history (from the real-time detection tab)
    CREATE TABLE IF NOT EXISTS emotion_history (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        source      TEXT DEFAULT 'detection',  -- 'detection' or 'diary'
        session_id  TEXT DEFAULT '',
        emotion     TEXT NOT NULL,
        confidence  REAL DEFAULT 0.0,
        recorded_at TEXT DEFAULT (datetime('now'))
    );
    """)
    # Set schema version
    existing = conn.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
    if not existing:
        conn.execute("INSERT INTO schema_version (version) VALUES (?)", (_SCHEMA_VERSION,))
    conn.commit()
    return conn


class MemoryStore:
    """Thread-safe persistent storage for EmotiScan."""

    def __init__(self, db_path: str | None = None):
        self._db_path = db_path or DB_FILE
        self._lock = threading.Lock()
        self._conn = init_db(self._db_path)
        # Ensure audio dir exists
        self._audio_dir = os.path.join(os.path.dirname(self._db_path) or DATA_DIR, "audio")
        os.makedirs(self._audio_dir, exist_ok=True)
        logger.info("MemoryStore initialized at %s", self._db_path)

    # ── Diary sessions ────────────────────────────────────────────

    def save_session(self, session_dict: dict) -> None:
        """Persist a completed diary session (dict form) to the database."""
        sid = session_dict.get("session_id", "")
        with self._lock:
            self._conn.execute("""
                INSERT OR REPLACE INTO diary_sessions
                (session_id, start_time, end_time, summary, compassionate_response,
                 research_queries, web_results, arxiv_results)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                sid,
                session_dict.get("start_time", ""),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                session_dict.get("summary", ""),
                session_dict.get("compassionate_response", ""),
                json.dumps(session_dict.get("research_queries", [])),
                json.dumps(session_dict.get("web_results", [])),
                json.dumps(session_dict.get("arxiv_results", [])),
            ))
            # Delete old entries for this session (replace scenario)
            self._conn.execute("DELETE FROM diary_entries WHERE session_id = ?", (sid,))
            for idx, entry in enumerate(session_dict.get("entries", [])):
                if not isinstance(entry, dict):
                    continue
                self._conn.execute("""
                    INSERT INTO diary_entries
                    (session_id, entry_index, timestamp, text, face_emotion,
                     face_confidence, voice_sentiment, audio_emotion,
                     fused_emotion, fused_confidence, face_emotion_timeline, audio_file_path)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    sid, idx,
                    entry.get("timestamp", ""),
                    entry.get("text", ""),
                    entry.get("face_emotion", "neutral"),
                    entry.get("face_confidence", 0.0),
                    json.dumps(entry.get("voice_sentiment", {})),
                    json.dumps(entry.get("audio_emotion", {})),
                    entry.get("fused_emotion", "neutral"),
                    entry.get("fused_confidence", 0.0),
                    json.dumps(entry.get("face_emotion_timeline", [])),
                    entry.get("audio_file_path", ""),
                ))
            self._conn.commit()
        logger.info("Saved diary session %s with %d entries",
                     sid, len(session_dict.get("entries", [])))

    def load_all_sessions(self) -> list[dict]:
        """Load all diary sessions (most recent first) as dicts."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM diary_sessions ORDER BY created_at DESC"
            ).fetchall()
        sessions = []
        for row in rows:
            sid = row["session_id"]
            entries = self._load_entries(sid)
            sessions.append({
                "session_id": sid,
                "start_time": row["start_time"],
                "end_time": row["end_time"],
                "summary": row["summary"],
                "compassionate_response": row["compassionate_response"],
                "research_queries": json.loads(row["research_queries"] or "[]"),
                "web_results": json.loads(row["web_results"] or "[]"),
                "arxiv_results": json.loads(row["arxiv_results"] or "[]"),
                "entries": entries,
            })
        return sessions

    def load_session(self, session_id: str) -> Optional[dict]:
        """Load a single diary session by ID."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM diary_sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        if not row:
            return None
        entries = self._load_entries(session_id)
        return {
            "session_id": row["session_id"],
            "start_time": row["start_time"],
            "end_time": row["end_time"],
            "summary": row["summary"],
            "compassionate_response": row["compassionate_response"],
            "research_queries": json.loads(row["research_queries"] or "[]"),
            "web_results": json.loads(row["web_results"] or "[]"),
            "arxiv_results": json.loads(row["arxiv_results"] or "[]"),
            "entries": entries,
        }

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and its entries. Returns True if it existed."""
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM diary_sessions WHERE session_id = ?", (session_id,)
            )
            self._conn.commit()
        # Also clean up audio files
        for f in Path(self._audio_dir).glob(f"{session_id}_*"):
            f.unlink(missing_ok=True)
        return cursor.rowcount > 0

    def _load_entries(self, session_id: str) -> list[dict]:
        """Load diary entries for a session."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM diary_entries WHERE session_id = ? ORDER BY entry_index",
                (session_id,)
            ).fetchall()
        entries = []
        for r in rows:
            entries.append({
                "timestamp": r["timestamp"],
                "text": r["text"],
                "face_emotion": r["face_emotion"],
                "face_confidence": r["face_confidence"],
                "voice_sentiment": json.loads(r["voice_sentiment"] or "{}"),
                "audio_emotion": json.loads(r["audio_emotion"] or "{}"),
                "fused_emotion": r["fused_emotion"],
                "fused_confidence": r["fused_confidence"],
                "face_emotion_timeline": json.loads(r["face_emotion_timeline"] or "[]"),
                "audio_file_path": r["audio_file_path"],
            })
        return entries

    def get_session_count(self) -> int:
        """Return the total number of saved sessions."""
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) as cnt FROM diary_sessions").fetchone()
        return row["cnt"] if row else 0

    # ── Audio file persistence ────────────────────────────────────

    def save_audio(self, session_id: str, entry_index: int,
                   audio_bytes: bytes, sample_rate: int = 16000) -> str:
        """Save audio bytes to a WAV file on disk. Returns the file path."""
        filename = f"{session_id}_{entry_index}.wav"
        filepath = os.path.join(self._audio_dir, filename)
        try:
            # If audio_bytes is raw PCM, wrap in WAV header
            if not audio_bytes[:4] == b'RIFF':
                with wave.open(filepath, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)  # 16-bit
                    wf.setframerate(sample_rate)
                    wf.writeframes(audio_bytes)
            else:
                with open(filepath, 'wb') as f:
                    f.write(audio_bytes)
        except Exception as e:
            logger.warning("Failed to save audio %s: %s", filepath, e)
            return ""
        return filepath

    def load_audio(self, filepath: str) -> Optional[bytes]:
        """Load audio bytes from a saved file."""
        if not filepath or not os.path.exists(filepath):
            return None
        with open(filepath, 'rb') as f:
            return f.read()

    # ── Chat messages ─────────────────────────────────────────────

    def save_chat_message(self, role: str, content: str,
                          session_id: str = "") -> None:
        """Save a single chat message."""
        with self._lock:
            self._conn.execute("""
                INSERT INTO chat_messages (session_id, role, content)
                VALUES (?, ?, ?)
            """, (session_id, role, content))
            self._conn.commit()

    def save_chat_history(self, messages: list[dict],
                          session_id: str = "") -> None:
        """Replace all chat messages for a session_id with the given list."""
        with self._lock:
            self._conn.execute(
                "DELETE FROM chat_messages WHERE session_id = ?", (session_id,)
            )
            for msg in messages:
                self._conn.execute("""
                    INSERT INTO chat_messages (session_id, role, content)
                    VALUES (?, ?, ?)
                """, (session_id, msg.get("role", "user"), msg.get("content", "")))
            self._conn.commit()

    def load_chat_history(self, session_id: str = "") -> list[dict]:
        """Load chat messages for a given session_id ('' = general chat)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT role, content, created_at FROM chat_messages "
                "WHERE session_id = ? ORDER BY id",
                (session_id,)
            ).fetchall()
        return [{"role": r["role"], "content": r["content"],
                 "created_at": r["created_at"]} for r in rows]

    def clear_chat_history(self, session_id: str = "") -> None:
        """Delete all chat messages for a session."""
        with self._lock:
            self._conn.execute(
                "DELETE FROM chat_messages WHERE session_id = ?", (session_id,)
            )
            self._conn.commit()

    # ── Emotion history ───────────────────────────────────────────

    def save_emotion_snapshot(self, emotion: str, confidence: float,
                              source: str = "detection",
                              session_id: str = "") -> None:
        """Save a single emotion detection event."""
        with self._lock:
            self._conn.execute("""
                INSERT INTO emotion_history (source, session_id, emotion, confidence)
                VALUES (?, ?, ?, ?)
            """, (source, session_id, emotion, confidence))
            self._conn.commit()

    def save_emotion_history(self, history: list[dict],
                             source: str = "detection",
                             session_id: str = "") -> None:
        """Bulk-save emotion history entries."""
        with self._lock:
            for h in history:
                self._conn.execute("""
                    INSERT INTO emotion_history
                    (source, session_id, emotion, confidence, recorded_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    source, session_id,
                    h.get("emotion", "neutral"),
                    h.get("confidence", 0.0),
                    h.get("time", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                ))
            self._conn.commit()

    def load_emotion_history(self, source: str = "detection",
                             session_id: str = "",
                             limit: int = 200) -> list[dict]:
        """Load recent emotion history."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT emotion, confidence, recorded_at FROM emotion_history "
                "WHERE source = ? AND session_id = ? "
                "ORDER BY id DESC LIMIT ?",
                (source, session_id, limit)
            ).fetchall()
        # Return in chronological order
        return [{"emotion": r["emotion"], "confidence": r["confidence"],
                 "time": r["recorded_at"]} for r in reversed(rows)]

    def clear_emotion_history(self, source: str = "detection",
                              session_id: str = "") -> None:
        """Clear emotion history for a source."""
        with self._lock:
            self._conn.execute(
                "DELETE FROM emotion_history WHERE source = ? AND session_id = ?",
                (source, session_id)
            )
            self._conn.commit()

    # ── Stats ─────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Return aggregate stats across all data."""
        with self._lock:
            sessions = self._conn.execute(
                "SELECT COUNT(*) as cnt FROM diary_sessions"
            ).fetchone()["cnt"]
            entries = self._conn.execute(
                "SELECT COUNT(*) as cnt FROM diary_entries"
            ).fetchone()["cnt"]
            chats = self._conn.execute(
                "SELECT COUNT(*) as cnt FROM chat_messages"
            ).fetchone()["cnt"]
            emotions = self._conn.execute(
                "SELECT COUNT(*) as cnt FROM emotion_history"
            ).fetchone()["cnt"]
        return {
            "total_sessions": sessions,
            "total_entries": entries,
            "total_chat_messages": chats,
            "total_emotion_snapshots": emotions,
        }

    # ── Cleanup ───────────────────────────────────────────────────

    def close(self):
        """Close the database connection."""
        with self._lock:
            self._conn.close()
