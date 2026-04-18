"""
Tests for memory_store.py ??SQLite persistent storage.

Covers:
  - DB init and schema creation
  - Diary session save/load/delete
  - Chat message persistence
  - Emotion history persistence
  - Audio file save/load
  - Stats reporting
  - Thread safety basics
  - Edge cases (empty data, duplicates, missing fields)
"""

import sys
import os
import json
import tempfile
import shutil
import threading
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def tmp_dir():
    """Create a temporary directory for test DB and audio files."""
    d = tempfile.mkdtemp(prefix="emotiscan_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def store(tmp_dir):
    """Create a MemoryStore with a temp DB."""
    db_path = os.path.join(tmp_dir, "test.db")
    # Patch config before importing
    import config
    orig_data_dir = config.DATA_DIR
    orig_db_file = config.DB_FILE
    config.DATA_DIR = tmp_dir
    config.DB_FILE = db_path
    from services.memory_store import MemoryStore
    ms = MemoryStore(db_path=db_path)
    yield ms
    ms.close()
    config.DATA_DIR = orig_data_dir
    config.DB_FILE = orig_db_file


@pytest.fixture
def sample_session():
    """Return a sample diary session dict."""
    return {
        "session_id": "test1234",
        "start_time": "2026-03-18 16:00:00",
        "summary": "User felt happy about passing all tests.",
        "compassionate_response": "Great job! Your positivity is inspiring.",
        "research_queries": ["positive psychology well-being"],
        "web_results": [{"title": "Happy", "url": "https://example.com", "snippet": "Be happy"}],
        "arxiv_results": [{"title": "Happiness study", "url": "https://arxiv.org/123", "authors": ["Smith"]}],
        "entries": [
            {
                "timestamp": "16:01:00",
                "text": "I feel great today!",
                "face_emotion": "happy",
                "face_confidence": 0.92,
                "voice_sentiment": {"polarity": 0.8, "emotion": "happy", "confidence": 0.85},
                "audio_emotion": {"energy": 0.7, "estimated_emotion": "happy", "confidence": 0.75},
                "fused_emotion": "happy",
                "fused_confidence": 0.85,
                "face_emotion_timeline": [
                    {"time": "16:01:00", "emotion": "happy", "confidence": 0.9},
                    {"time": "16:01:01", "emotion": "happy", "confidence": 0.88},
                ],
            },
            {
                "timestamp": "16:02:00",
                "text": "Tests are passing, feeling accomplished.",
                "face_emotion": "happy",
                "face_confidence": 0.88,
                "voice_sentiment": {"polarity": 0.6, "emotion": "happy", "confidence": 0.7},
                "audio_emotion": {"energy": 0.5, "estimated_emotion": "neutral", "confidence": 0.6},
                "fused_emotion": "happy",
                "fused_confidence": 0.75,
                "face_emotion_timeline": [],
            },
        ],
    }


# ?€?€ DB Init ?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€

class TestDBInit:
    def test_creates_db_file(self, store, tmp_dir):
        db_path = os.path.join(tmp_dir, "test.db")
        assert os.path.exists(db_path)

    def test_creates_tables(self, store):
        tables = store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        table_names = {r["name"] for r in tables}
        assert "diary_sessions" in table_names
        assert "diary_entries" in table_names
        assert "chat_messages" in table_names
        assert "emotion_history" in table_names
        assert "schema_version" in table_names

    def test_schema_version_set(self, store):
        row = store._conn.execute("SELECT version FROM schema_version").fetchone()
        assert row["version"] == 1

    def test_wal_mode(self, store):
        mode = store._conn.execute("PRAGMA journal_mode").fetchone()
        assert mode[0] == "wal"

    def test_audio_dir_created(self, store, tmp_dir):
        audio_dir = os.path.join(tmp_dir, "audio")
        assert os.path.isdir(audio_dir)


# ?€?€ Diary Sessions ?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€

class TestDiarySessions:
    def test_save_and_load_session(self, store, sample_session):
        store.save_session(sample_session)
        loaded = store.load_session("test1234")
        assert loaded is not None
        assert loaded["session_id"] == "test1234"
        assert loaded["summary"] == sample_session["summary"]
        assert loaded["compassionate_response"] == sample_session["compassionate_response"]
        assert len(loaded["entries"]) == 2
        assert loaded["entries"][0]["text"] == "I feel great today!"
        assert loaded["entries"][0]["fused_emotion"] == "happy"

    def test_load_all_sessions(self, store, sample_session):
        store.save_session(sample_session)
        session2 = dict(sample_session, session_id="test5678", start_time="2026-03-18 17:00:00")
        session2["entries"] = [sample_session["entries"][0]]
        store.save_session(session2)
        all_sessions = store.load_all_sessions()
        assert len(all_sessions) == 2
        ids = {s["session_id"] for s in all_sessions}
        assert ids == {"test1234", "test5678"}

    def test_load_session_not_found(self, store):
        assert store.load_session("nonexistent") is None

    def test_delete_session(self, store, sample_session):
        store.save_session(sample_session)
        assert store.delete_session("test1234") is True
        assert store.load_session("test1234") is None
        assert store.delete_session("test1234") is False  # already deleted

    def test_save_replaces_existing(self, store, sample_session):
        store.save_session(sample_session)
        updated = dict(sample_session, summary="Updated summary")
        store.save_session(updated)
        loaded = store.load_session("test1234")
        assert loaded["summary"] == "Updated summary"
        assert len(loaded["entries"]) == 2  # entries re-created

    def test_session_count(self, store, sample_session):
        assert store.get_session_count() == 0
        store.save_session(sample_session)
        assert store.get_session_count() == 1

    def test_json_fields_preserved(self, store, sample_session):
        store.save_session(sample_session)
        loaded = store.load_session("test1234")
        assert loaded["research_queries"] == ["positive psychology well-being"]
        assert loaded["web_results"][0]["title"] == "Happy"
        assert loaded["arxiv_results"][0]["authors"] == ["Smith"]

    def test_entry_voice_sentiment_preserved(self, store, sample_session):
        store.save_session(sample_session)
        loaded = store.load_session("test1234")
        vs = loaded["entries"][0]["voice_sentiment"]
        assert vs["polarity"] == 0.8
        assert vs["emotion"] == "happy"

    def test_face_emotion_timeline_preserved(self, store, sample_session):
        store.save_session(sample_session)
        loaded = store.load_session("test1234")
        ftl = loaded["entries"][0]["face_emotion_timeline"]
        assert len(ftl) == 2
        assert ftl[0]["emotion"] == "happy"

    def test_empty_entries_session(self, store):
        session = {
            "session_id": "empty01",
            "start_time": "2026-03-18 18:00:00",
            "entries": [],
            "summary": "",
            "compassionate_response": "",
            "research_queries": [],
            "web_results": [],
            "arxiv_results": [],
        }
        store.save_session(session)
        loaded = store.load_session("empty01")
        assert loaded is not None
        assert loaded["entries"] == []

    def test_entry_with_missing_fields(self, store):
        """Entries with missing optional fields should still save."""
        session = {
            "session_id": "partial1",
            "start_time": "2026-03-18 18:00:00",
            "entries": [{"timestamp": "18:00:00"}],  # minimal entry
            "summary": "",
        }
        store.save_session(session)
        loaded = store.load_session("partial1")
        assert loaded is not None
        assert loaded["entries"][0]["face_emotion"] == "neutral"


# ?€?€ Chat Messages ?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€

class TestChatMessages:
    def test_save_and_load_chat(self, store):
        messages = [
            {"role": "user", "content": "How am I feeling?"},
            {"role": "assistant", "content": "You seem happy today!"},
        ]
        store.save_chat_history(messages, session_id="")
        loaded = store.load_chat_history(session_id="")
        assert len(loaded) == 2
        assert loaded[0]["role"] == "user"
        assert loaded[1]["content"] == "You seem happy today!"

    def test_save_single_message(self, store):
        store.save_chat_message("user", "Hello", session_id="")
        store.save_chat_message("assistant", "Hi there!", session_id="")
        loaded = store.load_chat_history(session_id="")
        assert len(loaded) == 2

    def test_session_scoped_chat(self, store):
        store.save_chat_history([{"role": "user", "content": "General chat"}], session_id="")
        store.save_chat_history([{"role": "user", "content": "Diary chat"}], session_id="sess123")
        general = store.load_chat_history(session_id="")
        diary = store.load_chat_history(session_id="sess123")
        assert len(general) == 1
        assert general[0]["content"] == "General chat"
        assert len(diary) == 1
        assert diary[0]["content"] == "Diary chat"

    def test_clear_chat(self, store):
        store.save_chat_history([{"role": "user", "content": "test"}], session_id="")
        store.clear_chat_history(session_id="")
        loaded = store.load_chat_history(session_id="")
        assert len(loaded) == 0

    def test_replace_chat_history(self, store):
        store.save_chat_history([{"role": "user", "content": "old"}], session_id="")
        store.save_chat_history([{"role": "user", "content": "new"}], session_id="")
        loaded = store.load_chat_history(session_id="")
        assert len(loaded) == 1
        assert loaded[0]["content"] == "new"

    def test_empty_chat_history(self, store):
        loaded = store.load_chat_history(session_id="nonexistent")
        assert loaded == []


# ?€?€ Emotion History ?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€

class TestEmotionHistory:
    def test_save_and_load_emotion(self, store):
        store.save_emotion_snapshot("happy", 0.9, source="detection")
        store.save_emotion_snapshot("sad", 0.7, source="detection")
        loaded = store.load_emotion_history(source="detection", limit=10)
        assert len(loaded) == 2
        assert loaded[0]["emotion"] == "happy"
        assert loaded[1]["emotion"] == "sad"

    def test_bulk_save(self, store):
        history = [
            {"emotion": "happy", "confidence": 0.9, "time": "2026-03-18 16:01:00"},
            {"emotion": "neutral", "confidence": 0.5, "time": "2026-03-18 16:02:00"},
            {"emotion": "sad", "confidence": 0.8, "time": "2026-03-18 16:03:00"},
        ]
        store.save_emotion_history(history, source="detection")
        loaded = store.load_emotion_history(source="detection", limit=10)
        assert len(loaded) == 3
        assert loaded[0]["emotion"] == "happy"
        assert loaded[2]["emotion"] == "sad"

    def test_source_separation(self, store):
        store.save_emotion_snapshot("happy", 0.9, source="detection")
        store.save_emotion_snapshot("sad", 0.7, source="diary", session_id="s1")
        detection = store.load_emotion_history(source="detection")
        diary = store.load_emotion_history(source="diary", session_id="s1")
        assert len(detection) == 1
        assert len(diary) == 1

    def test_limit(self, store):
        for i in range(20):
            store.save_emotion_snapshot("happy", 0.9, source="detection")
        loaded = store.load_emotion_history(source="detection", limit=5)
        assert len(loaded) == 5

    def test_clear_emotion_history(self, store):
        store.save_emotion_snapshot("happy", 0.9, source="detection")
        store.clear_emotion_history(source="detection")
        loaded = store.load_emotion_history(source="detection")
        assert len(loaded) == 0


# ?€?€ Audio Files ?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€

class TestAudioFiles:
    def test_save_and_load_raw_pcm(self, store):
        raw_pcm = b"\x00\x01" * 8000  # 1 second of fake 16-bit PCM
        path = store.save_audio("test1234", 0, raw_pcm, sample_rate=16000)
        assert path.endswith(".wav")
        assert os.path.exists(path)
        loaded = store.load_audio(path)
        assert loaded is not None
        assert len(loaded) > len(raw_pcm)  # WAV header added

    def test_save_wav_bytes(self, store):
        import wave
        import io
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"\x00\x01" * 1000)
        wav_bytes = buf.getvalue()
        path = store.save_audio("test1234", 1, wav_bytes)
        loaded = store.load_audio(path)
        assert loaded == wav_bytes

    def test_load_nonexistent(self, store):
        assert store.load_audio("") is None
        assert store.load_audio("/nonexistent/path.wav") is None

    def test_delete_session_cleans_audio(self, store, sample_session):
        path = store.save_audio("test1234", 0, b"\x00" * 100)
        store.save_session(sample_session)
        assert os.path.exists(path)
        store.delete_session("test1234")
        assert not os.path.exists(path)


# ?€?€ Stats ?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€

class TestStats:
    def test_empty_stats(self, store):
        stats = store.get_stats()
        assert stats["total_sessions"] == 0
        assert stats["total_entries"] == 0
        assert stats["total_chat_messages"] == 0
        assert stats["total_emotion_snapshots"] == 0

    def test_populated_stats(self, store, sample_session):
        store.save_session(sample_session)
        store.save_chat_message("user", "hello", session_id="")
        store.save_emotion_snapshot("happy", 0.9)
        stats = store.get_stats()
        assert stats["total_sessions"] == 1
        assert stats["total_entries"] == 2
        assert stats["total_chat_messages"] == 1
        assert stats["total_emotion_snapshots"] == 1


# ?€?€ Thread Safety ?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€

class TestThreadSafety:
    def test_concurrent_writes(self, store):
        """Multiple threads writing sessions shouldn't crash."""
        errors = []

        def write_session(i):
            try:
                store.save_session({
                    "session_id": f"thread_{i}",
                    "start_time": "2026-03-18 16:00:00",
                    "entries": [{"timestamp": "16:00:00", "text": f"Thread {i}"}],
                    "summary": f"Thread {i} summary",
                })
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=write_session, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        assert len(errors) == 0, f"Errors: {errors}"
        assert store.get_session_count() == 10

    def test_concurrent_reads_and_writes(self, store, sample_session):
        """Reads during writes shouldn't crash."""
        store.save_session(sample_session)
        errors = []

        def reader():
            try:
                for _ in range(20):
                    store.load_all_sessions()
            except Exception as e:
                errors.append(e)

        def writer():
            try:
                for i in range(10):
                    store.save_chat_message("user", f"msg {i}")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=reader),
            threading.Thread(target=writer),
            threading.Thread(target=reader),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        assert len(errors) == 0, f"Errors: {errors}"


# ?€?€ Integration: DiarySession ??MemoryStore ?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€

class TestDiaryIntegration:
    def test_diary_session_roundtrip(self, store):
        """A DiarySession object can be saved and loaded as dict."""
        from diary.diary_session import DiarySession, DiaryEntry
        session = DiarySession(
            session_id="rt_test",
            start_time="2026-03-18 16:00:00",
        )
        entry = DiaryEntry(
            timestamp="16:01:00",
            text="Hello from diary",
            face_emotion="happy",
            face_confidence=0.9,
            voice_sentiment={"polarity": 0.5, "emotion": "happy", "confidence": 0.7},
            audio_emotion={"energy": 0.6, "estimated_emotion": "happy", "confidence": 0.65},
            fused_emotion="happy",
            fused_confidence=0.8,
            face_emotion_timeline=[{"time": "16:01:00", "emotion": "happy", "confidence": 0.9}],
        )
        session.entries.append(entry)
        session.summary = "A happy roundtrip test."
        session.compassionate_response = "Keep smiling!"

        # Save via to_dict()
        store.save_session(session.to_dict())

        # Load back
        loaded = store.load_session("rt_test")
        assert loaded is not None
        assert loaded["summary"] == "A happy roundtrip test."
        assert loaded["entries"][0]["text"] == "Hello from diary"
        assert loaded["entries"][0]["face_emotion_timeline"][0]["emotion"] == "happy"

    def test_multiple_sessions_stored(self, store):
        """Multiple sessions should all be stored and retrievable."""
        for i in range(3):
            store.save_session({
                "session_id": f"order_{i}",
                "start_time": f"2026-03-18 {16 + i}:00:00",
                "entries": [],
                "summary": f"Session {i}",
            })
        all_sessions = store.load_all_sessions()
        assert len(all_sessions) == 3
        ids = {s["session_id"] for s in all_sessions}
        assert ids == {"order_0", "order_1", "order_2"}
