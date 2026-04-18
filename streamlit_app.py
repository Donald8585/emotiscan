"""
EmotiScan: Real-Time Emotion-Aware Research & Mood Booster
Run: streamlit run streamlit_app.py
"""

import streamlit as st
import time
import socket
import subprocess
import threading
import numpy as np
from datetime import datetime

from config import (
    PAGE_TITLE, PAGE_ICON, LAYOUT, OLLAMA_URL, OLLAMA_MODEL,
    MCP_RANKER_HOST, MCP_RANKER_PORT, GPU_LAYERS,
    EMOTION_LABELS, NEGATIVE_EMOTIONS, EMOTION_RESEARCH_TOPICS,
    QUERY_EXPANSIONS, WEBRTC_STUN_SERVER, AUTO_SAVE, DATA_DIR,
)
from services.memory_store import MemoryStore
from ui.ascii_art_generator import (
    STATIC_HEADER, STATIC_SEPARATOR, TOPIC_ART,
    format_paper_card, get_emotion_art, get_mood_booster,
    generate_dynamic_art, get_topic_art,
)
from services.llm_service import LLMService
from services.web_search_service import search_emotion_articles, search_general, search_coping_strategies
from mcp.mcp_client import search_papers, rank_results
from diary.diary_session import DiarySessionManager, DiarySession, DiaryEntry
from services.speech_service import SpeechService

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT,
    initial_sidebar_state="expanded",
)

# Hide Streamlit's default Deploy button and top-right toolbar menu
st.markdown(
    """
    <style>
    [data-testid="stToolbar"] {visibility: hidden; height: 0; position: fixed;}
    [data-testid="stDeployButton"] {display: none;}
    #MainMenu {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Helpers ──────────────────────────────────────────────────────

def check_port(host, port, timeout=1):
    try:
        s = socket.create_connection((host, port), timeout=timeout)
        s.close()
        return True
    except Exception:
        return False


def detect_gpu():
    info = {"available": False, "name": "None", "vram": "N/A"}
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0 and r.stdout.strip():
            parts = r.stdout.strip().split("\n")[0].split(", ")
            info["available"] = True
            info["name"] = parts[0].strip() if parts else "Unknown"
            info["vram"] = (parts[1].strip() + " MB") if len(parts) > 1 else "N/A"
    except Exception:
        pass
    return info


def rel_bar(score, width=20):
    filled = int(score * width)
    return "#" * filled + "." * (width - filled)


def expand_query(topic):
    t = topic.lower().strip()
    if t in QUERY_EXPANSIONS:
        return topic + " " + QUERY_EXPANSIONS[t]
    if len(topic.split()) <= 2:
        return topic + " " + topic + " research papers academic"
    return topic



def generate_insights(topic, papers, news):
    insights = []
    if papers:
        top = papers[0]
        insights.append(f"Top-ranked paper: **{top['title']}** (relevance: {round(top.get('relevance_score', 0), 2)})")
        avg_score = sum(p.get("relevance_score", 0) for p in papers) / len(papers)
        insights.append(f"Average paper relevance: {round(avg_score, 2)} across {len(papers)} papers")
    if news:
        insights.append(f"Trending: **{news[0]['title']}** ({news[0].get('points', 0)} points)")
    if not insights:
        insights.append("No results matched your query. Try a broader topic.")
    return insights


def make_digest(topic, ranked_papers, ranked_news):
    out = f"## Research Digest: {topic}\n\n### Key Papers\n\n"
    if ranked_papers:
        for i, p in enumerate(ranked_papers[:3], 1):
            auth = ", ".join(p.get("authors", [])[:2])
            out += f"{i}. **{p['title']}** ({auth})\n"
            out += f"   - {p.get('abstract', '')[:150]}...\n"
            out += f"   - Relevance: {round(p.get('relevance_score', 0), 2)}\n\n"
    else:
        out += "_No papers found._\n\n"
    out += "### Key Insights\n\n"
    for ins in generate_insights(topic, ranked_papers[:3], ranked_news[:3]):
        out += f"- {ins}\n"
    return out


# ── WebRTC / Emotion Video Processor (module level) ─────────────
# Defined here so both Emotion Detection tab and Voice Diary tab can use them.

_webrtc_imports_ok = False
EmotionVideoProcessor = None
EmotionAudioVideoProcessor = None
DiaryVideoProcessor = None
DiaryAudioProcessor = None
_diary_buffer = None
rtc_config = None

try:
    from streamlit_webrtc import (
        webrtc_streamer as _webrtc_streamer,
        VideoProcessorBase,
        AudioProcessorBase,
        RTCConfiguration,
    )
    import av as _av
    import io as _io
    import wave as _wave
    import struct as _struct

    # ── Shared audio buffer (module-level, survives WebRTC stop) ─────
    # The WebRTC processor objects are garbage-collected when the stream stops,
    # so we keep the audio + emotion data in a module-level buffer.
    class _SharedDiaryBuffer:
        """Thread-safe buffer that persists across WebRTC lifecycle."""
        def __init__(self):
            self._lock = threading.Lock()
            self.audio_frames: list = []       # list of PCM bytes chunks
            self.sample_rate: int = 48000
            self.emotion_timeline: list = []   # [(elapsed_s, emotion, confidence)]
            self.start_time: float = time.time()
            self.current_emotion: str = "neutral"
            self.current_confidence: float = 0.0

        def append_audio(self, pcm_bytes: bytes):
            with self._lock:
                self.audio_frames.append(pcm_bytes)

        def append_emotion(self, elapsed: float, emotion: str, confidence: float):
            with self._lock:
                self.emotion_timeline.append((elapsed, emotion, confidence))

        def set_current_emotion(self, emotion: str, confidence: float):
            with self._lock:
                self.current_emotion = emotion
                self.current_confidence = confidence

        def get_current_emotion(self) -> tuple:
            with self._lock:
                return self.current_emotion, self.current_confidence

        @property
        def has_audio(self) -> bool:
            with self._lock:
                return len(self.audio_frames) > 0

        @property
        def audio_duration_seconds(self) -> float:
            with self._lock:
                total = sum(len(f) for f in self.audio_frames)
            return total / (self.sample_rate * 2)  # 16-bit mono = 2 bytes/sample

        @property
        def emotion_during_recording(self) -> dict:
            """Return the dominant emotion weighted by how long each emotion
            was active during the recording.  The timeline only stores
            *transitions* (not every frame), so a simple count would treat
            a 30-second stretch the same as a 0.1-second blip.  Instead we
            compute the elapsed time each emotion occupied."""
            with self._lock:
                tl = list(self.emotion_timeline)
                cur_emo = self.current_emotion
                cur_conf = self.current_confidence
                now_elapsed = time.time() - self.start_time
            if not tl:
                return {"emotion": cur_emo, "confidence": cur_conf}

            # Time-weighted: each emotion's duration = (next timestamp - this timestamp)
            # Last entry extends to now (current recording time)
            durations: dict[str, float] = {}
            conf_sums: dict[str, float] = {}
            for i, (t, emo, conf) in enumerate(tl):
                if i + 1 < len(tl):
                    dt = tl[i + 1][0] - t
                else:
                    dt = now_elapsed - t  # last entry runs until now
                dt = max(dt, 0.0)
                durations[emo] = durations.get(emo, 0.0) + dt
                conf_sums[emo] = conf_sums.get(emo, 0.0) + conf * dt

            if not durations:
                return {"emotion": cur_emo, "confidence": cur_conf}
            dominant = max(durations, key=durations.get)
            total_dur = durations[dominant]
            avg_conf = conf_sums[dominant] / total_dur if total_dur > 0 else 0.5
            return {"emotion": dominant, "confidence": avg_conf}

        def get_audio_wav(self) -> bytes:
            with self._lock:
                frames = list(self.audio_frames)
                sr = self.sample_rate
            if not frames:
                return b""
            pcm = b"".join(frames)
            buf = _io.BytesIO()
            with _wave.open(buf, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes(pcm)
            return buf.getvalue()

        def clear(self):
            with self._lock:
                self.audio_frames.clear()
                self.emotion_timeline.clear()
                self.start_time = time.time()

    _diary_buffer = _SharedDiaryBuffer()

    # ── Video Processor (async detection for zero-lag video) ────────────
    class EmotionVideoProcessor(VideoProcessorBase):
        """Thread-safe video processor with async background detection.

        Detection runs in a background thread so recv() NEVER blocks.
        The video feed stays smooth at full frame rate while detection
        results update asynchronously.
        """

        # Downsample to this width for detection only (display stays full res)
        # 320px keeps faces large enough for Haar cascade (MIN_FACE_SIZE in detector)
        DETECT_WIDTH = 320
        # Minimum interval between detection submissions (seconds).
        # 0.5s = 2 detections/sec — balances responsiveness vs CPU load.
        DETECT_INTERVAL = 0.5

        def __init__(self):
            self._lock = threading.Lock()
            self._emotion = "neutral"
            self._confidence = 0.0
            self._all_emotions = {}
            self._detector = None
            self._frame_idx = 0
            self._last_results = None     # last detection results (for redraw)
            self._last_detect_time = 0.0  # time of last detection submission
            self._detect_busy = False     # True while bg thread is running
            self._pending_results = None  # results from bg thread
            self._last_scale = 1.0
            self._last_img_shape = None

        def _get_detector(self):
            if self._detector is None:
                try:
                    from services.emotion_detector import get_shared_detector
                    self._detector = get_shared_detector()
                except Exception:
                    self._detector = None
            return self._detector

        @property
        def emotion(self):
            with self._lock:
                return self._emotion

        @property
        def confidence(self):
            with self._lock:
                return self._confidence

        @property
        def all_emotions(self):
            with self._lock:
                return dict(self._all_emotions)

        def _run_detection(self, small, scale, img_shape):
            """Run detection in background thread — NEVER call from recv()."""
            try:
                detector = self._get_detector()
                if detector and detector.is_ready:
                    _, results = detector.detect_emotions(small)
                    if results:
                        # Scale bboxes back to original resolution
                        if scale != 1.0:
                            for r in results:
                                bx, by, bw, bh = r["bbox"]
                                r["bbox"] = [
                                    int(bx / scale), int(by / scale),
                                    int(bw / scale), int(bh / scale),
                                ]
                        top = results[0]
                        with self._lock:
                            self._emotion = top["emotion"]
                            self._confidence = top["confidence"]
                            self._all_emotions = top.get("all_emotions", {})
                            self._pending_results = results
                            self._last_scale = scale
                            self._last_img_shape = img_shape
            except Exception:
                pass
            finally:
                self._detect_busy = False

        def _draw_results_on(self, img, results):
            """Draw bounding boxes + emotion labels on the image."""
            detector = self._get_detector()
            if not detector:
                return img
            annotated = img.copy()
            for r in results:
                x, y, rw, rh = r["bbox"]
                detector._draw_annotation(
                    annotated, x, y, rw, rh,
                    r["emotion"], r["confidence"]
                )
            return annotated

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            self._frame_idx += 1
            now = time.time()

            # Consume pending results from background detection
            with self._lock:
                pending = self._pending_results
                if pending is not None:
                    self._pending_results = None
                    self._last_results = pending

            # Submit new detection if not busy and enough time has passed
            if (not self._detect_busy
                    and (now - self._last_detect_time) >= self.DETECT_INTERVAL):
                self._last_detect_time = now
                self._detect_busy = True

                # Downsample for detection
                h, w = img.shape[:2]
                if w > self.DETECT_WIDTH:
                    import cv2 as _cv2_resize
                    scale = self.DETECT_WIDTH / w
                    small = _cv2_resize.resize(
                        img, (self.DETECT_WIDTH, int(h * scale)),
                        interpolation=_cv2_resize.INTER_AREA
                    )
                else:
                    small = img.copy()
                    scale = 1.0

                t = threading.Thread(
                    target=self._run_detection,
                    args=(small, scale, img.shape),
                    daemon=True,
                )
                t.start()

            # Always draw on the CURRENT frame (not a cached frame) so the
            # video stays smooth and lag-free.  Use last known results.
            with self._lock:
                last = getattr(self, "_last_results", None)
            if last is not None:
                annotated = self._draw_results_on(img, last)
                return _av.VideoFrame.from_ndarray(annotated, format="bgr24")
            return frame


    # ── Diary Video Processor (writes emotion to shared buffer) ──────
    class DiaryVideoProcessor(EmotionVideoProcessor):
        """Video emotion detector that also writes to the shared diary buffer.

        Uses a longer detection interval (1.0s vs 0.5s) since the diary tab
        has more rendering overhead (audio input, chart, entries list).  The
        video feed itself stays smooth because recv() never blocks — only
        the background detection thread runs less often.
        """
        DETECT_INTERVAL = 1.0  # 1 detection/sec (vs 0.5s in emotion tab)

        def __init__(self):
            super().__init__()
            self._last_pushed_emotion = None

        def recv(self, frame):
            result = super().recv(frame)
            # Always push current emotion to buffer so the live panel
            # can read it at any time. Append to timeline only on CHANGE
            # to keep the timeline compact.
            emo = self.emotion
            conf = self.confidence
            _diary_buffer.set_current_emotion(emo, conf)
            if emo != self._last_pushed_emotion:
                self._last_pushed_emotion = emo
                elapsed = time.time() - _diary_buffer.start_time
                _diary_buffer.append_emotion(elapsed, emo, conf)
            return result


    # ── Diary Audio Processor (captures mic audio to shared buffer) ──
    class DiaryAudioProcessor(AudioProcessorBase):
        """Captures ALL audio frames from the microphone into the shared buffer.

        Uses recv_queued() instead of recv() because async mode's recv()
        only delivers the LATEST frame and drops everything in between,
        which creates huge gaps in the audio recording.
        recv_queued() delivers ALL accumulated frames since the last call.
        """

        def __init__(self):
            super().__init__()
            self._frames_received = 0

        def _process_frame(self, frame) -> None:
            """Convert a single av.AudioFrame to PCM and append to buffer."""
            try:
                audio_array = frame.to_ndarray()
                if audio_array.ndim > 1:
                    # Multi-channel: average to mono
                    mono = audio_array.mean(axis=0)
                else:
                    mono = audio_array.flatten()

                _diary_buffer.sample_rate = frame.sample_rate or 48000

                if mono.dtype.kind == 'f':
                    pcm = (mono * 32767).clip(-32768, 32767).astype('int16')
                else:
                    pcm = mono.astype('int16')

                pcm_bytes = pcm.tobytes()
                if len(pcm_bytes) > 0:
                    _diary_buffer.append_audio(pcm_bytes)
                    self._frames_received += 1
            except Exception:
                try:
                    raw = frame.planes[0].to_bytes()
                    if len(raw) > 0:
                        _diary_buffer.append_audio(raw)
                        self._frames_received += 1
                except Exception:
                    pass

        async def recv_queued(self, frames):
            """Process ALL queued audio frames (no drops)."""
            for frame in frames:
                self._process_frame(frame)
            # Must return the frames list for playback pipeline
            return frames

        def recv(self, frame):
            """Fallback for non-async mode."""
            self._process_frame(frame)
            return frame


    # Keep backward compat reference
    EmotionAudioVideoProcessor = DiaryVideoProcessor

    rtc_config = RTCConfiguration({"iceServers": [{"urls": [WEBRTC_STUN_SERVER]}]})
    _webrtc_imports_ok = True

except ImportError:
    _webrtc_streamer = None
except Exception:
    _webrtc_streamer = None


# ── Session State ────────────────────────────────────────────────

defaults = {
    "memory_topics": [],
    "digest_history": [],
    "current_digest": None,
    "chat_history": [],
    "detected_emotion": "neutral",
    "emotion_confidence": 0.0,
    "emotion_history": [],
    "art_history": [],
    "diary_session": None,
    "diary_history": [],
    "diary_chat_history": [],
    "last_diary_audio": None,
    "last_ended_session": None,
    "diary_face_emotion_live": "neutral",
    "diary_face_confidence_live": 0.0,
    "diary_webrtc_audio_processed": False,
    "diary_entry_count_at_last_save": 0,
    "diary_emotion_history": [],
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ── Persistent Memory Store ──────────────────────────────────────
# Singleton: one MemoryStore per Streamlit server process
@st.cache_resource
def _get_memory_store():
    return MemoryStore()

memory = _get_memory_store()

# Load persisted data into session_state on first run
if "_memory_loaded" not in st.session_state:
    st.session_state._memory_loaded = True
    try:
        # Restore diary sessions
        _saved_sessions = memory.load_all_sessions()
        if _saved_sessions and not st.session_state.diary_history:
            st.session_state.diary_history = _saved_sessions
        # Restore general chat history
        _saved_chat = memory.load_chat_history(session_id="")
        if _saved_chat and not st.session_state.chat_history:
            st.session_state.chat_history = [{"role": m["role"], "content": m["content"]} for m in _saved_chat]
        # Restore diary chat history (last session)
        if _saved_sessions:
            _last_sid = _saved_sessions[0].get("session_id", "")
            _saved_diary_chat = memory.load_chat_history(session_id=_last_sid)
            if _saved_diary_chat and not st.session_state.diary_chat_history:
                st.session_state.diary_chat_history = [{"role": m["role"], "content": m["content"]} for m in _saved_diary_chat]
        # Restore emotion detection history — but detect contamination first.
        # If ALL saved emotions are the same label, the DB was likely corrupted
        # by a previous rebalancing/bootstrap run.  Purge and start fresh.
        _saved_emo = memory.load_emotion_history(source="detection", limit=50)
        if _saved_emo and not st.session_state.emotion_history:
            _emo_labels = [e.get("emotion", "neutral") if isinstance(e, dict) else "neutral"
                           for e in _saved_emo]
            _unique_emos = set(_emo_labels)
            if len(_saved_emo) >= 10 and len(_unique_emos) == 1 and "neutral" not in _unique_emos:
                # All entries are the same non-neutral emotion — contaminated
                import logging
                logging.getLogger(__name__).warning(
                    "Detected contaminated emotion history (all '%s'). Purging.",
                    _emo_labels[0],
                )
                memory.clear_emotion_history(source="detection")
            else:
                st.session_state.emotion_history = _saved_emo
    except Exception as _e:
        import logging
        logging.getLogger(__name__).warning("Failed to load persisted data: %s", _e)


def _persist_diary_session(session_obj):
    """Save a completed diary session to disk."""
    if not AUTO_SAVE:
        return
    try:
        if isinstance(session_obj, DiarySession):
            memory.save_session(session_obj.to_dict())
        elif isinstance(session_obj, dict):
            memory.save_session(session_obj)
    except Exception as _e:
        import logging
        logging.getLogger(__name__).warning("Failed to persist diary session: %s", _e)


def _get_dominant_emotion(session_obj):
    """Get dominant emotion from a DiarySession or dict."""
    if isinstance(session_obj, DiarySession):
        return session_obj.dominant_emotion
    entries = session_obj.get("entries", []) if isinstance(session_obj, dict) else []
    emos = [e.get("fused_emotion", "neutral") for e in entries if isinstance(e, dict)]
    return max(set(emos), key=emos.count) if emos else "neutral"


def _get_best_emotion(session_obj):
    """Get the best non-neutral emotion across all modalities.

    Uses best_emotion on DiarySession (checks face, voice, audio, fused).
    For dict sessions, manually aggregates all modality emotions.
    """
    if isinstance(session_obj, DiarySession):
        return session_obj.best_emotion
    entries = session_obj.get("entries", []) if isinstance(session_obj, dict) else []
    all_emos = []
    for e in entries:
        if isinstance(e, dict):
            all_emos.append(e.get("fused_emotion", "neutral"))
            all_emos.append(e.get("face_emotion", "neutral"))
            vs = e.get("voice_sentiment", {})
            all_emos.append(vs.get("emotion", "neutral") if isinstance(vs, dict) else "neutral")
            ae = e.get("audio_emotion", {})
            all_emos.append(ae.get("estimated_emotion", "neutral") if isinstance(ae, dict) else "neutral")
    non_neutral = [em for em in all_emos if em and em != "neutral"]
    if non_neutral:
        return max(set(non_neutral), key=non_neutral.count)
    return "neutral"


def _get_session_transcript(session_obj) -> str:
    """Extract the full transcript text from a session."""
    if isinstance(session_obj, DiarySession):
        ctx = session_obj.get_full_context()
        return ctx["transcript"]
    entries = session_obj.get("entries", []) if isinstance(session_obj, dict) else []
    return " ".join(e.get("text", "") for e in entries if isinstance(e, dict))


def _persist_chat(messages, session_id=""):
    """Save chat history to disk."""
    if not AUTO_SAVE:
        return
    try:
        memory.save_chat_history(messages, session_id=session_id)
    except Exception as _e:
        import logging
        logging.getLogger(__name__).warning("Failed to persist chat: %s", _e)


# ── Service checks ───────────────────────────────────────────────
# Parse Ollama host/port from OLLAMA_URL (supports Docker host.docker.internal)
try:
    from urllib.parse import urlparse as _parse_url
    _ollama_parsed = _parse_url(OLLAMA_URL)
    _ollama_host = _ollama_parsed.hostname or "localhost"
    _ollama_port = _ollama_parsed.port or 8003
except Exception:
    _ollama_host, _ollama_port = "localhost", 8003
ollama_up = check_port(_ollama_host, _ollama_port)
mcp_up = check_port(MCP_RANKER_HOST, MCP_RANKER_PORT)
gpu_info = detect_gpu()
llm = LLMService()

# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.title("EmotiScan")
    st.caption("Emotion-Aware Research & Mood Booster")
    st.divider()

    st.subheader("Service Status")
    sc1, sc2 = st.columns(2)
    sc1.metric("Ollama", "UP" if ollama_up else "OFF")
    sc2.metric("Ranker", "UP" if mcp_up else "OFF")

    if gpu_info["available"]:
        st.success(f"GPU: {gpu_info['name']} ({gpu_info['vram']})")
    else:
        st.warning("GPU: Not detected (CPU mode)")

    st.divider()
    st.subheader("Current Mood")
    st.markdown(f"**{st.session_state.detected_emotion.upper()}** ({st.session_state.emotion_confidence:.0%})")

    st.divider()
    st.subheader("Settings")
    num_gpu_layers = st.slider("GPU Layers", 0, 99, value=99 if gpu_info["available"] else 0)

    st.divider()
    st.subheader("Recent Topics")
    if st.session_state.memory_topics:
        for t in st.session_state.memory_topics[-5:]:
            st.caption(f"- {t}")
    else:
        st.caption("No history yet.")
    if st.button("Clear Memory", width='stretch'):
        st.session_state.memory_topics = []
        st.session_state.digest_history = []
        st.session_state.current_digest = None
        st.session_state.chat_history = []
        # Clear persisted general chat
        if AUTO_SAVE:
            try:
                memory.clear_chat_history(session_id="")
            except Exception:
                pass
        st.success("Memory cleared!")

# ── Tabs ─────────────────────────────────────────────────────────
tab_emotion, tab_research, tab_mood, tab_diary, tab_chat, tab_about = st.tabs([
    "Emotion Detection", "Research Digest", "Mood Booster", "Voice Diary", "Chat", "About/Status"
])


# ════════════════════════════════════════════════════════════════
# TAB 1: Emotion Detection
# ════════════════════════════════════════════════════════════════
with tab_emotion:
    st.header("Real-Time Emotion Detection")

    # ── Backend status check ──
    try:
        from services.emotion_detector import EmotionDetector as _StatusDetector
        _status_det = _StatusDetector()
        _det_status = _status_det.status_info
        _backend_label = _det_status["backend"]
        _gpu_label = _det_status.get("gpu_device", "n/a")
        if _det_status["can_classify_emotions"]:
            _extra = f" | device: **{_gpu_label}**" if _gpu_label not in ("n/a", "cpu") else ""
            _emo_device = _det_status.get("emotion_device", "cpu")
            _emo_model = _det_status.get("emotion_model", "")
            if _backend_label == "yolo_hsemotion" and _gpu_label != "cpu":
                st.success(f"Emotion backend: **YOLOv8 + HSEmotion** (GPU-accelerated){_extra}")
            elif _backend_label == "yolo_hsemotion":
                st.success(f"Emotion backend: **YOLOv8 + HSEmotion** (CPU mode)")
            else:
                st.success(f"Emotion backend: **{_backend_label}** (real emotion classification active)")
        else:
            st.error(
                f"Emotion backend: **{_backend_label}** — cannot classify real emotions.\n\n"
                f"**Fix:** Run this in your terminal:\n"
                f"```\n/d/Python/python.exe -m pip install ultralytics hsemotion timm\n```\n\n"
                f"Then restart the Streamlit app."
            )
            if _det_status["errors_detail"]:
                with st.expander("Diagnostic details"):
                    for err in _det_status["errors_detail"]:
                        st.code(err)
    except Exception:
        pass

    # Try to set up webrtc, fall back gracefully
    webrtc_available = False
    webrtc_error = None
    ctx = None

    # Webcam session counter -- increment to force a fresh WebRTC component
    if "webcam_session" not in st.session_state:
        st.session_state.webcam_session = 0

    if _webrtc_imports_ok and EmotionVideoProcessor is not None:
        try:
            st.info(
                "Click **START** below to activate your webcam. "
                "Your browser will ask for camera permission -- allow it."
            )

            # Dynamic key so the component resets properly on restart
            webrtc_key = f"emotiscan-webcam-{st.session_state.webcam_session}"

            ctx = _webrtc_streamer(
                key=webrtc_key,
                rtc_configuration=rtc_config,
                video_processor_factory=EmotionVideoProcessor,
                media_stream_constraints={
                    "video": {"width": {"ideal": 640}, "height": {"ideal": 480}},
                    "audio": False,
                },
                async_processing=True,
            )
            webrtc_available = True

            # Reset button to fix stuck webcam
            if st.button("Reset Webcam", help="Click if webcam won't restart after stopping"):
                st.session_state.webcam_session += 1
                st.rerun()

        except Exception as e:
            webrtc_error = f"WebRTC init error: {e}"
    else:
        webrtc_error = "streamlit-webrtc not installed"

    # ── Live emotion readout (auto-refreshing via st.fragment) ──

    # Fallback: no webrtc at all
    if not webrtc_available:
        if webrtc_error:
            st.error(f"WebRTC unavailable: {webrtc_error}")
        st.info("Using manual emotion selection instead.")

    # Always show manual selector as backup (even alongside webcam)
    with st.expander("Manual Emotion Override", expanded=not webrtc_available):
        selected_emotion = st.selectbox("Select Emotion", EMOTION_LABELS, index=EMOTION_LABELS.index("neutral"))
        if st.button("Apply Manual Emotion"):
            st.session_state.detected_emotion = selected_emotion
            st.session_state.emotion_confidence = 0.8
            st.success(f"Set emotion to: {selected_emotion.upper()}")

    # This fragment auto-reruns every 1s independently of the main page,
    # so emotion stats, confidence bars, and the chart update in real time.
    @st.fragment(run_every=1.0)
    def _live_emotion_panel():
        """Auto-refreshing panel for live emotion readout."""
        if not webrtc_available or ctx is None:
            return

        if not ctx.state.playing:
            st.warning("Camera is not streaming yet. Press **START** above and allow camera access.")
            return

        if ctx.video_processor is None:
            st.info("Initializing emotion detector...")
            return

        emotion = ctx.video_processor.emotion
        confidence = ctx.video_processor.confidence
        all_emo = ctx.video_processor.all_emotions

        # Update session state
        st.session_state.detected_emotion = emotion
        st.session_state.emotion_confidence = confidence

        # Append to history (throttle: one entry per second)
        now_str = datetime.now().strftime("%H:%M:%S")
        last_time = (
            st.session_state.emotion_history[-1]["time"]
            if st.session_state.emotion_history
            else ""
        )
        if now_str != last_time:
            st.session_state.emotion_history.append({
                "time": now_str,
                "emotion": emotion,
                "confidence": confidence,
            })
            st.session_state.emotion_history = st.session_state.emotion_history[-50:]
            # Persist every 10 snapshots
            if AUTO_SAVE and len(st.session_state.emotion_history) % 10 == 0:
                try:
                    memory.save_emotion_history(
                        st.session_state.emotion_history[-10:], source="detection"
                    )
                except Exception:
                    pass

        # ── Emotion readout ──
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Detected Emotion", emotion.upper(), f"{confidence:.0%}")
        with col2:
            if all_emo:
                for emo_name, emo_val in sorted(all_emo.items(), key=lambda x: -x[1]):
                    st.progress(min(float(emo_val), 1.0), text=f"{emo_name}: {emo_val:.0%}")

        # ── Emotion history chart ──
        history = st.session_state.emotion_history[-30:]
        if history:
            try:
                import plotly.graph_objects as go
                emotion_to_num = {e: i for i, e in enumerate(EMOTION_LABELS)}
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=[h["time"] for h in history],
                    y=[emotion_to_num.get(h["emotion"], 6) for h in history],
                    mode="lines+markers",
                    name="Emotion",
                    text=[f"{h['emotion']} ({h['confidence']:.0%})" for h in history],
                ))
                fig.update_layout(
                    title="Emotion History",
                    yaxis=dict(
                        ticktext=EMOTION_LABELS,
                        tickvals=list(range(len(EMOTION_LABELS))),
                    ),
                    height=300,
                )
                st.plotly_chart(fig, width='stretch')
            except ImportError:
                st.caption("Install plotly for emotion history chart.")

    _live_emotion_panel()

    st.divider()

    # Research this emotion button
    current_emo = st.session_state.detected_emotion
    if st.button(f"Research '{current_emo.upper()}'", type="primary"):
        with st.spinner(f"Researching {current_emo}..."):
            research = llm.research_emotion(current_emo)
            st.markdown(research)

            web_results = search_emotion_articles(current_emo, max_results=3)
            if web_results:
                st.subheader("Related Articles")
                for r in web_results:
                    st.markdown(f"- [{r['title']}]({r['url']}): {r['snippet'][:100]}...")

    # Mood booster panel
    st.divider()
    st.subheader("Mood Booster")
    art = get_mood_booster(current_emo)
    st.code(art, language=None)


# ════════════════════════════════════════════════════════════════
# TAB 2: Research Digest
# ════════════════════════════════════════════════════════════════
with tab_research:
    st.header("Research Digest")

    # Emotion-aware topic suggestions
    current_emo = st.session_state.detected_emotion
    if current_emo != "neutral":
        suggested = EMOTION_RESEARCH_TOPICS.get(current_emo, "")
        if suggested:
            st.info(f"Based on your mood ({current_emo}), try researching: _{suggested}_")

    col_topic, col_opts = st.columns([3, 1])
    with col_topic:
        topic = st.text_input("Research Topic", placeholder="e.g. Large Language Models", key="research_topic")
    with col_opts:
        num_papers = st.slider("Max Papers", 3, 10, 5, key="num_papers")

    with st.expander("Advanced Options"):
        ac1, ac2, ac3 = st.columns(3)
        with ac1:
            use_arxiv = st.checkbox("ArXiv", value=True)
        with ac2:
            use_ranker = st.checkbox("Auto-Rank", value=True)
        with ac3:
            use_web_search = st.checkbox("Web Search", value=True)

        dr1, dr2 = st.columns(2)
        with dr1:
            date_from = st.date_input("From date", value=None, key="digest_date_from")
        with dr2:
            date_to = st.date_input("To date", value=None, key="digest_date_to")
        sort_by = st.selectbox("Sort by", options=["date", "relevance", "lastUpdated"], index=0)

    def _add_topic_to_memory():
        t = st.session_state.get("research_topic", "").strip()
        if t and t not in st.session_state.memory_topics:
            st.session_state.memory_topics.append(t)

    run_btn = st.button("Generate Digest", type="primary", width='stretch', disabled=not topic, on_click=_add_topic_to_memory)

    if run_btn and topic:

        progress = st.progress(0, text="Starting agent...")
        status = st.status("Agent is working...", expanded=True)

        with status:
            st.write("Step 1/5: Planning queries...")
            progress.progress(10, text="Planning...")
            expanded = expand_query(topic)

            st.write("Step 2/5: Fetching ArXiv papers...")
            progress.progress(30, text="Fetching papers...")
            papers = []
            if use_arxiv:
                try:
                    papers = search_papers(
                        query=topic, max_results=num_papers, sort_by=sort_by,
                        date_from=str(date_from) if date_from else None,
                        date_to=str(date_to) if date_to else None,
                        categories=None,
                    )
                    st.write(f"Fetched {len(papers)} papers from ArXiv")
                except Exception as e:
                    st.warning(f"ArXiv failed: {e}")

            articles = []

            st.write("Step 3/5: Web search...")
            progress.progress(50, text="Web searching...")
            web_results = []
            if use_web_search:
                web_results = search_general(f"{topic} research", max_results=3, enforce_relevance=False)
                st.write(f"Found {len(web_results)} web results")

            st.write("Step 4/5: Ranking results...")
            progress.progress(70, text="Ranking...")
            if use_ranker:
                ranked_papers = rank_results(expanded, papers)
                ranked_news = rank_results(expanded, articles)
            else:
                ranked_papers = papers
                ranked_news = articles
                for r in ranked_papers + ranked_news:
                    r["relevance_score"] = 0.5

            # Normalize web results into the same schema as ranked sources
            ranked_web = [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "abstract": r.get("snippet", ""),
                    "source": "web",
                    "relevance_score": 0.5,
                }
                for r in web_results
            ]

            st.write("Step 5/5: Generating digest...")
            progress.progress(85, text="Summarizing...")

            all_ranked = sorted(
                ranked_papers + ranked_news + ranked_web,
                key=lambda x: x.get("relevance_score", 0),
                reverse=True,
            )
            if ollama_up:
                st.write("Using LLM to summarise sources...")
                digest = llm.summarize_digest(topic, ranked_papers, web_results)
            else:
                st.write("Ollama offline — using template digest.")
                digest = make_digest(topic, ranked_papers, ranked_news)

            progress.progress(100, text="Done!")
            status.update(label="Digest ready!", state="complete")

        st.session_state.current_digest = {
            "topic": topic,
            "digest": digest,
            "sources": all_ranked,
            "papers": ranked_papers,
            "news": ranked_news,
            "web_results": web_results,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
        st.session_state.digest_history.append(st.session_state.current_digest)

    if st.session_state.current_digest:
        data = st.session_state.current_digest

        st.subheader(f"Digest: {data['topic']}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Sources", len(data.get("sources", [])))
        m2.metric("LLM", "Qwen3" if ollama_up else "Offline")
        m3.metric("History", len(st.session_state.digest_history))
        m4.metric("Ranker", "MCP" if mcp_up else "TF-IDF")

        st.markdown(data["digest"])

        with st.expander(f"View All Sources ({len(data.get('sources', []))})", expanded=False):
            for i, src in enumerate(data.get("sources", []), 1):
                with st.container(border=True):
                    col_a, col_b = st.columns([4, 1])
                    with col_a:
                        _src_map = {"arxiv": "paper", "web": "web"}
                        src_type = _src_map.get(src.get("source"), "news")
                        st.markdown(f"**{i}. [{src_type}] {src.get('title', 'Untitled')}**")
                        abstract = src.get("abstract", src.get("content", ""))
                        if len(abstract) > 200:
                            abstract = abstract[:200] + "..."
                        st.caption(abstract)
                    with col_b:
                        score = src.get("relevance_score", 0)
                        st.metric("Score", f"{round(score, 2)}")
                        if src.get("url"):
                            st.link_button("Open", src["url"], width='stretch')

        st.download_button(
            label="Download Digest (.md)",
            data=(
                f"# Research Digest: {data['topic']}\n\n{data['digest']}\n\n"
                + (
                    "## Web Sources\n\n"
                    + "\n".join(
                        f"- [{s.get('title', 'Untitled')}]({s.get('url', '')})  \n"
                        f"  {s.get('abstract', '')[:200]}"
                        for s in data.get("sources", [])
                        if s.get("source") == "web"
                    )
                    if any(s.get("source") == "web" for s in data.get("sources", []))
                    else ""
                )
            ),
            file_name=f"digest_{data['topic'].replace(' ', '_')}.md",
            mime="text/markdown",
            width='stretch',
        )

    # Digest history
    st.divider()
    st.subheader("Digest History")
    if st.session_state.digest_history:
        for entry in reversed(st.session_state.digest_history):
            with st.expander(f"{entry['topic']} - {entry['timestamp']}"):
                st.markdown(entry["digest"])
    else:
        st.info("No digests generated yet. Enter a topic above and click Generate!")


# ════════════════════════════════════════════════════════════════
# TAB 3: Mood Booster
# ════════════════════════════════════════════════════════════════
with tab_mood:
    st.header("Mood Booster")
    st.caption("ASCII art therapy for your soul (certified chaotic)")

    emo_select = st.selectbox(
        "Select Emotion (or auto-detect)",
        ["auto-detect"] + EMOTION_LABELS,
        key="mood_emotion",
    )
    selected_emo = st.session_state.detected_emotion if emo_select == "auto-detect" else emo_select

    # Large ASCII art
    st.subheader(f"Current Mood: {selected_emo.upper()}")
    art = get_emotion_art(selected_emo)
    st.code(art, language=None)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("New Emotion Art", width='stretch'):
            art = get_emotion_art(selected_emo)
            st.code(art, language=None)

    with col2:
        if st.button("Mood Booster!", type="primary", width='stretch'):
            booster = get_mood_booster(selected_emo)
            st.code(booster, language=None)

    st.divider()
    st.subheader("Dynamic Art Generator")
    art_text = st.text_input("Text for art", value="EmotiScan", key="art_text")
    art_style = st.selectbox("Style", ["wave", "matrix", "spiral", "blocks"], key="art_style")

    if st.button("Generate Dynamic Art", width='stretch'):
        dynamic = generate_dynamic_art(art_text, art_style)
        st.code(dynamic, language=None)
        st.session_state.art_history.append({"text": art_text, "style": art_style, "art": dynamic})

    # LLM joke/encouragement
    st.divider()
    st.subheader("AI Encouragement")
    if st.button("Get encouragement from AI", width='stretch'):
        with st.spinner("Generating..."):
            response = llm.generate_mood_response(selected_emo)
            st.markdown(response)

    # Art history
    if st.session_state.art_history:
        st.divider()
        st.subheader("Art History")
        for entry in reversed(st.session_state.art_history[-5:]):
            with st.expander(f"{entry['text']} ({entry['style']})"):
                st.code(entry["art"], language=None)


# ════════════════════════════════════════════════════════════════
# TAB 4: Voice Diary
# ════════════════════════════════════════════════════════════════
with tab_diary:
    st.header("Voice Diary")
    st.caption("Record your feelings with multimodal emotion detection (face + voice + text)")

    _diary_mgr = DiarySessionManager()

    # ── Post-session results display (shown after ending a session) ──
    if st.session_state.last_ended_session is not None and st.session_state.diary_session is None:
        _les = st.session_state.last_ended_session
        st.success("Session complete! Here are your insights:")

        # Compassionate response
        _les_comp = getattr(_les, "compassionate_response", "") if isinstance(_les, DiarySession) else _les.get("compassionate_response", "")
        if _les_comp:
            st.info(_les_comp)

        # Summary
        _les_summary = _les.summary if isinstance(_les, DiarySession) else _les.get("summary", "")
        if _les_summary:
            st.markdown(f"**Summary:** {_les_summary}")

        _les_best = _get_best_emotion(_les)
        _les_transcript = _get_session_transcript(_les)

        # Suggested solutions (pre-computed during end_session — no re-call)
        _solutions = (
            getattr(_les, "suggested_solutions", "")
            if isinstance(_les, DiarySession)
            else _les.get("suggested_solutions", "")
        )
        if _solutions:
            st.markdown("**Suggested Solutions:**")
            st.markdown(_solutions)

        # Coping strategies (pre-computed during end_session)
        _coping = (
            getattr(_les, "coping_strategies", [])
            if isinstance(_les, DiarySession)
            else _les.get("coping_strategies", [])
        )
        if _coping:
            st.markdown("**Coping Strategies:**")
            for _c in _coping:
                st.markdown(f"- [{_c['title']}]({_c['url']}): {_c['snippet'][:100]}...")

        # ArXiv papers
        _les_arxiv = getattr(_les, "arxiv_results", []) if isinstance(_les, DiarySession) else _les.get("arxiv_results", [])
        if _les_arxiv:
            st.markdown("**Relevant Research:**")
            for _p in _les_arxiv[:5]:
                st.markdown(f"- [{_p.get('title', 'Untitled')}]({_p.get('url', '')})")

        # Web results
        _les_web = getattr(_les, "web_results", []) if isinstance(_les, DiarySession) else _les.get("web_results", [])
        if _les_web:
            st.markdown("**Web Resources:**")
            for _w in _les_web[:5]:
                st.markdown(f"- [{_w.get('title', '')}]({_w.get('url', '')})")

        # ── Post-session chat (immediately available after results) ──
        st.divider()
        st.subheader("Talk About Your Feelings")
        st.caption("Chat with the AI counselor about your session, feelings, or anything on your mind.")

        for _msg in st.session_state.diary_chat_history:
            with st.chat_message(_msg["role"]):
                st.markdown(_msg["content"])

        if _diary_input := st.chat_input("Share what's on your mind...", key="diary_post_session_chat"):
            st.session_state.diary_chat_history.append({"role": "user", "content": _diary_input})
            with st.chat_message("user"):
                st.markdown(_diary_input)

            with st.chat_message("assistant"):
                with st.spinner("Listening..."):
                    _session_ctx = {
                        "summary": _les_summary,
                        "dominant_emotion": _les_best,
                        "compassionate_response": _les_comp,
                        "transcript": _les_transcript,
                    }
                    try:
                        _resp = llm.compassionate_chat(
                            _diary_input,
                            st.session_state.diary_chat_history,
                            _session_ctx,
                        )
                    except Exception:
                        _resp = llm.chat(
                            f"[User emotion: {_les_best}] {_diary_input}",
                            st.session_state.diary_chat_history,
                        )
                    st.markdown(_resp)
                    st.session_state.diary_chat_history.append({"role": "assistant", "content": _resp})
                    # Persist diary chat
                    _les_id = _les.session_id if isinstance(_les, DiarySession) else _les.get("session_id", "")
                    _persist_chat(st.session_state.diary_chat_history, session_id=_les_id)

        if st.button("Start Fresh", width='stretch'):
            st.session_state.last_ended_session = None
            st.session_state.diary_chat_history = []
            st.rerun()

        st.divider()

    # ── Session controls ──
    col_start, col_end = st.columns(2)
    with col_start:
        if st.button("Start New Session", type="primary", width='stretch',
                      disabled=st.session_state.diary_session is not None):
            st.session_state.diary_session = _diary_mgr.start_session()
            st.session_state.diary_chat_history = []
            st.session_state.last_diary_audio = None
            st.session_state.last_ended_session = None
            st.session_state.diary_face_emotion_live = "neutral"
            st.session_state.diary_face_confidence_live = 0.0
            st.session_state.diary_webrtc_audio_processed = False
            st.session_state.diary_entry_count_at_last_save = 0
            st.session_state.diary_emotion_history = []
            st.session_state._diary_buffer_needs_clear = True  # clear buffer on next render
            st.rerun()
    with col_end:
        if st.button("End Session", width='stretch',
                      disabled=st.session_state.diary_session is None):
            _end_sess_obj = st.session_state.diary_session
            _end_progress = st.progress(0, text="")
            _end_status = st.empty()

            # Step 1: Summarize
            _end_status.info("(๑˃ᴗ˂)ﻭ  Summarizing your diary session...")
            _end_progress.progress(10, text="Summarizing...")
            _end_ctx = _end_sess_obj.get_full_context()
            _end_sess_obj.summary = _diary_mgr.get_session_summary(_end_sess_obj, ctx=_end_ctx)

            # Step 2: Generate research queries
            _end_status.info("(｡•̀ᴗ-)✧  Figuring out what to research for you...")
            _end_progress.progress(25, text="Planning research...")
            from config import DIARY_AUTO_RESEARCH
            if DIARY_AUTO_RESEARCH:
                _end_sess_obj.research_queries = _diary_mgr.get_research_queries(_end_sess_obj, ctx=_end_ctx)

            # Step 3: LLM compassionate feedback
            _end_status.info("(ノ´ヮ`)ノ*: ・゚✧  Qwen is crafting thoughtful feedback...")
            _end_progress.progress(40, text="Generating feedback...")
            _end_sess_obj.compassionate_response = _diary_mgr.get_compassionate_response(_end_sess_obj, ctx=_end_ctx)

            # Step 4: ArXiv search
            _end_status.info("φ(゜▽゜*)♪  Searching academic papers on ArXiv...")
            _end_progress.progress(55, text="Searching ArXiv...")
            try:
                _end_sess_obj.arxiv_results = _diary_mgr._search_arxiv(
                    _end_sess_obj.research_queries, max_per_query=3
                )
            except Exception:
                _end_sess_obj.arxiv_results = []

            # Step 5: Web search
            _end_status.info("(つ≧▽≦)つ  Searching the web for helpful resources...")
            _end_progress.progress(75, text="Web searching...")
            try:
                from services.web_search_service import search_general, search_emotion_articles
                _end_web = []
                for _eq in (_end_sess_obj.research_queries or [])[:3]:
                    _end_web.extend(search_general(_eq, max_results=3))
                _emo_for_search = _end_ctx["best_emotion"]
                if _emo_for_search != "neutral":
                    _end_web.extend(search_emotion_articles(_emo_for_search, max_results=3))
                _seen: set = set()
                _end_sess_obj.web_results = [
                    r for r in _end_web
                    if r.get("url", "") not in _seen and not _seen.add(r.get("url", ""))
                ]
            except Exception:
                _end_sess_obj.web_results = []

            # Step 6: Pre-compute coping strategies & solutions (so they don't re-run on every page load)
            _end_status.info("(○'ω'○)  Preparing coping strategies & solutions...")
            _end_progress.progress(88, text="Preparing strategies...")
            _emo_for_search = _end_ctx.get("best_emotion", "neutral")
            try:
                _coping_ctx = _end_ctx.get("transcript", "") or _end_sess_obj.summary or ""
                _end_sess_obj.coping_strategies = search_coping_strategies(
                    _emo_for_search, max_results=3, context=_coping_ctx
                )
            except Exception:
                _end_sess_obj.coping_strategies = []
            try:
                _end_transcript = _end_ctx.get("transcript", "")
                # Build context including video-detected face emotions so LLM
                # knows what the camera saw (not just the transcript text)
                _face_emos = _end_ctx.get("face_emotions", [])
                _voice_emos = _end_ctx.get("voice_emotions", [])
                _sensor_ctx = ""
                if _face_emos:
                    # Summarize face emotions as percentage breakdown
                    _fc: dict[str, int] = {}
                    for _fe in _face_emos:
                        _fc[_fe] = _fc.get(_fe, 0) + 1
                    _ft = len(_face_emos)
                    _fb = ", ".join(f"{e}: {round(100*c/_ft)}%" for e, c in
                                    sorted(_fc.items(), key=lambda x: -x[1]))
                    _sensor_ctx += f"\nVideo camera emotion breakdown: {_fb} (over {_ft} readings)"
                if _voice_emos:
                    _sensor_ctx += f"\nVoice tone detected emotions: {_voice_emos}"
                _problem_text = (_end_transcript or _end_sess_obj.summary) + _sensor_ctx
                _end_sess_obj.suggested_solutions = llm.suggest_solutions(
                    _emo_for_search,
                    _problem_text,
                    face_emotion=_emo_for_search,
                )
            except Exception:
                _end_sess_obj.suggested_solutions = ""

            # Step 7: Done!
            _end_status.info("(ﾉ◕ヮ◕)ﾉ*:・ﾟ✧  All done! Here are your results~")
            _end_progress.progress(100, text="Complete!")

            st.session_state.diary_history.append(_end_sess_obj)
            st.session_state.last_ended_session = _end_sess_obj
            _persist_diary_session(_end_sess_obj)
            st.session_state.diary_session = None
            st.session_state.last_diary_audio = None
            import time as _end_time
            _end_time.sleep(0.5)  # brief pause so user sees completion
            st.rerun()

    # ── Active session panel ──
    if st.session_state.diary_session is not None:
        _sess = st.session_state.diary_session
        st.info(f"Session **{_sess.session_id}** active since {_sess.start_time} — "
                f"{len(_sess.entries)} entries recorded")

        # ── Split video + audio streams ──
        # Two independent WebRTC streams for reliability:
        #   1) Video-only stream at 640x480 for emotion detection (presentation quality)
        #   2) Audio-only stream for voice recording
        # Both write to _diary_buffer independently.
        _diary_video_ctx = None
        _diary_audio_ctx = None
        _diary_video_ok = False
        _diary_audio_ok = False

        if _webrtc_imports_ok and DiaryVideoProcessor is not None and DiaryAudioProcessor is not None:
            # Only clear shared buffer ONCE when a new session starts
            if _diary_buffer is not None and st.session_state.get("_diary_buffer_needs_clear", False):
                _diary_buffer.clear()
                st.session_state._diary_buffer_needs_clear = False

            # ── Video stream (emotion detection) ──
            st.markdown("##### Emotion Camera")
            st.caption("Press START to monitor your facial emotions in real time.")
            try:
                _diary_video_ctx = _webrtc_streamer(
                    key=f"diary-video-{_sess.session_id}",
                    rtc_configuration=rtc_config,
                    video_processor_factory=DiaryVideoProcessor,
                    media_stream_constraints={
                        "video": {"width": {"ideal": 640}, "height": {"ideal": 480}},
                        "audio": False,
                    },
                    async_processing=True,
                )
                _diary_video_ok = True
            except Exception:
                _diary_video_ok = False

        # ── Audio recording (st.audio_input — works without HTTPS) ──
        st.markdown("##### Voice Recorder")
        st.caption("Click the microphone icon to record your voice diary.")
        _audio_input_bytes = st.audio_input(
            "Record your diary entry",
            key=f"diary_audio_input_{_sess.session_id}",
        )
        # Track if new audio arrived
        _has_new_audio = (
            _audio_input_bytes is not None
            and _audio_input_bytes != st.session_state.get("_last_audio_input_id")
        )
        if _has_new_audio:
            st.session_state["_last_audio_input_id"] = _audio_input_bytes
            st.session_state["_diary_audio_bytes"] = _audio_input_bytes.read()
            _audio_input_bytes.seek(0)  # reset for playback
            st.audio(_audio_input_bytes, format="audio/wav")

        # Track whether video stream is active
        _any_stream_active = False
        if _diary_video_ok and _diary_video_ctx is not None:
            if _diary_video_ctx.state.playing:
                _any_stream_active = True
        st.session_state["_diary_stream_active"] = _any_stream_active

        # Live status panel: face emotion + video indicator
        _diary_streams_ok = (_diary_video_ok and _diary_buffer is not None)
        if _diary_streams_ok:

            @st.fragment(run_every=2.0)
            def _diary_av_live_panel():
                _video_playing = (
                    _diary_video_ctx is not None and _diary_video_ctx.state.playing
                )
                if not _video_playing:
                    return

                # Read emotion from the video processor directly (most up-to-date),
                # with shared buffer as fallback.
                _emo, _conf = "neutral", 0.0
                if (
                    _diary_video_ctx is not None
                    and _diary_video_ctx.video_processor is not None
                ):
                    _emo = _diary_video_ctx.video_processor.emotion
                    _conf = _diary_video_ctx.video_processor.confidence
                else:
                    _emo, _conf = _diary_buffer.get_current_emotion()
                st.session_state.diary_face_emotion_live = _emo
                st.session_state.diary_face_confidence_live = _conf

                # ── Append to diary emotion history (one per second) ──
                now_str = datetime.now().strftime("%H:%M:%S")
                _deh = st.session_state.diary_emotion_history
                last_t = _deh[-1]["time"] if _deh else ""
                if now_str != last_t:
                    _deh.append({"time": now_str, "emotion": _emo, "confidence": _conf})
                    st.session_state.diary_emotion_history = _deh[-120:]

                c1, c2 = st.columns(2)
                c1.metric("Face Emotion", f"{_emo.upper()}", f"{_conf:.0%}")
                _rec_emo = _diary_buffer.emotion_during_recording
                c2.metric("Session Mood", _rec_emo["emotion"].upper(), f"{_rec_emo['confidence']:.0%}")

                # ── Live emotion history chart ──
                _hist = st.session_state.diary_emotion_history[-60:]
                if _hist:
                    try:
                        import plotly.graph_objects as go
                        _emo2num = {e: i for i, e in enumerate(EMOTION_LABELS)}
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=[h["time"] for h in _hist],
                            y=[_emo2num.get(h["emotion"], 6) for h in _hist],
                            mode="lines+markers",
                            name="Face Emotion",
                            text=[f"{h['emotion']} ({h['confidence']:.0%})" for h in _hist],
                            line=dict(color="#FF6B6B"),
                            marker=dict(size=5),
                        ))
                        fig.update_layout(
                            title="Diary Emotion Timeline (Live)",
                            yaxis=dict(
                                ticktext=EMOTION_LABELS,
                                tickvals=list(range(len(EMOTION_LABELS))),
                            ),
                            height=280,
                            margin=dict(l=10, r=10, t=35, b=10),
                        )
                        st.plotly_chart(fig, width='stretch')
                    except ImportError:
                        st.caption("Install plotly for emotion history chart.")

            _diary_av_live_panel()

        # ── Save Entry (uses audio from st.audio_input + emotion from video) ──
        _audio_bytes_for_save = st.session_state.get("_diary_audio_bytes")
        _has_recorded_audio = _audio_bytes_for_save is not None and len(_audio_bytes_for_save) > 100

        if _has_recorded_audio:
            st.success("Audio recorded. Click **Save Entry** to process.")

        if st.button("💾 Save Entry", type="primary", width='stretch',
                      key="diary_save_av_entry"):
            if _has_recorded_audio:
                with st.spinner("Processing audio + face emotions..."):
                    _rec_emo = _diary_buffer.emotion_during_recording if _diary_buffer else {"emotion": "neutral", "confidence": 0.5}
                    _timeline_snap = list(st.session_state.diary_emotion_history)
                    _entry = _diary_mgr.add_entry(
                        _sess,
                        _audio_bytes_for_save,
                        face_emotion=_rec_emo.get("emotion", st.session_state.diary_face_emotion_live),
                        face_confidence=_rec_emo.get("confidence", st.session_state.diary_face_confidence_live),
                        face_emotion_timeline=_timeline_snap,
                    )
                if _entry:
                    # Clear audio for next entry
                    st.session_state["_diary_audio_bytes"] = None
                    st.session_state["_last_audio_input_id"] = None
                    if _diary_buffer:
                        _diary_buffer.clear()
                    st.session_state.diary_emotion_history = []
                    st.success(f"Entry added — Fused: **{_entry.fused_emotion}** ({_entry.fused_confidence:.0%})")
                    st.rerun()
                else:
                    st.warning("Failed to process entry.")
            else:
                st.warning("No audio recorded yet. Use the Voice Recorder above to record, then save.")

        # ── Fallback: manual emotion selector when video unavailable ──
        if not _diary_video_ok:
            with st.expander("Manual Emotion Override"):
                _face_emo = st.selectbox(
                    "Face Emotion (manual)",
                    EMOTION_LABELS,
                    index=EMOTION_LABELS.index(st.session_state.diary_face_emotion_live),
                    key="diary_face_emotion_fallback",
                )
                st.session_state.diary_face_emotion_live = _face_emo

        # ── Session timeline ──
        if _sess.entries:
            st.divider()
            st.subheader("Session Timeline")
            for _i, _e in enumerate(_sess.entries):
                _e_obj = _e if isinstance(_e, DiaryEntry) else None
                _ts = _e_obj.timestamp if _e_obj else _e.get("timestamp", "")
                _txt = _e_obj.text if _e_obj else _e.get("text", "")
                _femo = _e_obj.fused_emotion if _e_obj else _e.get("fused_emotion", "neutral")
                _fconf = _e_obj.fused_confidence if _e_obj else _e.get("fused_confidence", 0.5)
                with st.expander(f"Entry {_i+1} at {_ts} — {_femo} ({_fconf:.0%})"):
                    st.markdown(f"**Transcription:** {_txt}")
                    if _e_obj:
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Face", f"{_e_obj.face_emotion} ({_e_obj.face_confidence:.0%})")
                        c2.metric("Voice", _e_obj.voice_sentiment.get("emotion", "neutral"))
                        c3.metric("Audio", _e_obj.audio_emotion.get("estimated_emotion", "neutral"))
                        # Show saved face emotion timeline chart if available
                        _ftl = getattr(_e_obj, "face_emotion_timeline", []) or []
                        if _ftl:
                            try:
                                import plotly.graph_objects as go
                                _e2n = {e: i for i, e in enumerate(EMOTION_LABELS)}
                                _fig_e = go.Figure()
                                _fig_e.add_trace(go.Scatter(
                                    x=[h.get("time", "") if isinstance(h, dict) else "" for h in _ftl],
                                    y=[_e2n.get(h.get("emotion", "neutral") if isinstance(h, dict) else "neutral", 6) for h in _ftl],
                                    mode="lines+markers",
                                    name="Face Emotion",
                                    text=[f"{h.get('emotion','?')} ({h.get('confidence',0):.0%})" if isinstance(h, dict) else "" for h in _ftl],
                                    line=dict(color="#FF6B6B"),
                                    marker=dict(size=4),
                                ))
                                _fig_e.update_layout(
                                    title="Face Emotion During Entry",
                                    yaxis=dict(ticktext=EMOTION_LABELS, tickvals=list(range(len(EMOTION_LABELS)))),
                                    height=220,
                                    margin=dict(l=10, r=10, t=30, b=10),
                                )
                                st.plotly_chart(_fig_e, width='stretch')
                            except ImportError:
                                pass
    else:
        if st.session_state.last_ended_session is None:
            st.info("Click **Start New Session** to begin recording your voice diary.")

    # ── Session History ──
    st.divider()
    st.subheader("Session History")
    if st.session_state.diary_history:
        for _sh in reversed(st.session_state.diary_history):
            _sh_id = _sh.session_id if isinstance(_sh, DiarySession) else _sh.get("session_id", "?")
            _sh_time = _sh.start_time if isinstance(_sh, DiarySession) else _sh.get("start_time", "")
            _sh_entries = _sh.entries if isinstance(_sh, DiarySession) else _sh.get("entries", [])
            _sh_best = _get_best_emotion(_sh)
            _sh_transcript = _get_session_transcript(_sh)

            with st.expander(f"Session {_sh_id} — {_sh_time} ({len(_sh_entries)} entries, {_sh_best})"):
                # Compassionate response
                _comp_resp = getattr(_sh, "compassionate_response", "") if isinstance(_sh, DiarySession) else _sh.get("compassionate_response", "")
                if _comp_resp:
                    st.info(_comp_resp)

                # Summary
                _summary = _sh.summary if isinstance(_sh, DiarySession) else _sh.get("summary", "")
                if _summary:
                    st.markdown(f"**Summary:** {_summary}")

                # LLM solutions (pre-computed during end_session — no re-call on each render)
                _sh_solutions = (
                    getattr(_sh, "suggested_solutions", "")
                    if isinstance(_sh, DiarySession)
                    else _sh.get("suggested_solutions", "")
                )
                if _sh_solutions:
                    st.markdown("**Suggested Solutions:**")
                    st.markdown(_sh_solutions)

                # Coping strategies (pre-computed during end_session)
                _sh_coping = (
                    getattr(_sh, "coping_strategies", [])
                    if isinstance(_sh, DiarySession)
                    else _sh.get("coping_strategies", [])
                )
                if _sh_coping:
                    st.markdown("**Coping Strategies:**")
                    for _sc in _sh_coping:
                        st.markdown(f"- [{_sc['title']}]({_sc['url']}): {_sc['snippet'][:100]}...")

                # ArXiv results
                _arxiv_res = getattr(_sh, "arxiv_results", []) if isinstance(_sh, DiarySession) else _sh.get("arxiv_results", [])
                if _arxiv_res:
                    st.markdown("**ArXiv Papers:**")
                    for _ap in _arxiv_res[:5]:
                        _ap_title = _ap.get("title", "Untitled")
                        _ap_url = _ap.get("url", "")
                        _ap_authors = ", ".join(_ap.get("authors", [])[:3]) if isinstance(_ap.get("authors"), list) else str(_ap.get("authors", ""))
                        st.markdown(f"- [{_ap_title}]({_ap_url}) — {_ap_authors}")

                # Web results
                _web_res = getattr(_sh, "web_results", []) if isinstance(_sh, DiarySession) else _sh.get("web_results", [])
                if _web_res:
                    st.markdown("**Web Results:**")
                    for _wr in _web_res[:5]:
                        st.markdown(f"- [{_wr.get('title', '')}]({_wr.get('url', '')}): {_wr.get('snippet', '')[:100]}...")

                # Download + Delete
                if isinstance(_sh, DiarySession):
                    _md = _sh.to_markdown()
                else:
                    # Build markdown from dict-based session
                    _md_lines = [f"# Diary Session: {_sh_id}", f"**Started:** {_sh_time}", ""]
                    if _sh.get("summary"): _md_lines.append(f"## Summary\n{_sh['summary']}\n")
                    if _sh.get("compassionate_response"): _md_lines.append(f"## Supportive Response\n{_sh['compassionate_response']}\n")
                    _md_lines.append("## Entries\n")
                    for _me in _sh.get("entries", []):
                        _md_lines.append(f"### Entry at {_me.get('timestamp', '')}")
                        _md_lines.append(f"**Text:** {_me.get('text', '')}")
                        _md_lines.append(f"**Fused:** {_me.get('fused_emotion', 'neutral')} ({_me.get('fused_confidence', 0):.0%})")
                        _md_lines.append("")
                    _md = "\n".join(_md_lines)
                _btn_dl, _btn_del = st.columns(2)
                with _btn_dl:
                    st.download_button(
                        "Download Session (.md)",
                        data=_md,
                        file_name=f"diary_{_sh_id}.md",
                        mime="text/markdown",
                        key=f"dl_{_sh_id}",
                        width='stretch',
                    )
                with _btn_del:
                    if st.button("Delete Session", key=f"del_{_sh_id}", width='stretch'):
                        memory.delete_session(_sh_id)
                        st.session_state.diary_history = [
                            s for s in st.session_state.diary_history
                            if (s.session_id if isinstance(s, DiarySession) else s.get("session_id")) != _sh_id
                        ]
                        st.rerun()
    else:
        st.info("No diary sessions yet. Start one above!")

    # ── Talk About Your Feelings chat (for past sessions / general use) ──
    # Only show this section when NOT in post-session mode (chat is shown inline above)
    if st.session_state.last_ended_session is None:
        st.divider()
        st.subheader("Talk About Your Feelings")
        if st.session_state.diary_history:
            _last_sess = st.session_state.diary_history[-1]
            st.caption("Continue chatting about your feelings based on your most recent session.")

            for _msg in st.session_state.diary_chat_history:
                with st.chat_message(_msg["role"]):
                    st.markdown(_msg["content"])

            if _diary_input := st.chat_input("Share what's on your mind...", key="diary_chat_input"):
                st.session_state.diary_chat_history.append({"role": "user", "content": _diary_input})
                with st.chat_message("user"):
                    st.markdown(_diary_input)

                with st.chat_message("assistant"):
                    with st.spinner("Listening..."):
                        _best_emo = _get_best_emotion(_last_sess)
                        _session_ctx = {
                            "summary": _last_sess.summary if isinstance(_last_sess, DiarySession) else _last_sess.get("summary", ""),
                            "dominant_emotion": _best_emo,
                            "compassionate_response": getattr(_last_sess, "compassionate_response", "") if isinstance(_last_sess, DiarySession) else _last_sess.get("compassionate_response", ""),
                            "transcript": _get_session_transcript(_last_sess),
                        }
                        try:
                            _resp = llm.compassionate_chat(
                                _diary_input,
                                st.session_state.diary_chat_history,
                                _session_ctx,
                            )
                        except Exception:
                            _resp = llm.chat(
                                f"[User emotion: {_best_emo}] {_diary_input}",
                                st.session_state.diary_chat_history,
                            )
                        st.markdown(_resp)
                        st.session_state.diary_chat_history.append({"role": "assistant", "content": _resp})
                        # Persist diary chat
                        _last_sid = _last_sess.session_id if isinstance(_last_sess, DiarySession) else _last_sess.get("session_id", "")
                        _persist_chat(st.session_state.diary_chat_history, session_id=_last_sid)
        else:
            st.info("Complete a diary session first to unlock the compassionate chat.")


# ════════════════════════════════════════════════════════════════
# TAB 5: Chat
# ════════════════════════════════════════════════════════════════
with tab_chat:
    st.header("Chat with EmotiScan AI")
    current_emo = st.session_state.detected_emotion
    st.caption(f"Context: your current detected emotion is **{current_emo}**")

    # Clear chat history button
    if st.button("🗑️ Clear Chat History", key="clear_general_chat"):
        st.session_state.chat_history = []
        try:
            memory.clear_chat_history(session_id="")
        except Exception as _e:
            logging.getLogger(__name__).warning("Failed to clear chat history: %s", _e)
        st.rerun()

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if user_input := st.chat_input("Ask about emotions, research, anything..."):
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Use compassionate chat if diary sessions exist
                if st.session_state.diary_history:
                    _last = st.session_state.diary_history[-1]
                    _ctx = {
                        "summary": _last.summary if isinstance(_last, DiarySession) else _last.get("summary", ""),
                        "dominant_emotion": _get_best_emotion(_last),
                        "compassionate_response": getattr(_last, "compassionate_response", "") if isinstance(_last, DiarySession) else _last.get("compassionate_response", ""),
                        "transcript": _get_session_transcript(_last),
                    }
                    try:
                        response = llm.compassionate_chat(user_input, st.session_state.chat_history, _ctx)
                    except Exception:
                        context_msg = f"[User's current emotion: {current_emo}] {user_input}"
                        response = llm.chat(context_msg, st.session_state.chat_history)
                else:
                    context_msg = f"[User's current emotion: {current_emo}] {user_input}"
                    response = llm.chat(context_msg, st.session_state.chat_history)
                st.markdown(response)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                # Persist general chat
                _persist_chat(st.session_state.chat_history, session_id="")


# ════════════════════════════════════════════════════════════════
# TAB 6: About / Status
# ════════════════════════════════════════════════════════════════
with tab_about:
    st.header("About EmotiScan")
    st.code(STATIC_HEADER, language=None)

    # Service health
    st.subheader("Service Health")
    health_cols = st.columns(4)
    with health_cols[0]:
        st.metric("Ollama", "ONLINE" if ollama_up else "OFFLINE")
    with health_cols[1]:
        st.metric("MCP Ranker", "ONLINE" if mcp_up else "OFFLINE")
    with health_cols[2]:
        st.metric("GPU", gpu_info["name"] if gpu_info["available"] else "None")
    with health_cols[3]:
        webcam_status = "Available" if webrtc_available else "Manual Mode"
        st.metric("Webcam", webcam_status)

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""### Tech Stack
| Component | Technology |
|---|---|
| **Emotion Detection** | YOLOv8 + FER |
| **LLM** | Qwen3 8B (Ollama) |
| **Papers** | ArXiv API (MCP) |
| **Web Search** | DuckDuckGo |
| **Ranking** | TF-IDF / MCP |
| **Voice Diary** | faster-whisper STT |
| **Mood Fusion** | Face+Voice+Text |
| **UI** | Streamlit + WebRTC |
| **Charts** | Plotly |
| **ASCII Art** | Numpy patterns |
| **Memory** | SQLite (WAL mode) |
""")
    with c2:
        st.markdown("""### Features
- Real-time emotion detection via webcam
- Emotion-aware research digest
- ASCII art mood boosters
- Voice diary with multimodal emotion detection
- Compassionate AI counselor chat
- AI chat with emotion context
- Web + ArXiv paper search
- Local TF-IDF & MCP ranking
- Dynamic numpy-based art generation
- Persistent memory across restarts (SQLite)
""")

    st.divider()
    gpu_md = "### GPU Configuration\n"
    if gpu_info["available"]:
        gpu_md += f"| Property | Value |\n|---|---|\n"
        gpu_md += f"| **GPU** | {gpu_info['name']} |\n"
        gpu_md += f"| **VRAM** | {gpu_info['vram']} |\n"
        gpu_md += f"| **Layers** | {num_gpu_layers} |\n"
    else:
        gpu_md += "No NVIDIA GPU detected. CPU-only mode.\n"
    st.markdown(gpu_md)

    # ── Persistent Memory Stats ──
    st.divider()
    st.subheader("Persistent Memory")
    try:
        _mem_stats = memory.get_stats()
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Saved Sessions", _mem_stats["total_sessions"])
        mc2.metric("Diary Entries", _mem_stats["total_entries"])
        mc3.metric("Chat Messages", _mem_stats["total_chat_messages"])
        mc4.metric("Emotion Snapshots", _mem_stats["total_emotion_snapshots"])
        st.caption(f"Data stored in: `{DATA_DIR}`")

        # Data management buttons
        _mgmt1, _mgmt2, _mgmt3 = st.columns(3)
        with _mgmt1:
            if st.button("Clear Diary History", width='stretch'):
                for _s in memory.load_all_sessions():
                    memory.delete_session(_s["session_id"])
                st.session_state.diary_history = []
                st.session_state.diary_chat_history = []
                st.session_state.last_ended_session = None
                st.success("All diary history cleared from disk.")
                st.rerun()
        with _mgmt2:
            if st.button("Clear Chat History", width='stretch'):
                memory.clear_chat_history(session_id="")
                st.session_state.chat_history = []
                st.success("Chat history cleared from disk.")
                st.rerun()
        with _mgmt3:
            if st.button("🗑️ Purge Emotion DB", width='stretch',
                         help="Clear saved emotion detection data (fixes stale/corrupted readings)"):
                memory.clear_emotion_history(source="detection")
                memory.clear_emotion_history(source="diary")
                st.session_state.emotion_history = []
                st.session_state.diary_emotion_history = []
                st.success("(๑’ω’)๑  Emotion database purged! Fresh start~")
                st.rerun()
    except Exception as _e:
        st.warning(f"Memory store unavailable: {_e}")
