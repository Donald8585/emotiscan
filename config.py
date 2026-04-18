"""
Central configuration for EmotiScan.
All constants, ports, model names, feature flags in one place.
"""

# ── Ollama / LLM ────────────────────────────────────────────────
# OLLAMA_URL can be overridden via env var for Docker:
#   - Local dev:  http://localhost:8003  (default)
#   - Docker:     http://host.docker.internal:8003  (reaches host Ollama)
#   - Compose:    http://ollama:8003  (if using ollama service in compose)
import os as _os_cfg
OLLAMA_URL = _os_cfg.environ.get("OLLAMA_URL",
             _os_cfg.environ.get("OLLAMA_HOST", "http://localhost:8003"))
OLLAMA_MODEL = _os_cfg.environ.get("OLLAMA_MODEL", "qwen3:8b")
OLLAMA_TIMEOUT = 120.0
OLLAMA_NUM_CTX = 4096
OLLAMA_TEMPERATURE = 0.7

# ── MCP Servers ─────────────────────────────────────────────────
# MCP_RANKER_HOST/PORT overridable via env for Docker networking
MCP_RANKER_HOST = _os_cfg.environ.get("MCP_RANKER_HOST", "localhost")
MCP_RANKER_PORT = int(_os_cfg.environ.get("MCP_RANKER_PORT", "8000"))
MCP_RANKER_URL = f"http://{MCP_RANKER_HOST}:{MCP_RANKER_PORT}/mcp"

# MCP_ARXIV_HOST/PORT overridable via env for Docker networking
MCP_ARXIV_HOST = _os_cfg.environ.get("MCP_ARXIV_HOST", "localhost")
MCP_ARXIV_PORT = int(_os_cfg.environ.get("MCP_ARXIV_PORT", "8001"))
MCP_ARXIV_URL = f"http://{MCP_ARXIV_HOST}:{MCP_ARXIV_PORT}/mcp"

# ── YOLO / Emotion Detection ───────────────────────────────────
YOLO_MODEL = "yolov8n-face.pt"  # Ultralytics YOLOv8-face model (auto-downloads)
YOLO_FACE_CONF = 0.40           # Min confidence for YOLO face detection
YOLO_FACE_IOU = 0.50            # NMS IoU threshold
EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
NEGATIVE_EMOTIONS = {"angry", "disgust", "fear", "sad"}
POSITIVE_EMOTIONS = {"happy", "surprise"}
NEUTRAL_EMOTIONS = {"neutral"}

# ── GPU ─────────────────────────────────────────────────────────
GPU_LAYERS = 99  # 0 = CPU-only, 99 = all layers on GPU

# ── GPU Auto-Detection ──────────────────────────────────────────
GPU_AVAILABLE = False
GPU_DEVICE_NAME = "cpu"
try:
    import torch as _torch
    if _torch.cuda.is_available():
        GPU_AVAILABLE = True
        GPU_DEVICE_NAME = _torch.cuda.get_device_name(0)
except ImportError:
    pass

if not GPU_AVAILABLE:
    try:
        import tensorflow as _tf
        _gpus = _tf.config.list_physical_devices('GPU')
        if _gpus:
            GPU_AVAILABLE = True
            GPU_DEVICE_NAME = _gpus[0].name
    except ImportError:
        pass

# Device string for YOLO / PyTorch models — "cuda" if GPU, else "cpu"
YOLO_DEVICE = "cuda" if GPU_AVAILABLE else "cpu"
WHISPER_DEVICE = "cuda" if GPU_AVAILABLE else "cpu"
WHISPER_COMPUTE_TYPE = "float16" if GPU_AVAILABLE else "int8"

# ── Web Search ──────────────────────────────────────────────────
WEB_SEARCH_MAX_RESULTS = 5
WEB_SEARCH_CACHE_TTL = 300  # seconds

# ── WebRTC ──────────────────────────────────────────────────────
WEBRTC_STUN_SERVER = "stun:stun.l.google.com:19302"

# ── Streamlit ───────────────────────────────────────────────────
PAGE_TITLE = "EmotiScan"
PAGE_ICON = "🎭"
LAYOUT = "wide"

# ── Emotion → research topic mapping ───────────────────────────
EMOTION_RESEARCH_TOPICS = {
    "happy": "positive psychology happiness neuroscience",
    "sad": "sadness depression coping mechanisms psychology",
    "angry": "anger management emotional regulation neuroscience",
    "surprise": "surprise emotion startle response psychology",
    "fear": "fear anxiety amygdala neuroscience",
    "disgust": "disgust emotion moral psychology evolutionary",
    "neutral": "emotional baseline mindfulness resting state brain",
}

# ── Voice / STT ──────────────────────────────────────────────────
WHISPER_MODEL = "base"  # tiny/base/small for faster-whisper
STT_SAMPLE_RATE = 16000
STT_MAX_DURATION = 120  # max seconds per recording

# ── Mood fusion weights ─────────────────────────────────────────
FACE_EMOTION_WEIGHT = 0.5
TEXT_SENTIMENT_WEIGHT = 0.3
AUDIO_EMOTION_WEIGHT = 0.2

# ── Diary ────────────────────────────────────────────────────────
MAX_DIARY_ENTRIES = 50
DIARY_AUTO_RESEARCH = True

# ── Persistent Storage ───────────────────────────────────────────
DATA_DIR = _os_cfg.environ.get("EMOTISCAN_DATA_DIR", _os_cfg.path.join(_os_cfg.path.dirname(_os_cfg.path.abspath(__file__)), "emotiscan_data"))
DB_FILE = _os_cfg.path.join(DATA_DIR, "emotiscan.db")
AUTO_SAVE = True  # auto-persist diary sessions, chat, and emotion history

# ── Query expansion for research digest ─────────────────────────
QUERY_EXPANSIONS = {
    "llm": "large language models LLM transformer GPT attention",
    "rag": "retrieval augmented generation RAG vector search embedding",
    "nlp": "natural language processing NLP text understanding",
    "ml": "machine learning ML deep learning neural network",
    "cv": "computer vision image recognition CNN convolutional",
    "rl": "reinforcement learning reward policy agent environment",
    "mcp": "model context protocol MCP tool integration agent server",
    "ai": "artificial intelligence AI deep learning neural network",
}
