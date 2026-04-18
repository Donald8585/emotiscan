# ═══════════════════════════════════════════════════════════════════════
# EmotiScan Docker Image
# Multi-stage build: works on CPU (school PCs) + optional NVIDIA GPU
# ═══════════════════════════════════════════════════════════════════════

FROM python:3.11-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies (OpenCV, audio, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install Python dependencies ──────────────────────────────────
# Copy requirements first for Docker layer caching
COPY requirements.txt .

# Install PyTorch — CPU or CUDA depending on build arg
#   CPU build (~200MB):  docker compose up              (default)
#   CUDA build (~2GB):   docker compose --profile gpu up
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir \
    torch torchvision --index-url ${TORCH_INDEX}

# Install remaining requirements
# Pin timm<1.0 (critical for hsemotion compatibility)
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application code ────────────────────────────────────────
COPY streamlit_app.py .
COPY config.py .
COPY services/ services/
COPY mcp/ mcp/
COPY diary/ diary/
COPY ui/ ui/
COPY setup_gpu.sh .
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Copy tests (optional, for verification)
COPY tests/ tests/

# ── Pre-download models (so first run is fast) ───────────────────
RUN python -c "\
from huggingface_hub import hf_hub_download; \
hf_hub_download(repo_id='arnabdhar/YOLOv8-Face-Detection', filename='model.pt'); \
print('YOLOv8 face model downloaded')" || echo "YOLOv8 model will download on first run"

# Pre-download HSEmotion model
RUN python -c "\
import torch, functools; \
_orig = torch.load; \
torch.load = lambda *a, **kw: _orig(*a, **{**kw, 'weights_only': kw.get('weights_only', False)}); \
from hsemotion.facial_emotions import HSEmotionRecognizer; \
HSEmotionRecognizer(model_name='enet_b0_8_best_afew', device='cpu'); \
print('HSEmotion model downloaded')" || echo "HSEmotion model will download on first run"

# ── Create data directories for persistence ──────────────────────
RUN mkdir -p /app/data

# ── Streamlit configuration ──────────────────────────────────────
RUN mkdir -p /root/.streamlit
RUN echo '\
[server]\n\
port = 80\n\
address = "0.0.0.0"\n\
headless = true\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
maxUploadSize = 200\n\
\n\
[browser]\n\
gatherUsageStats = false\n\
\n\
[theme]\n\
primaryColor = "#FF6B6B"\n\
' > /root/.streamlit/config.toml

EXPOSE 80 8002

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:80/_stcore/health || exit 1

# ── Default: start Streamlit ─────────────────────────────────────
CMD ["./entrypoint.sh"]
