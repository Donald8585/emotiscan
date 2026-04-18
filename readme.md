# EmotiScan 🧠💫

**Real-Time Emotion-Aware Research Mood Booster**

EmotiScan is a multimodal, emotion-aware Streamlit app that detects your mood in real time (face + voice + text), then feeds that context into an LLM agent that pulls research papers, web articles, coping strategies, and ASCII-art mood boosters tailored to how you're feeling. Built with local LLMs (Ollama), YOLOv8 + HSEmotion for face emotion, faster-whisper for speech, and a small MCP tool layer for paper searching and ranking.

---

## ✨ Features

- **Real-time face emotion detection** via webcam (YOLOv8-face + HSEmotion), async detection thread so the video feed never lags.
- **Voice diary** with `faster-whisper` STT, audio emotion, and multimodal mood fusion (face + voice + text).
- **Emotion-aware research agent** — pulls ArXiv papers and DuckDuckGo web results, ranks them with a local TF-IDF MCP ranker, and summarizes with an LLM.
- **Compassionate AI counselor chat** that uses your detected emotion as context.
- **ASCII-art mood boosters** (static + dynamic numpy-generated patterns).
- **Persistent memory** via SQLite (WAL mode) — diary sessions, chat history, emotion timeline.
- **GPU auto-detect** (CUDA / TF) with graceful CPU fallback.
- **Fully dockerized** with optional NVIDIA GPU support.

---

## 🧱 Tech Stack

| Component            | Technology                         |
| -------------------- | ---------------------------------- |
| UI                   | Streamlit + streamlit-webrtc       |
| Face Emotion         | YOLOv8-face + HSEmotion (timm)     |
| Speech-to-Text       | faster-whisper                     |
| LLM                  | Qwen3-8B via Ollama                |
| Papers               | ArXiv API (MCP tool)               |
| Web Search           | DuckDuckGo                         |
| Ranking              | TF-IDF over MCP (FastMCP)          |
| Charts               | Plotly                             |
| Storage              | SQLite (WAL)                       |
| Packaging            | Docker + docker-compose            |

---

## 📦 Project Structure
```text
emotiscan/
├── streamlit_app.py # Main Streamlit UI (tabs: Emotion, Research, Mood, Diary, Chat, About)
├── config.py # Central config: ports, models, weights, feature flags
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── docker-run.sh / docker-run.bat
├── entrypoint.sh
├── setup_gpu.sh
├── services/
│ ├── emotion_detector.py # YOLOv8 + HSEmotion face emotion pipeline
│ ├── speech_service.py # faster-whisper STT
│ ├── llm_service.py # Ollama client + prompts
│ ├── web_search_service.py # DuckDuckGo + caching
│ ├── memory_store.py # SQLite persistence
│ └── mood_fusion.py # Face + voice + text fusion
├── mcp/
│ ├── mcp_client.py
│ ├── ranker_tool_server.py # TF-IDF ranker (FastMCP)
│ └── arxiv_search_tool.py
├── diary/
│ └── diary_session.py
└── ui/
  └── ascii_art_generator.py
```

---

## 🚀 Quickstart

### Option 1 — Docker (recommended)

```bash
git clone https://github.com/Donald8585/emotiscan.git
cd emotiscan
docker compose up --build
```

Then open **http://localhost:8501**.

- GPU users: make sure NVIDIA Container Toolkit is installed, then run `./setup_gpu.sh` once before `docker compose up`.
- Ollama is expected on `http://localhost:8003` (override with `OLLAMA_URL` env var).

### Option 2 — Local Python

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt

# 1) Start the MCP ranker (separate terminal)
python mcp/ranker_tool_server.py

# 2) Start Ollama locally and pull the model
ollama serve
ollama pull qwen3:8b

# 3) Run the app
streamlit run streamlit_app.py
```

App runs on **http://localhost:8501**.

---

## ⚙️ Configuration

All knobs live in `config.py` and are overridable via environment variables:

| Variable            | Default                         | Purpose                          |
| ------------------- | ------------------------------- | -------------------------------- |
| `OLLAMA_URL`        | `http://localhost:8003`         | Ollama server endpoint           |
| `OLLAMA_MODEL`      | `qwen3:8b`                      | LLM for chat / research          |
| `MCP_RANKER_HOST`   | `localhost`                     | TF-IDF ranker host               |
| `MCP_RANKER_PORT`   | `8000`                          | TF-IDF ranker port               |
| `MCP_ARXIV_HOST`    | `localhost`                     | ArXiv MCP tool host              |
| `MCP_ARXIV_PORT`    | `8001`                          | ArXiv MCP tool port              |
| `EMOTISCAN_DATA_DIR`| `./emotiscan_data`              | SQLite + logs directory          |
| `GPU_LAYERS`        | `99`                            | 0 = CPU only, 99 = all on GPU    |

Mood fusion weights (face 0.5 / text 0.3 / audio 0.2) live in `config.py` and can be tuned.

---

## 🧪 Tabs Overview

- **Emotion Detection** — live webcam with async YOLOv8 + HSEmotion inference and an emotion history chart.
- **Research Digest** — emotion-aware topic suggestions, ArXiv + web search, TF-IDF ranking, LLM-generated digest.
- **Mood Booster** — ASCII-art therapy (static + dynamic numpy patterns).
- **Voice Diary** — WebRTC audio+video capture, faster-whisper STT, multimodal emotion fusion, compassionate chat.
- **Chat** — general LLM chat with current emotion injected as context.
- **About / Status** — service health (Ollama, MCP ranker, GPU), version info.

---

## 🖥️ Hardware

- **CPU only:** works, but HSEmotion + Whisper will be slow.
- **GPU:** auto-detected via `torch.cuda` / TensorFlow. CUDA is used for YOLOv8, HSEmotion, and Whisper (`float16`).

---

## 🔐 Privacy

- All inference runs locally (Ollama, YOLO, HSEmotion, Whisper).
- Only external calls are **ArXiv API** and **DuckDuckGo** for search.
- Diary, chat, and emotion history are stored locally in SQLite inside `EMOTISCAN_DATA_DIR`.

---

## 🛠️ Development

```bash
pytest
```

PRs welcome. Style: black + reasonable type hints. Keep all hard-coded constants in `config.py`.

---

## 📄 License

MIT.

---

## 👥 Authors

|:---:|:---:|:---:|
| [<img src="https://github.com/Donald8585.png" width="60px;"/><br /><sub><a href="https://github.com/Donald8585">Donald8585</a></sub>](https://github.com/Donald8585) | [<img src="https://github.com/dannytong2010.png" width="60px;"/><br /><sub><a href="https://github.com/dannytong2010">dannytong2010</a></sub>](https://github.com/dannytong2010) | [<img src="https://github.com/Iris-Yan99.png" width="60px;"/><br /><sub><a href="https://github.com/Iris-Yan99">Iris-Yan99</a></sub>](https://github.com/Iris-Yan99) |
| So Chit Wai, Alfred | Tong Chin Pang, Danny | Zhen Churou, Iris |

---

## 🙏 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [HSEmotion](https://github.com/HSE-asavchenko/face-emotion-recognition)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [Ollama](https://ollama.com/)
- [Streamlit](https://streamlit.io/) & [streamlit-webrtc](https://github.com/whitphx/streamlit-webrtc)
- [FastMCP](https://github.com/jlowin/fastmcp)
