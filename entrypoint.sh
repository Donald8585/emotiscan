#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# EmotiScan Docker Entrypoint
# Starts the MCP ranker server in the background, then launches Streamlit.
# ═══════════════════════════════════════════════════════════════════════
set -e

echo "╔══════════════════════════════════════════════╗"
echo "║  EmotiScan Starting...  (ﾉ◕ヮ◕)ﾉ*:・ﾟ✧       ║"
echo "╚══════════════════════════════════════════════╝"

# ── Start MCP ArXiv Search Server in background ─────────────────
echo "[1/3] Starting MCP ArXiv Search on port 8001..."
python mcp/arxiv_search_tool.py &
ARXIV_PID=$!

# ── Start MCP Ranker Server in background ────────────────────────
echo "[2/3] Starting MCP Ranker on port ${MCP_RANKER_PORT:-8002}..."
python mcp/ranker_tool_server.py &
RANKER_PID=$!

# Wait for both servers to be ready (up to 10 seconds)
for i in $(seq 1 10); do
    ARXIV_OK=0; RANKER_OK=0
    python -c "import socket; s=socket.create_connection(('localhost', 8001), timeout=1); s.close()" 2>/dev/null && ARXIV_OK=1
    python -c "import socket; s=socket.create_connection(('localhost', ${MCP_RANKER_PORT:-8002}), timeout=1); s.close()" 2>/dev/null && RANKER_OK=1
    [ $ARXIV_OK -eq 1 ] && [ $RANKER_OK -eq 1 ] && echo "  ✓ MCP ArXiv Search ready (PID: $ARXIV_PID)" && echo "  ✓ MCP Ranker ready (PID: $RANKER_PID)" && break
    sleep 1
done

# ── Check Ollama connectivity ────────────────────────────────────
# Try multiple Ollama locations (host.docker.internal > ollama service > localhost)
OLLAMA="${OLLAMA_URL:-}"
if [ -z "$OLLAMA" ]; then
    # Auto-detect: try host.docker.internal first (host Ollama), then compose service
    for url in "http://host.docker.internal:8003" "http://ollama:8003" "http://localhost:8003"; do
        if curl -s --max-time 2 "$url/api/tags" > /dev/null 2>&1; then
            export OLLAMA_URL="$url"
            OLLAMA="$url"
            break
        fi
    done
    if [ -z "$OLLAMA" ]; then
        # Default to host.docker.internal (most common Docker setup)
        export OLLAMA_URL="http://host.docker.internal:8003"
        OLLAMA="$OLLAMA_URL"
    fi
fi
echo "[3/3] Checking Ollama at $OLLAMA ..."
if curl -s --max-time 3 "$OLLAMA/api/tags" > /dev/null 2>&1; then
    echo "  OK  Ollama is reachable - LLM features enabled"
else
    echo "  !!  Ollama not reachable - LLM features disabled"
    echo "      To enable LLM: start Ollama on your host machine,"
    echo "      or use: docker compose --profile ollama up"
fi

echo ""
echo "Starting Streamlit on port ${STREAMLIT_SERVER_PORT:-80}..."
echo "Open http://localhost:${STREAMLIT_SERVER_PORT:-80} in your browser"
echo ""

# ── Start Streamlit (foreground) ─────────────────────────────────
exec streamlit run streamlit_app.py
