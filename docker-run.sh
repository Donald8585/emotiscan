#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# EmotiScan - Docker Launcher (Linux/Mac/Git Bash)
# Usage: bash docker-run.sh           (CPU mode)
#        bash docker-run.sh --gpu     (with NVIDIA GPU)
#        bash docker-run.sh --ollama  (include Ollama container)
# ═══════════════════════════════════════════════════════════════

set -e

echo ""
echo "╔═══════════════════════════════════════╗"
echo "║     EmotiScan Docker Launcher         ║"
echo "╚═══════════════════════════════════════╝"
echo ""

# Parse arguments — default to CPU profile
PROFILES="--profile cpu"
SERVICE="emotiscan"
for arg in "$@"; do
    case $arg in
        --gpu)
            PROFILES="--profile gpu"
            SERVICE="emotiscan-gpu"
            echo "  → GPU mode enabled"
            ;;
        --ollama)
            PROFILES="$PROFILES --profile ollama"
            echo "  → Ollama container enabled"
            ;;
    esac
done

# Check Docker
if ! docker info >/dev/null 2>&1; then
    echo "[ERROR] Docker is not running!"
    echo "Please start Docker Desktop or install Docker first."
    exit 1
fi

echo "[1/3] Building EmotiScan image (first run takes ~5-10 min)..."
docker compose $PROFILES build $SERVICE

echo ""
echo "[2/3] Starting EmotiScan..."
docker compose $PROFILES up -d

echo ""
echo "[3/3] Waiting for app to start..."
sleep 8

echo ""
echo "═══════════════════════════════════════════"
echo "  EmotiScan is running!"
echo "  Open: http://localhost:80"
echo ""
echo "  To stop:  docker compose $PROFILES down"
echo "  Logs:     docker compose $PROFILES logs -f"
echo "═══════════════════════════════════════════"
echo ""

# Try to open browser (works on most systems)
if command -v xdg-open &>/dev/null; then
    xdg-open http://localhost:80 2>/dev/null || true
elif command -v open &>/dev/null; then
    open http://localhost:80 2>/dev/null || true
elif command -v start &>/dev/null; then
    start http://localhost:80 2>/dev/null || true
fi
