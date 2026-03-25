#!/bin/bash
# Strix Halo LLM Gateway startup

set -e

LLAMA_BIN="$HOME/llama.cpp/build-vulkan/bin/llama-server"
PRESET="$HOME/models/models.ini"
GATEWAY="$HOME/gateway.py"

echo "=== Strix Halo LLM Gateway ==="

# Kill any existing instances
pkill -f "llama-server.*models-preset" 2>/dev/null && echo "Stopped old llama-server" || true
pkill -f "gateway.py" 2>/dev/null && echo "Stopped old gateway" || true
sleep 2

# Enable extra swap if not active
if ! swapon --show | grep -q swapfile2; then
    sudo swapon /swapfile2 2>/dev/null && echo "Enabled extra swap" || true
fi

# Start llama-server router
echo "Starting llama-server router on :8080..."
nohup "$LLAMA_BIN" \
    --models-preset "$PRESET" \
    --models-max 2 \
    --host 127.0.0.1 --port 8080 \
    &>/tmp/llama-router.log &
LLAMA_PID=$!

# Wait for router to be ready
for i in $(seq 1 30); do
    if curl -s http://127.0.0.1:8080/v1/models >/dev/null 2>&1; then
        echo "  Router ready (PID $LLAMA_PID)"
        break
    fi
    sleep 1
done

# Start gateway
echo "Starting gateway on :9090..."
nohup python3 "$GATEWAY" &>/tmp/gateway.log &
GW_PID=$!
sleep 2

if curl -s http://localhost:9090/api/stats >/dev/null 2>&1; then
    echo "  Gateway ready (PID $GW_PID)"
else
    echo "  WARNING: Gateway may not be ready yet"
fi

# Summary
MODELS=$(curl -s http://127.0.0.1:8080/v1/models 2>/dev/null | python3 -c "import sys,json;[print(f'  - {m[\"id\"]} ({m[\"status\"][\"value\"]})') for m in json.load(sys.stdin).get('data',[])]" 2>/dev/null)

echo ""
echo "=== Ready ==="
echo "Dashboard: http://$(hostname -I | awk '{print $1}'):9090"
echo "API:       http://$(hostname -I | awk '{print $1}'):9090/v1"
echo "Models:"
echo "$MODELS"
echo ""
echo "Logs: /tmp/llama-router.log, /tmp/gateway.log"
