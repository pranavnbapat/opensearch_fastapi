#!/usr/bin/env sh
set -eu

# Start server in background
ollama serve &
SERVER_PID=$!

# Wait until it's up
until wget -qO- http://127.0.0.1:11434/api/tags >/dev/null 2>&1; do
  sleep 0.5
done

# Pull model (idempotent)
ollama pull "${OLLAMA_MODEL:-qwen2.5:3b}"

# Warm model into RAM with a tiny prompt (forces initial load)
wget -qO- --header="Content-Type: application/json" \
  --post-data "{\"model\":\"${OLLAMA_MODEL:-qwen2.5:3b}\",\"stream\":false,\"messages\":[{\"role\":\"user\",\"content\":\"warmup\"}]}" \
  http://127.0.0.1:11434/api/chat >/dev/null 2>&1 || true

# Bring server to foreground
wait "$SERVER_PID"
