#!/bin/bash
# Double-click in Finder (or run ./agent/start.command) to open the P3 agent UI.
# Starts Ollama if needed, launches the Gradio app, opens the browser. Close this window to stop.
cd "$(dirname "$0")/.."
if ! curl -s -m 2 http://127.0.0.1:11434/api/tags >/dev/null; then
  echo "Starting Ollama..."
  open -a Ollama 2>/dev/null || open ~/Applications/Ollama.app
  for i in $(seq 1 30); do curl -s -m 1 http://127.0.0.1:11434/api/tags >/dev/null && break; sleep 1; done
fi
if lsof -i :7860 -sTCP:LISTEN >/dev/null 2>&1; then
  echo "UI already running."; open http://127.0.0.1:7860; exit 0
fi
( for i in $(seq 1 60); do curl -s -m 1 http://127.0.0.1:7860 >/dev/null && { open http://127.0.0.1:7860; break; }; sleep 1; done ) &
python3 -m agent.app
