#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
if ! command -v ollama >/dev/null; then
  echo "Install Ollama first: https://ollama.com/download"; exit 1
fi
[ -d agent-env ] || python3 -m venv agent-env
source agent-env/bin/activate
pip install -q --default-timeout 100 -r requirements.txt
MODEL="${SCHOLAR_MODEL:-llama3.2:3b}"
ollama list | grep -q "$MODEL" || ollama pull "$MODEL"
python server.py "$@"