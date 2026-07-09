#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/code"

PYTHON="$ROOT/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="python3"
fi

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Created code/.env from .env.example"
  echo "Edit code/.env and set LLM_PROVIDER plus the matching API key."
fi

set -a
# shellcheck disable=SC1091
source .env
set +a

if [[ "${LLM_PROVIDER:-dummy}" == "dummy" ]]; then
  echo "Warning: LLM_PROVIDER=dummy — answers will be placeholders."
  echo "Set LLM_PROVIDER=ollama|gemini|openai in code/.env for real outputs."
fi

exec "$PYTHON" -m uvicorn app:app --host 127.0.0.1 --port 8000
