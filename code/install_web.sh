#!/usr/bin/env bash
set -euo pipefail

pip install fastapi
pip install uvicorn
pip install python-dotenv

if [[ ! -f "$(dirname "$0")/.env" ]]; then
  cp "$(dirname "$0")/.env.example" "$(dirname "$0")/.env"
  echo "Created code/.env — edit it to set LLM_PROVIDER and API key."
fi
