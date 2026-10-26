#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")/.."

if [ ! -f artifacts/latest/experiment.json ]; then
  uv run rankguard-train --quick
fi
(cd frontend && npm install && npm run build)
uv run rankguard-api
