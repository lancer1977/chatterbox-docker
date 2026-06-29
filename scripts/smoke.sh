#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

echo "[smoke] running repo validation"
bash scripts/validate.sh

if command -v docker >/dev/null 2>&1; then
  echo "[smoke] checking local compose shape"
  docker compose -f deploy/docker-compose.local.yml config >/dev/null
else
  echo "[smoke] docker not available; skipping compose config check"
fi
