#!/usr/bin/env bash
set -euo pipefail

base_url="${1:-http://127.0.0.1:8080}"
base_url="${base_url%/}"

curl -fsS "$base_url/openapi.json" >/dev/null
echo "Chatterbox smoke passed for $base_url"
