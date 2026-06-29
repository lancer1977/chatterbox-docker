#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

image_ref="${CHATTERBOX_SMOKE_IMAGE:-chatterbox-docker:smoke}"
container_name="${CHATTERBOX_SMOKE_CONTAINER:-chatterbox-docker-smoke}"
port="${CHATTERBOX_SMOKE_PORT:-18080}"
url="http://127.0.0.1:${port}"

cleanup() {
  docker rm -f "$container_name" >/dev/null 2>&1 || true
}

trap cleanup EXIT

docker rm -f "$container_name" >/dev/null 2>&1 || true

docker run -d \
  --rm \
  --name "$container_name" \
  -e CHATTERBOX_SKIP_MODEL_LOAD=1 \
  -e CHATTERBOX_DEVICE=cpu \
  -p "127.0.0.1:${port}:80" \
  "$image_ref" >/dev/null

python3 - <<'PY' "$url"
import json
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen

url = sys.argv[1]
health_url = f"{url}/health"
root_url = f"{url}/"

for _ in range(60):
    try:
        with urlopen(health_url, timeout=2) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if payload.get("status") == "ok":
            break
    except URLError:
        time.sleep(1)
else:
    raise SystemExit(f"health check never became ready at {health_url}")

with urlopen(health_url, timeout=2) as response:
    payload = json.loads(response.read().decode("utf-8"))

assert payload["status"] == "ok", payload
assert payload["skip_model_load"] is True, payload
assert payload["model_loaded"] is False, payload

with urlopen(root_url, timeout=2) as response:
    payload = json.loads(response.read().decode("utf-8"))

assert payload["health"] == "/health", payload
assert payload["tts"] == "/tts", payload
PY

echo "Smoke probe passed for $image_ref"
