#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

image="${CHATTERBOX_IMAGE:-chatterbox-docker:smoke}"
container="${CHATTERBOX_CONTAINER:-chatterbox-docker-smoke}"
port="${CHATTERBOX_SMOKE_PORT:-8088}"

cleanup() {
  docker rm -f "$container" >/dev/null 2>&1 || true
}
trap cleanup EXIT

docker build -t "$image" .
cleanup
docker run -d \
  --name "$container" \
  -e CHATTERBOX_DEVICE=cpu \
  -e CHATTERBOX_SKIP_MODEL_LOAD=1 \
  -p "127.0.0.1:${port}:80" \
  "$image" >/dev/null

for _ in {1..45}; do
  if curl -fsS "http://127.0.0.1:${port}/healthz" >/tmp/chatterbox-healthz.json 2>/dev/null; then
    cat /tmp/chatterbox-healthz.json
    printf "\nChatterbox container smoke passed at http://127.0.0.1:%s/healthz\n" "$port"
    exit 0
  fi
  sleep 1
done

docker logs "$container" >&2 || true
echo "Timed out waiting for Chatterbox /healthz" >&2
exit 1
