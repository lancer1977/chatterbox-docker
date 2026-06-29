#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

required_paths=(
  "README.md"
  ".env.example"
  "code_health.md"
  "docs/project-atlas/README.md"
  "templates/README.md"
  ".devstudio/project.yaml"
  "Dockerfile"
  "requirements.txt"
  "main.py"
  "main.mac.py"
  "install.sh"
  "run.sh"
  "scripts/smoke.sh"
  "docs/deploy/README.md"
)

for path in "${required_paths[@]}"; do
  if [[ ! -e "$path" ]]; then
    echo "Missing required path: $path" >&2
    exit 1
  fi
done

bash -n install.sh
bash -n run.sh
bash -n scripts/smoke.sh
python3 -m py_compile main.py main.mac.py

if command -v devstudio >/dev/null 2>&1; then
  devstudio validate --repo "$repo_root"
else
  echo "devstudio not available; skipping DevStudio validation"
fi
