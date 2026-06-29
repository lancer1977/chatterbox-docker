# Code Health

## Current state

- CUDA-backed Python container for Chatterbox.
- CI validates Python/shell syntax and DevStudio shape.
- Docker image build is intentionally not the default gate because the current image pulls multi-gigabyte NVIDIA CUDA base layers.

## Validation

- `bash scripts/validate.sh`

## Follow-ups

- Add a lightweight CPU/test Dockerfile or split CUDA image build into an explicit release workflow.
