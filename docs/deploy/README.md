# Chatterbox Docker Deployment

Chatterbox Docker publishes a CUDA-backed FastAPI TTS container. The canonical
image destination is GitHub Container Registry:

```text
ghcr.io/lancer1977/chatterbox-docker:latest
ghcr.io/lancer1977/chatterbox-docker:<commit-sha>
```

## Build

Validate the repository contract before building:

```bash
bash scripts/validate.sh
docker build -t ghcr.io/lancer1977/chatterbox-docker:local .
```

The Docker build downloads the CUDA base image, Python dependencies, and
PyTorch CUDA wheels, so use a host with enough disk and network capacity.

## Runtime Configuration

The container listens on port 80 and expects optional speaker prompt files under
`/app/audio_prompts`.

Common runtime variables:

- `CHATTERBOX_DEVICE`: `cuda` in the production image, `cpu` only for local
  non-GPU experiments.
- `DEBUG`: set to `1`, `true`, or `yes` for verbose application logging.

No repository secrets are required for the build workflow. GHCR publishing uses
the built-in `GITHUB_TOKEN` with `packages: write`.

## Deploy

Deploy by pulling a successful workflow image on the target GPU host and
restarting the owned container or stack:

```bash
docker pull ghcr.io/lancer1977/chatterbox-docker:latest
docker run --rm --gpus all -p 8080:80 ghcr.io/lancer1977/chatterbox-docker:latest
```

If this image is managed by Portainer or another external stack, update the
stack image tag to the successful commit SHA rather than using a floating tag.

## Smoke Probe

After the container is listening, run:

```bash
scripts/smoke.sh http://127.0.0.1:8080
```

The smoke probe checks the FastAPI OpenAPI document without generating audio.
Use a real `/tts` request only after the model has loaded and speaker prompt
files are present.

## Rollback

Rollback by redeploying the previous successful commit SHA image from GHCR and
rerunning the smoke probe against the target service URL.
