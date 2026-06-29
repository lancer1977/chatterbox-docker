# Chatterbox Docker Release

This repo publishes a container image and deploys it to any Docker host or
Portainer stack that pulls the published tag.

## Image

- Registry: `ghcr.io/lancer1977/chatterbox-docker`
- Release tags:
  - `latest` for the current main branch
  - the immutable commit SHA for rollback

## Build

```bash
docker build -t chatterbox-docker:release .
```

## Smoke probe

Run the repo-local probe after the image is built:

```bash
CHATTERBOX_SMOKE_IMAGE=chatterbox-docker:release bash scripts/smoke.sh
```

The probe starts the container with `CHATTERBOX_SKIP_MODEL_LOAD=1`, checks
`/health`, and confirms the root route advertises the health and TTS endpoints.

## Runtime configuration

- `CHATTERBOX_DEVICE` sets the TTS runtime device, defaulting to `cpu`.
- `CHATTERBOX_SKIP_MODEL_LOAD=1` skips the model load for health and smoke runs.
- `DEBUG=1` enables debug logging.

The container writes generated WAV files under `/app/completed` and expects
speaker prompt files under `/app/audio_prompts`.

## Deploy target

The release image is meant for a Docker host or Portainer stack that pulls the
tagged GHCR image and mounts any required audio prompt or output volumes.

## Rollback

Redeploy the previous immutable SHA tag from GHCR. If the `latest` tag must be
rewound, retag the previous SHA locally or in the release pipeline and push it
again after smoke passes.
