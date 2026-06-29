# Chatterbox Docker Deployment

Chatterbox deploys as a CUDA-enabled FastAPI container that exposes HTTP on
container port `80`.

## Artifact

The image is built from `Dockerfile`. Until registry publishing is enabled, the
deployable artifact is explicitly delegated to the operator-host Docker build:

```bash
docker build -t chatterbox-docker:local .
```

When GHCR publishing is added, use `ghcr.io/lancer1977/chatterbox-docker:<tag>`
as the image destination and keep this runbook as the smoke gate.

## Runtime config

Required and optional runtime settings:

- `CHATTERBOX_DEVICE` - `cuda` for production GPU hosts, `cpu` for local smoke.
- `CHATTERBOX_SKIP_MODEL_LOAD` - set to `1` only for deployment/smoke checks that
  verify HTTP startup without downloading or loading the TTS model.
- `DEBUG` - optional debug logging switch.
- `/app/audio_prompts` - optional mounted speaker prompt directory.
- `/app/completed` - generated audio output directory.

## Validation

Run source validation:

```bash
bash scripts/validate.sh
```

Run the container smoke probe:

```bash
bash scripts/smoke-container.sh
```

The smoke script builds the image, starts it with
`CHATTERBOX_SKIP_MODEL_LOAD=1`, and verifies `GET /healthz`.

## Deploy

1. Build the image on the GPU host or pull the published image once GHCR is
   enabled.
2. Start the container with GPU access, `CHATTERBOX_DEVICE=cuda`, and persistent
   mounts for `/app/audio_prompts` and `/app/completed`.
3. Confirm startup:

   ```bash
   curl -fsS http://<host>:<port>/healthz
   ```

4. Confirm the real model path with a short `POST /tts` request using a known
   speaker prompt.

## Rollback

Redeploy the previous image tag or restore the previous host-local image. Verify
`/healthz`, then run the same short `POST /tts` request used for promotion.
