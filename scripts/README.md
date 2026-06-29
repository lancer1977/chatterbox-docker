# Chatterbox Docker Scripts

This folder contains repo-local validation helpers used by local operators and
GitHub Actions.

## Validation

Run the same check used by CI:

```bash
bash scripts/validate.sh
```

The workflow uploads `artifacts/validation.log` as the
`chatterbox-docker-validation` GitHub Actions artifact on every run, including
failures. No repository secrets are required for this validation-only artifact
path.

## Notes

- The current artifact home is GitHub Actions workflow artifacts.
- GHCR image publishing should be added only after the image name, tag policy,
  and runtime smoke gate are documented.

## Container smoke

Build and smoke the HTTP runtime without loading the TTS model:

```bash
bash scripts/smoke-container.sh
```

This verifies `GET /healthz` against the built image. Full promotion should also
exercise `POST /tts` on the GPU host with a known speaker prompt.
