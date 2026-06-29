# Deployment

`chatterbox-docker` ships a CUDA-backed container. The repo keeps the deploy
shape explicit, but the actual runtime handoff is delegated to the external
Portainer stack that consumes the image.

## Deploy target

- Image: `ghcr.io/lancer1977/chatterbox-docker:main`
- External runtime: Portainer using the copied stack skeleton in
  `deploy/portainer-stack.yml`
- Local smoke baseline: `deploy/docker-compose.local.yml`

## Operator flow

1. Build the container image from the repository root.
2. Publish the image tag used by the Portainer stack.
3. Update the stack environment values for image, host, and ports.
4. Redeploy the stack in Portainer.
5. Run the repo smoke probe after rollout.

## Commands

Build locally:

```bash
docker build -t ghcr.io/lancer1977/chatterbox-docker:main .
```

Run the repo smoke probe:

```bash
bash scripts/smoke.sh
```

Validate the compose shape used by the deploy template:

```bash
docker compose -f deploy/docker-compose.local.yml config
```

## Notes

- The repo-owned smoke probe is intentionally light. It verifies the Python and
  shell surface plus the compose definition without pulling the CUDA image.
- If you add a release workflow later, keep it aligned with this same image
  name and stack shape.
