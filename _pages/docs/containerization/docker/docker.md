---
layout: post
title: "Docker: containerize a Python project"
nav_order: 2
---

Docker is a practical way to package code and dependencies into a single runnable unit. This page focuses on building images and running containers. For orchestrating services locally with Docker Compose, see [Docker Compose: orchestrate local development]({{ site.baseurl }}/docs/containerization/docker-compose/docker-compose/).

---

## On this page

- [Key concepts](#key-concepts)
- [Quick start: build and run](#quick-start-build-and-run)
  - [Workflow](#workflow)
  - [Build an image: CI/CD artifact](#build-an-image-cicd-artifact)
  - [Run a container](#run-a-container)
  - [Override the command](#override-the-command)
- [Resources](#resources)
- [The reference Dockerfile](#the-reference-dockerfile)
  - [Build arguments](#build-arguments)
  - [Builder stage: install dependencies into a local venv](#builder-stage-install-dependencies-into-a-local-venv)
  - [Runtime stage: minimal image, non-root user](#runtime-stage-minimal-image-non-root-user)
- [Best practices](#best-practices)
- [Logs and debugging](#logs-and-debugging)

---

## Key concepts

- **Image vs container**: an image is the immutable artifact you build; a container is a running instance of that image.
- **Layers and cache**: each Dockerfile instruction produces a layer. If inputs do not change, Docker can reuse cached layers to speed up rebuilds.
- **Multi-stage builds**: build dependencies in a "builder" stage, then copy only what you need into a minimal "runtime" stage. This reduces image size and attack surface.
- **Volumes**: volumes are managed storage that outlives containers. Use them for data you must keep (for example databases, model artifacts you cache intentionally, or local dev state). Avoid writing important data into the container filesystem, which is disposable by default.

---

## Quick start: build and run

The reference Dockerfile at `pages/docs/containerization/docker/Dockerfile.example` expects a typical Python project layout at the project root: `pyproject.toml`, a dependency lock file (for `uv`, this is `uv.lock`), a Python package directory (controlled via the `PACKAGE_NAME` build argument), and `main.py` as the entrypoint. If your project differs, adapt `PACKAGE_NAME` and/or the container `CMD` (see [Override the command](#override-the-command)).

The Dockerfile uses BuildKit mounts (`RUN --mount=...`). If BuildKit is not already enabled, prefix builds with `DOCKER_BUILDKIT=1`. Add a `.dockerignore` at the project root to keep builds fast and avoid leaking local artifacts into images (an example is provided at `pages/docs/containerization/docker/.dockerignore.example`).

### Workflow

The reference Dockerfile uses a lockfile-driven install, a multi-stage build, a non-root runtime, and optional read-only runtime flags. These choices produce images that are reproducible, minimal, and hardened by default.

Three contexts drive the workflow:

- **Local dev**: use Compose for fast iteration with consistent flags (see [Docker Compose: orchestrate local development]({{ site.baseurl }}/docs/containerization/docker-compose/docker-compose/)).
- **CI/CD**: build a deterministic artifact, attach metadata, scan, and publish.
- **Runtime**: run with a production-like posture (least privilege, explicit writable paths).

#### CI/CD: build, scan, promote

Tag images so every artifact is traceable:

- **Immutable tag**: commit SHA (for example `sha-<git-sha>`)
- **Moving pointers**: `dev`, `staging`, `prod` (or semver release tags)

Pipeline stages:

- **Build** from the repo root (so `pyproject.toml` and the lock are in the build context)
- **Attach metadata** (timestamp, commit SHA, repo URL — see [build arguments](#build-arguments))
- **Scan / SBOM / policy checks**
- **Push** the immutable SHA tag
- **Promote** by retagging only after validation

The image you run in staging/prod should be the same digest you built and validated in CI.

### Build an image: CI/CD artifact

This Dockerfile uses a build argument named `PACKAGE_NAME` that must match the folder you want to copy into the image (your top-level Python package directory).

Use this when you want a deterministic artifact for CI and deployment. Enterprise workflows typically run security controls immediately after build (policy checks, vulnerability scan, SBOM generation), then publish the image to a registry.

Run the command from the project root so the build context includes `pyproject.toml`, your lock file, and your package directory.

```bash
docker build \
  -f pages/docs/containerization/docker/Dockerfile.example \
  -t <image>:<tag> \
  --build-arg PACKAGE_NAME=<package_dir> \
  .
```

Optional build arguments:

- `--build-arg PYTHON_VERSION=3.13`
- `--build-arg UV_IMAGE=ghcr.io/astral-sh/uv:latest`
- `--build-arg UID=90001`
- `--build-arg PORT=8080`

In enterprise environments, also consider:

- Pin base images by digest for repeatable builds (for example `python:3.13-slim@sha256:<digest>`).
- Pin tooling images by digest (the reference exposes `UV_IMAGE` so you can set it to a version tag or digest in CI).
- Run a vulnerability scan and produce an SBOM in CI, then sign images before publishing.

### Run a container

Run this when you want a quick "does it start" check without compose.

```bash
docker run --rm \
  -p 8080:8080 \
  -e ENVIRONMENT=prod \
  -e LOG_LEVEL=INFO \
  -e PORT=8080 \
  <image>:<tag>
```

If you want a production-like posture, add hardened runtime defaults:

```bash
docker run --rm \
  --init \
  --read-only \
  --cap-drop=ALL \
  --security-opt=no-new-privileges \
  --tmpfs /tmp:rw,nosuid,nodev,noexec,size=256m \
  --tmpfs /app/logs:rw,nosuid,nodev,noexec,size=64m \
  -p 8080:8080 \
  -e HOME=/tmp \
  -e XDG_CACHE_HOME=/tmp/.cache \
  -e XDG_CONFIG_HOME=/tmp/.config \
  -e ENVIRONMENT=prod \
  -e LOG_LEVEL=INFO \
  -e PORT=8080 \
  <image>:<tag>
```

### Override the command

The reference image uses a virtual environment at `/app/.venv` and runs `main.py`. You can override the command at runtime:

Use this when you want different entrypoints for dev, batch jobs, or troubleshooting.

```bash
docker run --rm <image>:<tag> /app/.venv/bin/python -m <module_or_package>
```

---

## Resources

Use these templates when you want a baseline quickly, then customize for your application.

<details class="boz-resource">
  <summary><code>.dockerignore.example</code></summary>

{% highlight text %}
{% include_relative .dockerignore.example %}
{% endhighlight %}
</details>

<details class="boz-resource">
  <summary><code>Dockerfile.example</code></summary>

{% highlight dockerfile %}
{% include_relative Dockerfile.example %}
{% endhighlight %}
</details>

---

## The reference Dockerfile

The reference implementation is `pages/docs/containerization/docker/Dockerfile.example`. It is designed around three goals:

- **Reproducibility**: install dependencies from a lock file.
- **Fast rebuilds**: cache dependency downloads.
- **Security**: run as a non-root user in the runtime image.

### Build arguments

Build arguments let you reuse the same Dockerfile across environments and projects. In this reference file they control:

```dockerfile
ARG PYTHON_VERSION=3.13
ARG UV_IMAGE=ghcr.io/astral-sh/uv:latest
ARG UID=90001
ARG PORT=8080
ARG PACKAGE_NAME=myapp
ARG BUILD_DATE=""
ARG VCS_REF=""
ARG VCS_URL=""
```

- **Python base image pin**: `PYTHON_VERSION`
- **Tooling image pin**: `UV_IMAGE` (set this to a version tag or digest in CI for repeatability)
- **Runtime user id**: `UID`
- **Container port**: `PORT` (used in the `EXPOSE` directive)
- **Which package directory to copy**: `PACKAGE_NAME`
- **Image build timestamp (traceability)**: `BUILD_DATE` (used for `org.opencontainers.image.created`)
- **Source revision (traceability)**: `VCS_REF` (used for `org.opencontainers.image.revision`). `VCS` stands for Version Control System (commonly Git).
- **Source repository URL (traceability)**: `VCS_URL` (used for `org.opencontainers.image.source`)

Use them whenever you need portability (for example: building the same project with a different Python minor version in CI), without duplicating Dockerfiles.

In enterprise CI, those traceability arguments are typically populated from the pipeline context so every published image can be linked back to an exact commit and repository:

```bash
docker build \
  -f pages/docs/containerization/docker/Dockerfile.example \
  -t <image>:<tag> \
  --build-arg PACKAGE_NAME=<package_dir> \
  --build-arg UV_IMAGE="ghcr.io/astral-sh/uv:<version-or-digest>" \
  --build-arg BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --build-arg VCS_REF="$(git rev-parse HEAD)" \
  --build-arg VCS_URL="$(git remote get-url origin)" \
  .
```

### Builder stage: install dependencies into a local venv

```dockerfile
FROM ${UV_IMAGE} AS uv
FROM python:${PYTHON_VERSION}-slim AS builder
COPY --from=uv /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-editable
```

This stage enforces a reproducibility contract: dependency installation is locked, and the layer inputs are intentionally narrow so application code changes do not invalidate the dependency cache.

The `PYTHONDONTWRITEBYTECODE` and `PYTHONUNBUFFERED` settings make container behavior more predictable: no `.pyc` files written during execution and logs emitted immediately (useful for platforms that collect stdout/stderr).

The mounts are doing two different jobs:

- `--mount=type=cache,target=/root/.cache/uv` keeps the dependency download cache across builds. That means rebuilds spend time reusing already-downloaded wheels/artifacts instead of re-fetching them, which is one of the biggest practical speedups in CI and on developer machines.
- `--mount=type=bind,source=...` makes only the dependency inputs available to that layer (for example `pyproject.toml` and a lock file). This keeps the layer's inputs intentionally small, so changes to application code do not invalidate the dependency-install layer.

If your dependencies require access to private resources, keep credentials out of the image and mount them only for the install step:

- **Private Python indexes**: pass a BuildKit secret and mount it for the `RUN` step that installs dependencies.

  ```bash
  docker build \
    -f pages/docs/containerization/docker/Dockerfile.example \
    -t <image>:<tag> \
    --build-arg PACKAGE_NAME=<package_dir> \
    --secret id=pip_conf,src=./pip.conf \
    .
  ```

  ```dockerfile
  RUN --mount=type=secret,id=pip_conf,target=/etc/pip.conf \
      --mount=type=cache,target=/root/.cache/uv \
      --mount=type=bind,source=uv.lock,target=uv.lock \
      --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
      uv sync --locked --no-install-project --no-editable
  ```

- **Private Git dependencies over SSH**: forward an SSH agent and mount it only for the install step.

  ```bash
  docker build \
    -f pages/docs/containerization/docker/Dockerfile.example \
    -t <image>:<tag> \
    --build-arg PACKAGE_NAME=<package_dir> \
    --ssh default \
    .
  ```

  ```dockerfile
  RUN --mount=type=ssh \
      --mount=type=cache,target=/root/.cache/uv \
      --mount=type=bind,source=uv.lock,target=uv.lock \
      --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
      uv sync --locked --no-install-project --no-editable
  ```

Avoid passing secrets via `ARG`/`ENV` or copying credential files into the image.

The command `uv sync --locked --no-install-project` reinforces the same intent: install only third-party dependencies first. Once that layer is cached, code edits rebuild quickly because Docker can reuse the dependency layer and only redo the later "copy code + install project" step.

Then the Dockerfile copies the project files and installs the project itself:

```dockerfile
COPY pyproject.toml uv.lock ./
COPY ${PACKAGE_NAME} ./${PACKAGE_NAME}
COPY main.py ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-editable
```

This step makes the "what changes invalidate which layers" trade-off explicit. By copying only the dependency metadata and the selected package directory, you preserve cache stability and make the image's runtime surface area reviewable.

### Runtime stage: minimal image, non-root user

```dockerfile
FROM python:${PYTHON_VERSION}-slim

RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

WORKDIR /app

COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
COPY --from=builder --chown=appuser:appuser /app/${PACKAGE_NAME} /app/${PACKAGE_NAME}
COPY --from=builder --chown=appuser:appuser /app/main.py /app/main.py

RUN mkdir -p /app/logs && chown -R appuser:appuser /app/logs

USER appuser
```

The runtime stage starts from a fresh base image so build-only artifacts do not leak into production.

- A non-privileged user is created and later used to run the application.
- This is a baseline security best practice: if something goes wrong inside the app, it should not have root privileges.

Only the minimum runtime artifacts are copied from the builder stage:

This stage should be intentionally minimal: copy the prebuilt environment and the application code, then drop privileges. The reference also uses ownership-aware copies (via `--chown=...`) so the runtime user can read and write where needed without permission surprises.

Finally, environment defaults and the runtime command are defined:

```dockerfile
ENV ENVIRONMENT=dev
ENV LOG_LEVEL=INFO

EXPOSE ${PORT}
CMD ["/app/.venv/bin/python", "main.py"]
```

Environment defaults make the container self-describing and easy to configure in different environments. The `ENV` lines define sensible defaults you can override at runtime, `EXPOSE ${PORT}` documents the expected container port (defaulting to 8080), and the venv-based `CMD` ensures the locked environment is the one being executed.

---

## Best practices

- **Keep images small**: multi-stage builds and `python:<version>-slim` reduce size.
- **Prefer lockfiles**: installs should be deterministic and reviewable.
- **Run as non-root**: the reference Dockerfile already does this.
- **Harden runtime permissions**: prefer a read-only filesystem, drop Linux capabilities by default, prevent privilege escalation, and mount `tmpfs` for explicit writable paths (commonly `/tmp` and an app-specific directory).
- **Treat images as release artifacts**: scan for vulnerabilities, generate an SBOM, and sign images before promotion across environments.
- **Log to stdout/stderr**: containers are easier to operate when logs go to the platform.
- **Make ports configurable**: use `PORT` and map with `-p host:container`.

### Common patterns

These patterns show up constantly in real systems. The goal is to make them explicit, so you avoid "mystery state" and inconsistent runs.

#### Model and artifact caching

If your service downloads models or artifacts at startup, decide deliberately where they live:

- **Ephemeral cache** (safe default): write to `/tmp` and accept that containers are disposable.
- **Named volume cache** (intentional performance trade-off): mount a named volume so repeated starts are faster, then document cache invalidation.

When you enable `--read-only` / `read_only: true`, also route caches to writable paths (commonly under `/tmp`) via environment variables such as `HOME`, `XDG_CACHE_HOME`, and `XDG_CONFIG_HOME`.

#### Datasets and "inputs"

Treat input data as an external dependency:

- **Local dev**: bind-mount a read-only dataset directory (so you do not accidentally write into it).
- **CI and prod**: pull inputs from object storage or a data platform, not from inside the image.

Avoid copying datasets into images. It makes builds slow, images huge, and data lifecycle management unclear.

#### GPUs

If you need GPUs, keep the container contract simple:

- Make GPU usage **optional** (the same image should run CPU-only).
- Prefer runtime configuration (for example `--gpus all`) over building separate "gpu" images unless you truly need different system dependencies.

---

## Logs and debugging

Containers are easiest to operate when you treat them as stateless processes and capture all logs and signals from the outside.

### Follow logs

Stream logs continuously from a running container:

```bash
docker logs -f <container-name-or-id>
```

### Debug a running container

Shell into a running container to inspect the filesystem, check environment variables, or run one-off commands:

```bash
docker exec -it <container> /bin/sh
```

Useful checks from inside the container:

- **Environment**: `env | sort` (verify variables are set as expected)
- **Processes**: `ps aux` (confirm the app is running as `appuser`, not root)
- **Filesystem**: `ls -la /app` (check permissions and files copied from the builder stage)
- **Network**: `cat /etc/hosts`, `hostname -i` (verify container networking)

### Inspect from outside

- **What is running**: `docker ps` (add `-a` to include stopped containers)
- **What images exist locally**: `docker images`
- **Resource usage**: `docker stats`
- **Stop a container**: `docker stop <container>`
- **Remove a container**: `docker rm <container>` (add `-f` to force)

### Logging conventions

- **Write logs to stdout/stderr**: avoid writing app logs to local files inside the container unless you have a clear persistence plan.
- **Use structured logs when possible**: JSON logs are easier to parse and ship to centralized logging systems.
- **Promote log level to config**: keep `LOG_LEVEL` (or equivalent) as an environment variable so you can increase verbosity without rebuilding.
- **Add a healthcheck for uptime signals**: a `HEALTHCHECK` makes failures visible to orchestrators and to `docker ps`.
- **Expose metrics and traces explicitly**: if you use Prometheus/OpenTelemetry, document the port and enablement env vars, then publish those ports only where needed.
