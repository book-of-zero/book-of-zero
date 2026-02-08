---
layout: post
title: "Docker: containerize a Python project"
categories: [docker, containers, deployment]
nav_order: 2
---

Docker is a practical way to package code and dependencies into a single runnable unit. That matters because you can ship the *same* environment across laptops, CI, and servers, reducing “works on my machine” and making runs easier to reproduce.

---

## On this page

- [Concepts in 2 minutes](#concepts-in-2-minutes)
- [Project assumptions](#project-assumptions)
- [Prerequisites](#prerequisites)
- [Quickstart: build and run](#quickstart-build-and-run)
  - [Local dev with compose](#local-dev-with-compose)
  - [Build an image: CI/CD artifact](#build-an-image-ci-cd-artifact)
  - [Run a container](#run-a-container)
  - [Override the command](#override-the-command)
- [The reference Dockerfile](#the-reference-dockerfile)
- [Adapting to other dependency managers](#adapting-to-other-dependency-managers)
- [Compose for local dev](#compose-for-local-dev)
- [Best practices](#best-practices)
- [Daily commands](#daily-commands)
- [Troubleshooting](#troubleshooting)

---

## Concepts in 2 minutes

- **Image vs container**: an image is the immutable artifact you build; a container is a running instance of that image.
- **Layers and cache**: each Dockerfile instruction produces a layer. If inputs do not change, Docker can reuse cached layers to speed up rebuilds.
- **Multi-stage builds**: build dependencies in a “builder” stage, then copy only what you need into a minimal “runtime” stage. This reduces image size and attack surface.
- **Volumes**: volumes are managed storage that outlives containers. Use them for data you must keep (for example databases, model artifacts you cache intentionally, or local dev state). Avoid writing important data into the container filesystem, which is disposable by default.

---

## Project assumptions

The reference Dockerfile expects a typical Python project layout:

- **Packaging**: `pyproject.toml` exists at the project root.
- **Locking**: a dependency lock exists and is used to install dependencies deterministically (for `uv`, this is `uv.lock`).
- **App code**: there is a Python package directory to copy into the image (the Dockerfile uses `PACKAGE_NAME` so you can choose it at build time).
- **Entrypoint**: `main.py` exists at the project root and is used as the container command in the example.

If your project differs, you will adapt `PACKAGE_NAME` and/or the container `CMD` (see [Adapting to other dependency managers](#adapting-to-other-dependency-managers) and [Override the command](#override-the-command)).

---

## Prerequisites

The reference Dockerfile uses BuildKit features (notably `RUN --mount=...`). Make sure:

- Docker is installed and working (`docker version`).
- BuildKit is enabled.
  - On many setups it is enabled by default.
  - If not, you can prefix builds with `DOCKER_BUILDKIT=1`.

You will also want a `.dockerignore` at the project root to keep builds fast and avoid leaking local artifacts into images.
An example is provided at `pages/docs/docker/.dockerignore.example` that you can use as a starting point (copy and adapt if it matches your repository).

### Resources (copy/paste templates)

Use these when you want a “works everywhere” baseline quickly, then customize for your app.

<details class="boz-resource">
  <summary><code>.dockerignore.example</code></summary>

```dockerignore
{% include_relative .dockerignore.example %}
```
</details>

<details class="boz-resource">
  <summary><code>Dockerfile.example</code></summary>

```dockerfile
{% include_relative Dockerfile.example %}
```
</details>

<details class="boz-resource">
  <summary><code>docker-compose.yaml.example</code></summary>

```yaml
{% include_relative docker-compose.yaml.example %}
```
</details>

<details class="boz-resource">
  <summary><code>docker-compose.hardened.yaml.example</code></summary>

```yaml
{% include_relative docker-compose.hardened.yaml.example %}
```
</details>

---

## Quickstart: build and run

This guide uses a reference Dockerfile at `pages/docs/docker/Dockerfile.example`.

Typical workflow:

- Use Compose for **local dev** (rebuilds on demand, consistent flags).
- Build an image for **CI/CD** (deterministic artifact you can scan and publish).
- Run with **hardened runtime defaults** when you want a production-like posture.

### Recommended workflow

This section connects the concepts to an operational workflow you can adopt as-is, then adapt to your stack.

#### Reference scenario

Assume you have a Python service you want to run in three contexts:

- **Local dev**: fast iteration, reproducible env, easy logs.
- **CI**: deterministic build, security checks, publish an artifact.
- **Runtime**: production-like posture, minimal permissions, explicit writable paths.

The examples in this page intentionally use:

- A **lockfile-driven install** (to keep dependencies deterministic)
- A **multi-stage build** (to keep runtime images minimal)
- A **non-root runtime** (baseline hardening)
- Optional **read-only runtime flags** (to catch unsafe writes early)

#### Day-to-day loop: local dev

This is a practical loop for local development:

- **Start the stack**: follow the steps in **Local dev with compose**.
- **Observe logs**: use the commands in [Logs and observability](#logs-and-observability).
- **Restart after changes** (when you are not bind-mounting code): `docker compose up --build --force-recreate`
- **Run a one-shot command in the same image** (migrations, a backfill, a smoke test):

```bash
docker compose run --rm <service-name> /app/.venv/bin/python -m <module>
```

- **Debug inside the container**: use the shell commands in **Daily commands**.

If your local workflow needs live code reload, prefer making that an explicit compose variant (bind-mount code + dev server command), while keeping the default path “rebuild the image” so you do not accidentally depend on host-only state.

#### CI/CD flow: build, publish, promote

The main operational goal is to produce an image that is traceable to a commit and safe to run.

Use a tagging convention that mirrors environments:

- **Immutable**: a commit SHA tag (for example `sha-<git-sha>`)
- **Moving pointers**: `dev`, `staging`, `prod` (or semver release tags)

A practical pipeline flow:

- **Build** from the repo root (so `pyproject.toml` and the lock are in the build context)
- **Attach metadata** (created timestamp, commit SHA, repo URL)
- **Scan / SBOM / policy checks**
- **Push** the immutable SHA tag
- **Promote** by retagging (or using the registry’s promotion mechanisms) only after validation

This page shows the build arguments and metadata pattern in the Dockerfile section below. The critical point is that the image you run in staging/prod should be the same digest you built and validated in CI.

### Local dev with compose

Use this when you want a standard developer experience: one command, consistent flags, and a stable place to capture local defaults.

1. Copy the examples into conventional names at the project root:
   - `pages/docs/docker/Dockerfile.example` → `Dockerfile`
   - `pages/docs/docker/docker-compose.yaml.example` → `docker-compose.yaml`
   - `pages/docs/docker/docker-compose.hardened.yaml.example` → `docker-compose.hardened.yaml` if you want a production-like posture
2. Create the env files for compose:
   - **`.env`**: used by compose for `${...}` variable substitution and passed into the container at runtime via `env_file` (application configuration).
3. Start the service:

Keep runtime secrets out of version control. Treat `.env` as local-only inputs and share templates (for example `.env.example`) when you need a documented baseline.

```bash
docker compose up --build
```

Optional production-like posture for local validation:

Use this when you want to surface hidden write assumptions early while keeping the baseline setup simple.

```bash
docker compose -f docker-compose.yaml -f docker-compose.hardened.yaml up --build
```

Minimal examples:

```text
# .env (compose variable substitution)
PACKAGE_NAME=myapp
PORT=8080
```

```text
# .env (runtime container environment)
ENVIRONMENT=dev
LOG_LEVEL=INFO
```

If you need more context on compose fields, see [Compose for local dev](#compose-for-local-dev) below.

### Build an image: CI/CD artifact

This Dockerfile uses a build argument named `PACKAGE_NAME` that must match the folder you want to copy into the image (your top-level Python package directory).

Use this when you want a deterministic artifact for CI and deployment. Enterprise workflows typically run security controls immediately after build (policy checks, vulnerability scan, SBOM generation), then publish the image to a registry.

Run the command from the project root so the build context includes `pyproject.toml`, your lock file, and your package directory.

```bash
docker build \
  -f pages/docs/docker/Dockerfile.example \
  -t <image>:<tag> \
  --build-arg PACKAGE_NAME=<package_dir> \
  .
```

Optional build arguments:

- `--build-arg PYTHON_VERSION=3.13`
- `--build-arg UV_IMAGE=ghcr.io/astral-sh/uv:latest`
- `--build-arg UID=90001`

In enterprise environments, also consider:

- Pin base images by digest for repeatable builds (for example `python:3.13-slim@sha256:<digest>`).
- Pin tooling images by digest (the reference exposes `UV_IMAGE` so you can set it to a version tag or digest in CI).
- Run a vulnerability scan and produce an SBOM in CI, then sign images before publishing.

### Run a container

Run this when you want a quick “does it start” check without compose.

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

## The reference Dockerfile

The reference implementation is `pages/docs/docker/Dockerfile.example`. It is designed around three goals:

- **Reproducibility**: install dependencies from a lock file.
- **Fast rebuilds**: cache dependency downloads.
- **Security**: run as a non-root user in the runtime image.

### Build arguments

Build arguments let you reuse the same Dockerfile across environments and projects. In this reference file they control:

```dockerfile
ARG PYTHON_VERSION=3.13
ARG UV_IMAGE=ghcr.io/astral-sh/uv:latest
ARG UID=90001
ARG PACKAGE_NAME=myapp
ARG BUILD_DATE=""
ARG VCS_REF=""
ARG VCS_URL=""
```

- **Python base image pin**: `PYTHON_VERSION`
- **Tooling image pin**: `UV_IMAGE` (set this to a version tag or digest in CI for repeatability)
- **Runtime user id**: `UID`
- **Which package directory to copy**: `PACKAGE_NAME`
- **Image build timestamp (traceability)**: `BUILD_DATE` (used for `org.opencontainers.image.created`)
- **Source revision (traceability)**: `VCS_REF` (used for `org.opencontainers.image.revision`). `VCS` stands for Version Control System (commonly Git).
- **Source repository URL (traceability)**: `VCS_URL` (used for `org.opencontainers.image.source`)

Use them whenever you need portability (for example: building the same project with a different Python minor version in CI), without duplicating Dockerfiles.

In enterprise CI, those traceability arguments are typically populated from the pipeline context so every published image can be linked back to an exact commit and repository:

```bash
docker build \
  -f pages/docs/docker/Dockerfile.example \
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
- `--mount=type=bind,source=...` makes only the dependency inputs available to that layer (for example `pyproject.toml` and a lock file). This keeps the layer’s inputs intentionally small, so changes to application code do not invalidate the dependency-install layer.

If your dependencies require access to private resources, keep credentials out of the image and mount them only for the install step:

- **Private Python indexes**: pass a BuildKit secret and mount it for the `RUN` step that installs dependencies.

  ```bash
  docker build \
    -f pages/docs/docker/Dockerfile.example \
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
    -f pages/docs/docker/Dockerfile.example \
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

The command `uv sync --locked --no-install-project` reinforces the same intent: install only third-party dependencies first. Once that layer is cached, code edits rebuild quickly because Docker can reuse the dependency layer and only redo the later “copy code + install project” step.

Then the Dockerfile copies the project files and installs the project itself:

```dockerfile
COPY pyproject.toml uv.lock ./
COPY ${PACKAGE_NAME} ./${PACKAGE_NAME}
COPY main.py ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-editable
```

This step makes the “what changes invalidate which layers” trade-off explicit. By copying only the dependency metadata and the selected package directory, you preserve cache stability and make the image’s runtime surface area reviewable.

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
ENV PORT=8080

EXPOSE ${PORT}
CMD ["/app/.venv/bin/python", "main.py"]
```

Environment defaults make the container self-describing and easy to configure in different environments. In the reference, the `ENV` lines define sensible defaults you can override at runtime, `EXPOSE ${PORT}` documents the expected container port, and the venv-based `CMD` ensures the locked environment is the one being executed.

## Adapting to other dependency managers

The important invariant is the same regardless of tooling:

- **Build**: install dependencies from a lock file into an isolated environment.
- **Runtime**: copy only the resulting environment and your application code into a minimal image.

### Poetry pattern

- **Inputs**: `pyproject.toml` + `poetry.lock`
- **Install step**: replace the `uv sync ...` steps with `poetry install` (typically with a virtualenv configured inside the image).
- **Copy step**: copy the created virtualenv (or site-packages) into the runtime stage, same as the reference Dockerfile copies `/app/.venv`.

### pip-tools pattern

- **Inputs**: `pyproject.toml` (or `requirements.in`) + compiled `requirements.txt` (or `requirements.lock`)
- **Install step**: use `pip install --require-hashes -r requirements.txt` (if you generate hashes) to keep installs deterministic.
- **Copy step**: copy the resulting venv into runtime, same multi-stage idea.

If you keep the design (lock-driven install, cached downloads, copy minimal runtime artifacts, run as non-root), you can swap dependency managers without changing the overall Docker approach.

---

## Compose for local dev

An example compose file is provided at `pages/docs/docker/docker-compose.yaml.example` as a baseline starting point for running the container locally.

An optional hardened override is provided at `pages/docs/docker/docker-compose.hardened.yaml.example`. It is designed to layer on top of the baseline file so you can switch between a low-friction dev posture and a production-like posture with a single additional `-f` flag.

This section focuses on how to read and adapt the compose file. For the setup and run steps, follow the earlier **Quickstart: Local dev with compose** section.

### When to use each compose field

- **`build`**: use when you want the image rebuilt from local sources as part of `up`. Avoid it when you want to pin to a prebuilt image digest.
  - **`build.args`**: use when the Dockerfile requires build-time arguments (the reference Dockerfile requires `PACKAGE_NAME` to match your top-level Python package directory; `UV_IMAGE` is optional if you want to pin the installer image).
    - Compose variable substitution for build args uses your shell environment and the project-level `.env` file. It does not use `env_file` (which only sets runtime container environment variables).
- **`env_file`**: use when you want a single place for environment-specific defaults (and to keep `docker-compose.yaml` stable across environments).
- **`environment`**: use for values you want to be explicit at the compose layer (and to override `env_file` selectively).
- **`ports`**: use when the service must be reachable from the host (local testing). Avoid publishing ports you do not need.
  - If you want a configurable port mapping, use compose variable substitution (for example `"${PORT:-8080}:${PORT:-8080}"`) and set `PORT` at runtime so the app and healthcheck agree.
- **`volumes`**: use when the container needs persistent data (named volumes) or when you want to mount local files into the container for dev (bind mounts). Prefer explicit mounts over relying on writes to the container filesystem.
- **`restart`**: controls how Docker restarts containers.
  - `restart: "no"`: use for one-shot tasks and batch jobs where failure should surface immediately (and exit codes should be visible to CI/operators).
  - `restart: on-failure[:N]`: use for transient failures where retrying is reasonable; consider adding a max retry count.
  - `restart: always`: use for long-running services that should be kept up regardless of manual stops (commonly used outside local dev).
  - `restart: unless-stopped`: use for long-running services that should restart on failure and daemon restart, but stay stopped if an operator intentionally stops them (common for local dev stacks).
- **`command`**: use when the runtime command differs between contexts (dev server vs worker vs batch).
- **`container_name`**: use only when you have a strong reason to force a fixed container name. Avoid it if you want to scale services (for example `docker compose up --scale ...`) or if you run multiple copies of the same stack (to prevent name collisions).
- **`init`, `read_only`, `cap_drop`, `security_opt`, `tmpfs`**: use when you want runtime hardening defaults (read-only filesystem, least privilege, and explicit writable paths such as `/tmp`).
  - `init: true` enables a minimal init process as PID 1 inside the container. It improves signal handling (shutdown behavior) and reaps zombie processes when your app spawns subprocesses.
  - `read_only: true` makes the container filesystem read-only. This is a practical way to detect unexpected writes early and to reduce the amount of mutable state a container can accumulate at runtime.
  - `cap_drop: [ALL]` removes Linux capabilities from the container to enforce least privilege. Add back only what you can justify for the workload.
  - `security_opt: ["no-new-privileges:true"]` prevents privilege escalation through setuid/setcap binaries, even if they exist in the image.
  - `tmpfs: [...]` creates explicit writable mounts in memory (for example `/tmp` and `/app/logs`). Combined with `read_only: true`, this makes writable paths intentional and easy to audit.
  - In practice, keep these settings in a separate override file (for example `docker-compose.hardened.yaml`) so the baseline setup stays easy to run.
  - A hardened setup can surface hidden assumptions in Python libraries that try to write to `~/.cache` or `~/.config`. If you use `read_only: true` and a non-root user, route caches/config to writable locations (commonly under `/tmp`) via environment variables such as `HOME`, `XDG_CACHE_HOME`, and `XDG_CONFIG_HOME`.
- **`healthcheck`**: use when you want an explicit readiness signal for local orchestration and troubleshooting. Prefer checks that do not require extra OS packages (the example uses Python to probe the listening port).

If you do not want to copy files, you can also adapt the compose file to reference the example Dockerfile path directly (set `dockerfile: pages/docs/docker/Dockerfile.example`).

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

These patterns show up constantly in real systems. The goal is to make them explicit, so you avoid “mystery state” and inconsistent runs.

#### Model and artifact caching

If your service downloads models or artifacts at startup, decide deliberately where they live:

- **Ephemeral cache** (safe default): write to `/tmp` and accept that containers are disposable.
- **Named volume cache** (intentional performance trade-off): mount a named volume so repeated starts are faster, then document cache invalidation.

When you enable `--read-only` / `read_only: true`, also route caches to writable paths (commonly under `/tmp`) via environment variables such as `HOME`, `XDG_CACHE_HOME`, and `XDG_CONFIG_HOME`.

#### Datasets and “inputs”

Treat input data as an external dependency:

- **Local dev**: bind-mount a read-only dataset directory (so you do not accidentally write into it).
- **CI and prod**: pull inputs from object storage or a data platform, not from inside the image.

Avoid copying datasets into images. It makes builds slow, images huge, and data lifecycle management unclear.

#### GPUs

If you need GPUs, keep the container contract simple:

- Make GPU usage **optional** (the same image should run CPU-only).
- Prefer runtime configuration (for example `--gpus all`) over building separate “gpu” images unless you truly need different system dependencies.

### Logs and observability

Containers are easiest to operate when you treat them as stateless processes and capture all logs and signals from the outside.

#### Follow logs

Use `docker logs -f` when you are running a single container and want to stream logs continuously.

```bash
docker logs -f <container-name-or-id>
```

With Compose, use `docker compose logs -f` when you want a unified stream across all services in a stack.

```bash
docker compose logs -f
```

Use `docker compose logs -f <service-name>` when you want a focused stream for one component.

```bash
docker compose logs -f <service-name>
```

#### Recommended conventions

- **Write logs to stdout/stderr**: avoid writing app logs to local files inside the container unless you have a clear persistence plan.
- **Use structured logs when possible**: JSON logs are easier to parse and ship to centralized logging systems.
- **Promote log level to config**: keep `LOG_LEVEL` (or equivalent) as an environment variable so you can increase verbosity without rebuilding.
- **Add a healthcheck for uptime signals**: a `HEALTHCHECK` makes failures visible to orchestrators and to `docker ps`.
- **Expose metrics and traces explicitly**: if you use Prometheus/OpenTelemetry, document the port and enablement env vars, then publish those ports only where needed.

---

## Daily commands

These are commonly used commands once containers are part of the workflow:

- **What is running**: `docker ps` (add `-a` to include stopped containers)
- **What images exist locally**: `docker images`
- **Stop a container**: `docker stop <container>`
- **Remove a container**: `docker rm <container>` (add `-f` to force)
- **Shell into a running container**: `docker compose exec <service-name> /bin/sh` (compose) or `docker exec -it <container> /bin/sh` (single container)
- **Resource usage**: `docker stats`
- **Compose status**: `docker compose ps`
- **Stop a compose stack**: `docker compose down` (add `-v` if you intentionally want to remove named volumes)

---

## Troubleshooting

### Lockfiles are not available in the build context

Make sure your build context includes `pyproject.toml` and the lock file referenced by the Dockerfile (for `uv`, this is `uv.lock`). Run `docker build` from the project root so those files are in the build context.

If your Dockerfile is not in the project root, keep the build context as `.` and point to the Dockerfile via `-f`, as shown in the Quickstart.

### Enable BuildKit mounts (`RUN --mount=...`)

The reference Dockerfile uses BuildKit mounts for caching and for keeping dependency layers narrowly scoped. If BuildKit is already enabled, this should work:

```bash
docker build -f pages/docs/docker/Dockerfile.example -t <image>:<tag> --build-arg PACKAGE_NAME=<package_dir> .
```

If you see an error about `RUN --mount=...` not being supported, enable BuildKit for that command by prefixing it with `DOCKER_BUILDKIT=1`:

```bash
DOCKER_BUILDKIT=1 docker build -f pages/docs/docker/Dockerfile.example -t <image>:<tag> --build-arg PACKAGE_NAME=<package_dir> .
```

### Ensure `PACKAGE_NAME` points to a real directory

Set `PACKAGE_NAME` to an existing folder in the build context:

Use a build command that includes `--build-arg PACKAGE_NAME=<package_dir>` and ensure `<package_dir>` exists in the build context.

### Port mapping does not work

- Ensure the container listens on `0.0.0.0:${PORT}`, not `127.0.0.1`.
- Confirm you published the port: `-p 8080:8080`.

### File permissions at runtime

The image runs as `appuser`. If your application needs to write files, write under directories owned by `appuser` (the reference creates `/app/logs` and sets ownership).
