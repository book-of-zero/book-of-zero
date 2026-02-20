---
layout: post
title: "Docker Compose: orchestrate local development"
nav_order: 3
---

Docker Compose gives you a single command to build, start, and wire together the services that make up your local development stack. This page covers the Compose workflow and configuration. For building images and running standalone containers, see [Docker: containerize a Python project]({{ site.baseurl }}/docs/containerization/docker/docker/).

---

## On this page

- [Key concepts](#key-concepts)
- [Quick start: local dev with compose](#quick-start-local-dev-with-compose)
  - [Setup steps](#setup-steps)
  - [Hardened posture](#hardened-posture)
  - [Day-to-day loop](#day-to-day-loop)
- [Resources](#resources)
- [Compose configuration reference](#compose-configuration-reference)
- [Logs and debugging](#logs-and-debugging)

---

## Key concepts

- **Services**: a service is a named container definition in `docker-compose.yaml`. Each service maps to one image (built or pulled) and one or more running containers.
- **Override files**: Compose merges multiple `-f` files in order. Use an override file (for example `docker-compose.hardened.yaml`) to layer settings on top of a baseline without editing it.
- **Variable substitution**: `${VAR:-default}` in a compose file resolves from your shell environment and the project-level `.env` file. This keeps the compose file stable across environments.
- **`env_file` vs `environment`**: `env_file` loads variables into the running container at runtime. `environment` sets variables directly in the compose file and can override `env_file` values. Neither is used for build-time `args` (those come from variable substitution or the shell).

---

## Quick start: local dev with compose

This page assumes Docker and Docker Compose are installed. If you are new to Docker images, start with the [Docker page]({{ site.baseurl }}/docs/containerization/docker/docker/) for image building fundamentals and the reference Dockerfile.

### Setup steps

1. Copy the examples into conventional names at the project root:
   - `_pages/docs/containerization/docker/Dockerfile.example` → `Dockerfile`
   - `_pages/docs/containerization/docker-compose/docker-compose.yaml.example` → `docker-compose.yaml`
   - `_pages/docs/containerization/docker-compose/docker-compose.hardened.yaml.example` → `docker-compose.hardened.yaml` if you want a production-like posture
2. Create the env files for compose:
   - **`.env`**: used by compose for `${...}` variable substitution and passed into the container at runtime via `env_file` (application configuration).
3. Start the service:

Keep runtime secrets out of version control. Treat `.env` as local-only inputs and share templates (for example `.env.example`) when you need a documented baseline.

```bash
docker compose up --build
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

### Hardened posture

Use this when you want to surface hidden write assumptions early while keeping the baseline setup simple.

```bash
docker compose -f docker-compose.yaml -f docker-compose.hardened.yaml up --build
```

The hardened override adds `read_only: true`, drops all Linux capabilities, prevents privilege escalation, and creates explicit writable `tmpfs` mounts for `/tmp`. This can surface hidden assumptions in Python libraries that try to write to `~/.cache` or `~/.config`. Route those caches to writable locations (commonly under `/tmp`) via environment variables such as `HOME`, `XDG_CACHE_HOME`, and `XDG_CONFIG_HOME`.

### Day-to-day loop

This is a practical loop for local development:

- **Start the stack**: follow the setup steps above.
- **Observe logs**: use the commands in [Logs and debugging](#logs-and-debugging).
- **Restart after changes** (when you are not bind-mounting code): `docker compose up --build --force-recreate`
- **Run a one-shot command in the same image** (migrations, a backfill, a smoke test):

```bash
docker compose run --rm <service-name> /app/.venv/bin/python -m <module>
```

- **Debug inside the container**: use the shell commands in [Logs and debugging](#logs-and-debugging).

If your local workflow needs live code reload, prefer making that an explicit compose variant (bind-mount code + dev server command), while keeping the default path "rebuild the image" so you do not accidentally depend on host-only state.

---

## Resources

Use these templates when you want a baseline quickly, then customize for your application.

<details class="boz-resource">
  <summary><code>docker-compose.yaml.example</code></summary>

{% highlight yaml %}
{% include_relative docker-compose.yaml.example %}
{% endhighlight %}
</details>

<details class="boz-resource">
  <summary><code>docker-compose.hardened.yaml.example</code></summary>

{% highlight yaml %}
{% include_relative docker-compose.hardened.yaml.example %}
{% endhighlight %}
</details>

---

## Compose configuration reference

An example compose file is provided at `_pages/docs/containerization/docker-compose/docker-compose.yaml.example` as a baseline starting point for running the container locally.

An optional hardened override is provided at `_pages/docs/containerization/docker-compose/docker-compose.hardened.yaml.example`. It is designed to layer on top of the baseline file so you can switch between a low-friction dev posture and a production-like posture with a single additional `-f` flag.

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
  - `tmpfs: [...]` creates explicit writable mounts in memory (for example `/tmp`). Combined with `read_only: true`, this makes writable paths intentional and easy to audit.
  - In practice, keep these settings in a separate override file (for example `docker-compose.hardened.yaml`) so the baseline setup stays easy to run.
  - A hardened setup can surface hidden assumptions in Python libraries that try to write to `~/.cache` or `~/.config`. If you use `read_only: true` and a non-root user, route caches/config to writable locations (commonly under `/tmp`) via environment variables such as `HOME`, `XDG_CACHE_HOME`, and `XDG_CONFIG_HOME`.
- **`healthcheck`**: use when you want an explicit readiness signal for local orchestration and troubleshooting. Prefer checks that do not require extra OS packages (the example uses Python to probe the listening port).

If you do not want to copy files, you can also adapt the compose file to reference the example Dockerfile path directly (set `dockerfile: _pages/docs/containerization/docker/Dockerfile.example`).

---

## Logs and debugging

With Compose, use `docker compose logs -f` when you want a unified stream across all services in a stack.

### Follow logs

Stream logs from all services:

```bash
docker compose logs -f
```

Stream logs from a single service:

```bash
docker compose logs -f <service-name>
```

For single-container log streaming, see the [Docker page: Logs and debugging]({{ site.baseurl }}/docs/containerization/docker/docker/#logs-and-debugging).

### Debug a running service

Shell into a running service to inspect the filesystem, check environment variables, or run one-off commands:

```bash
docker compose exec <service-name> /bin/sh
```

Run a one-shot command without starting the full stack:

```bash
docker compose run --rm <service-name> <command>
```

### Inspect from outside

- **Compose status**: `docker compose ps`
- **Stop a compose stack**: `docker compose down` (add `-v` if you intentionally want to remove named volumes)
- **Rebuild and restart**: `docker compose up --build --force-recreate`

For standalone Docker commands (`docker ps`, `docker images`, `docker stats`, etc.), see the [Docker page: Logs and debugging]({{ site.baseurl }}/docs/containerization/docker/docker/#logs-and-debugging).
