FROM python:3.13-slim

# git: only needed if you build with the `sage` extra (private git+https deps).
# curl: healthcheck. Drop both if unused.
RUN apt-get update && apt-get install -y --no-install-recommends git curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install into the image's system Python instead of a nested .venv.
ENV UV_PROJECT_ENVIRONMENT=/usr/local \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

# ds-utils-lite is an editable path dep, so the source tree must be present
# before sync — hence a single COPY rather than a deps-only cache layer.
COPY . /app

RUN --mount=type=secret,id=gh_token \
    GIT_CONFIG_COUNT=1 \
    GIT_CONFIG_KEY_0="url.https://x-access-token:$(cat /run/secrets/gh_token)@github.com/.insteadOf" \
    GIT_CONFIG_VALUE_0="https://github.com/" \
    uv sync --no-dev --extra web --extra cache --extra sage

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD curl -fsS http://127.0.0.1:8000/api/mode || exit 1

# 0.0.0.0, not run_web.py's 127.0.0.1 default — otherwise unreachable outside the container.
CMD ["uvicorn", "smart_assignment.webapp.app:app", "--host", "0.0.0.0", "--port", "8000"]
