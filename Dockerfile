# syntax=docker/dockerfile:1

FROM python:3.12-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install dependencies first (cached layer)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Copy source and install project
COPY src/ src/
RUN uv sync --frozen --no-dev

# Runtime
FROM python:3.12-slim

LABEL org.opencontainers.image.title="Mnemon" \
      org.opencontainers.image.description="Persistent memory layer for agentic workflows — Python API and HTTP MCP server. No API keys, no cloud dependencies, no LLM calls at runtime." \
      org.opencontainers.image.url="https://github.com/wwff-tech/mnemon" \
      org.opencontainers.image.source="https://github.com/wwff-tech/mnemon" \
      org.opencontainers.image.vendor="wwff-tech" \
      org.opencontainers.image.licenses="MIT"

COPY --from=base /app /app
WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

VOLUME /mnemon
ENV MNEMON_BASE_DIR=/mnemon

# Write container config with 0.0.0.0 bind so port mapping works
RUN mkdir -p /mnemon && \
    echo '{"bind_host": "0.0.0.0", "bind_port": 7474}' > /mnemon/config.json

EXPOSE 7474

CMD ["python", "-m", "mnemon.server"]
