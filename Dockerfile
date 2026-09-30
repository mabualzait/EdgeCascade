# ==============================================================================
# EdgeCascade: Production Multi-stage Container
# ==============================================================================
FROM python:3.11-slim-bookworm AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-dev.txt pyproject.toml ./
RUN pip install --no-cache-dir --user -r requirements.txt -r requirements-dev.txt

# Final runtime image
FROM python:3.11-slim-bookworm AS runtime

LABEL maintainer="EdgeCascade Community"
LABEL description="Semantic Fallback Router & Hybrid Inference Engine"

WORKDIR /app

# Install runtime utilities (curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create unprivileged user for security
RUN groupadd -g 10001 edgecascade && \
    useradd -u 10001 -g edgecascade -s /bin/bash -m edgecascade && \
    chown -R edgecascade:edgecascade /app

# Copy installed python wheels from builder
COPY --from=builder /root/.local /home/edgecascade/.local
ENV PATH="/home/edgecascade/.local/bin:${PATH}"
ENV PYTHONPATH="/app"

# Copy application code and configurations
COPY --chown=edgecascade:edgecascade edgecascade/ /app/edgecascade/
COPY --chown=edgecascade:edgecascade config/ /app/config/
COPY --chown=edgecascade:edgecascade tests/ /app/tests/
COPY --chown=edgecascade:edgecascade pyproject.toml README.md LICENSE /app/

USER edgecascade

EXPOSE 8080

HEALTHCHECK --interval=15s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

ENTRYPOINT ["python3", "-m", "edgecascade.main"]
CMD ["--config", "config/config.yaml", "--host", "0.0.0.0", "--port", "8080"]
