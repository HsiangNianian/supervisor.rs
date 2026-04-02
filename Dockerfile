FROM python:3.11-slim AS builder

# Install Rust toolchain
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl build-essential && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
    rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app

# Copy source
COPY Cargo.toml pyproject.toml setup.py ./
COPY src/ src/

# Install maturin and build
RUN pip install --no-cache-dir maturin && \
    maturin build --release --out dist

# --- Runtime stage ---
FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /app/dist/*.whl /tmp/

# Install the wheel and optional server dependencies
RUN pip install --no-cache-dir /tmp/*.whl && \
    pip install --no-cache-dir fastapi uvicorn pyyaml typer && \
    rm -rf /tmp/*.whl

# Copy config template
COPY docker-compose.yaml /app/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:8000/health'); r.raise_for_status()" || exit 1

ENTRYPOINT ["python", "-m", "supervisor.cli"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
