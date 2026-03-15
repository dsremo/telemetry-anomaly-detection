FROM python:3.11-slim AS base

# Security: non-root user
RUN groupadd -r sentinel && useradd -r -g sentinel sentinel

WORKDIR /app

# Install torch CPU-only first to avoid 3GB NVIDIA CUDA packages (CPU server)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install deps first (layer caching)
COPY pyproject.toml .
RUN pip install --no-cache-dir . 2>/dev/null || pip install --no-cache-dir -e .

# Copy source
COPY src/ src/
COPY configs/ configs/
COPY dashboard/ dashboard/
COPY landing/ landing/

# Install package
RUN pip install --no-cache-dir -e .

# Security: switch to non-root
USER sentinel

EXPOSE 8400

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import httpx; r=httpx.get('http://localhost:8400/api/v1/health',timeout=3); exit(0 if r.status_code==200 else 1)" || exit 1

CMD ["uvicorn", "dsremo.api.app:create_app", "--host", "0.0.0.0", "--port", "8400", "--factory"]
