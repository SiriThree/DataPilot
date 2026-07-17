FROM python:3.12-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv via official installer (platform-independent, avoids cross-arch COPY issue)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Copy project files
COPY pyproject.toml uv.lock ./
COPY README.md README.zh.md ./
COPY src/ src/
COPY main.py ./

# Install dependencies (non-editable, just the venv)
RUN uv sync --frozen --no-dev --no-editable

# Create required directories
RUN mkdir -p /input /output /logs

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

ENTRYPOINT ["uv", "run", "--no-sync", "python", "main.py"]
