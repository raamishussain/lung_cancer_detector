FROM python:3.12-slim

WORKDIR /app

# Install system dependencies if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv

# create virtual env
RUN python -m venv /app/.venv
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"

COPY pyproject.toml uv.lock ./

# install dependencies with uv
RUN uv sync --python=/app/.venv/bin/python --frozen --no-dev --no-install-project

COPY . .

CMD ["python", "-m", "app.app"]