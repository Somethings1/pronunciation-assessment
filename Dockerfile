# Production Dockerfile for Pronunciation Assessment API
FROM python:3.11-slim as backend

# Install system audio dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsndfile1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY server/requirements.txt /app/server/requirements.txt
RUN pip install --no-cache-dir -r /app/server/requirements.txt

# Copy server code and models
COPY server/ /app/server/
COPY test_samples/ /app/test_samples/

# Expose FastAPI port
EXPOSE 8000

ENV PYTHONUNBUFFERED=1
ENV PORT=8000

CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]
