# Use a slim Python base image
FROM python:3.11-slim

# Keep Python from writing .pyc files and make logs unbuffered
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8011 \
    HOST=0.0.0.0 \
    CLOUDRUN_HOST=ecom_white.taosun.net

WORKDIR /app

# Install build tools and headers needed for compiled deps (lxml, cryptography, uvloop, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    libffi-dev \
    libssl-dev \
    libxml2-dev \
    libxslt1-dev \
  && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first for better Docker cache usage
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose for local runs; Cloud Run will honor $PORT
EXPOSE 8011

# Default start command via Agentbeats Controller
CMD ["sh", "-c", "PORT=${PORT} HOST=${HOST} CLOUDRUN_HOST=${CLOUDRUN_HOST} agentbeats run_ctrl"]
