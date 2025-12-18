# Use a slim Python base image
FROM python:3.13-slim

# Keep Python from writing .pyc files and make logs unbuffered
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    AGENT_PORT=9002 \
    ROLE=white

WORKDIR /app

# Install build tools and headers needed for compiled deps (lxml, cryptography, uvloop, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    libffi-dev \
    libssl-dev \
  && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first for better Docker cache usage
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose the default port
EXPOSE 9002

# Run the white agent directly
CMD ["python", "my_white_agent.py"]
