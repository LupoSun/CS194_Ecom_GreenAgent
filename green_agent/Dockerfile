# Dockerfile for E-commerce Green Agent (AgentBeats-compatible)
FROM python:3.13-slim

# Prevent Python from writing .pyc files and make logs unbuffered
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=9009 \
    HOST=0.0.0.0 \
    ROLE=green

WORKDIR /app

# Install build tools needed for compiled dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    libffi-dev \
    libssl-dev \
  && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and data
COPY . .

# Expose the default port
EXPOSE 9009

# Start the green agent
# Note: the server reads AGENT_PORT, while Cloud Run sets PORT.
CMD ["sh", "-c", "ROLE=${ROLE} HOST=${HOST} AGENT_PORT=${PORT} python3 green_main_A2A.py"]
