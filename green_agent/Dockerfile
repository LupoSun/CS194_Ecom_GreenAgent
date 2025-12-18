# Dockerfile for E-commerce Green Agent (AgentBeats-compatible)
FROM python:3.13-slim

# Prevent Python from writing .pyc files and make logs unbuffered
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=9001 \
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
EXPOSE 9001

# Start the green agent
# AgentBeats will set AGENT_URL automatically in the platform
CMD ["sh", "-c", "python green_main_A2A.py"]
