FROM python:3.11-slim

# Prevent Python buffering
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies (safe minimal set)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# HF Spaces runs as a non-root user (uid 1000).
# Create a writable directory for runtime files (logs, q_table.json)
# and make the whole /app tree owned by that user.
RUN useradd -m -u 1000 appuser \
    && mkdir -p /app/data \
    && chown -R appuser:appuser /app

USER appuser

# Point env.py log and inference.py q_table at the writable data dir
ENV LOG_PATH=/app/data/env_step_logs.jsonl
ENV QTABLE_PATH=/app/data/q_table.json

# Expose port
EXPOSE 7860
ENV PORT=7860

# Start FastAPI server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]