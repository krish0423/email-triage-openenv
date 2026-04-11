FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd -m -u 1000 appuser \
    && mkdir -p /app/data \
    && chown -R appuser:appuser /app

USER appuser

ENV LOG_PATH=/app/data/env_step_logs.jsonl

EXPOSE 7860
ENV PORT=7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]