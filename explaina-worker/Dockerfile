FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg fonts-dejavu curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY worker_requirements.txt /app/worker_requirements.txt
RUN pip install --no-cache-dir -r /app/worker_requirements.txt
COPY worker_main.py /app/worker_main.py
COPY .build_timestamp /app/.build_timestamp
RUN echo "Built on: $(cat /app/.build_timestamp || echo 'unknown')" && python -m py_compile /app/worker_main.py
ENV PORT=8080 PYTHONUNBUFFERED=1 MAX_WORKERS=2 MEDIA_DIR=media CLEAN_MEDIA_FILES=false
HEALTHCHECK --interval=60s --timeout=5s --start-period=20s --retries=3 CMD curl -fs http://localhost:${PORT}/health || exit 1
EXPOSE 8080
CMD ["uvicorn", "worker_main:app", "--host", "0.0.0.0", "--port", "8080", "--log-level", "debug"]