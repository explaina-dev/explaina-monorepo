# --- base ---
FROM python:3.11-slim

# Build arg to bust cache
ARG CACHE_BUST=unknown
RUN echo "Cache bust: $CACHE_BUST"

WORKDIR /app

# Install Python dependencies
COPY worker_requirements.txt /app/worker_requirements.txt
RUN pip install --no-cache-dir -r /app/worker_requirements.txt

# Install ffmpeg and fonts
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    ca-certificates \
    fonts-dejavu-core \
    fonts-noto-core \
    && rm -rf /var/lib/apt/lists/*

# Create static files: silent mp3 and dummy watermark
RUN mkdir -p /app/static && \
    ffmpeg -f lavfi -i anullsrc=r=44100:cl=mono -t 2 /app/static/sample.mp3 && \
    ffmpeg -f lavfi -i color=c=white:s=100x100:d=0.1 -frames:v 1 /app/static/watermark.png

# Copy timestamp first to bust any cached source uploads
COPY .build_timestamp /app/.build_timestamp

# Copy application files
COPY worker_main.py /app/worker_main.py

# Verify the file was copied correctly
RUN echo "Build timestamp: $(cat /app/.build_timestamp)" && \
    echo "worker_main.py size: $(wc -c < /app/worker_main.py) bytes" && \
    echo "Testing Python syntax..." && \
    python -m py_compile /app/worker_main.py && \
    echo "✓ Syntax check passed!"

ENV PORT=8080
ENV PYTHONUNBUFFERED=1
EXPOSE 8080

CMD ["sh","-c","python -u -m uvicorn worker_main:app --host 0.0.0.0 --port ${PORT:-8080} --log-level debug"]
