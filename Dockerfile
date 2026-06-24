# ──────────────────────────────────────────────
# Transcribe v2 — Free Transcriber Web UI
# Build:  docker build -t transcribe-v2 .
# Run:    docker run -p 7860:7860 transcribe-v2
# ──────────────────────────────────────────────

FROM python:3.11-slim

# System dependencies: ffmpeg for audio conversion, build tools for librosa/numpy
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        gcc \
        && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source files
COPY app.py .
COPY transcribe_diarize.py .

# Expose the Gradio default port
EXPOSE 7860

# Whisper model cache — mount a volume here to persist downloaded models
# e.g. docker run -v whisper-models:/root/.cache/huggingface -p 7860:7860 transcribe-v2
ENV GRADIO_SERVER_NAME=0.0.0.0

CMD ["python", "app.py"]