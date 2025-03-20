# Use NVIDIA CUDA base image
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_PATH=/runpod-volume/models \
    DEVICE=cuda \
    DEBIAN_FRONTEND=noninteractive \
    RUNPOD_DEBUG_LEVEL=info \
    TRANSFORMERS_CACHE=/runpod-volume/cache

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-setuptools \
    python3-venv \
    python-is-python3 \
    git \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy requirements first for better cache utilization
COPY requirements.txt .

# Install dependencies
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt && \
    pip3 install --no-cache-dir runpod torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118

# Copy application code
COPY . .

# Create directories for models and cache
RUN mkdir -p /runpod-volume/models /runpod-volume/cache

# Pre-download the language detection model
RUN python -c "from transformers import pipeline; pipeline('text-classification', model='papluca/xlm-roberta-base-language-detection')"

# Create a runpod_start.sh script to properly handle both API and serverless modes
RUN echo '#!/bin/bash\n\
if [[ -n "$RUNPOD_SERVERLESS" && "$RUNPOD_SERVERLESS" == "1" ]]; then\n\
  echo "Starting in RunPod serverless mode..."\n\
  python handler.py\n\
else\n\
  echo "Starting in API mode..."\n\
  python -m uvicorn main:app --host 0.0.0.0 --port 3000\n\
fi\n\
' > /app/runpod_start.sh && chmod +x /app/runpod_start.sh

# Default port for API mode (RunPod serverless uses port 8000 automatically)
EXPOSE 3000

# Create volume mounts for persistent storage
VOLUME ["/runpod-volume/models", "/runpod-volume/cache"]

# Set entrypoint
ENTRYPOINT ["/app/runpod_start.sh"]
