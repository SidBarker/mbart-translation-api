# Use NVIDIA CUDA base image
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_PATH=/runpod-volume/models \
    DEVICE=cuda \
    DEBIAN_FRONTEND=noninteractive \
    RUNPOD_DEBUG_LEVEL=info \
    TRANSFORMERS_CACHE=/runpod-volume/cache \
    HF_HOME=/runpod-volume/huggingface \
    TORCH_HOME=/runpod-volume/torch

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

# Create directories for models and cache
RUN mkdir -p /runpod-volume/models /runpod-volume/cache /runpod-volume/huggingface /runpod-volume/torch

# Copy application code
COPY . .

# Pre-download the language detection model
RUN python -c "from transformers import pipeline; pipeline('text-classification', model='papluca/xlm-roberta-base-language-detection')"

# Create a script to download the model if not found locally
RUN echo '#!/usr/bin/env python3\n\
import os\n\
import sys\n\
import logging\n\
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast\n\
\n\
logging.basicConfig(level=logging.INFO)\n\
logger = logging.getLogger("model_download")\n\
\n\
MODEL_ID = "facebook/mbart-large-50-many-to-many-mmt"\n\
MODEL_PATH = os.environ.get("MODEL_PATH", "/runpod-volume/models")\n\
\n\
def download_model():\n\
    logger.info(f"Pre-downloading mBART model to {MODEL_PATH}")\n\
    try:\n\
        # Make sure directories exist\n\
        os.makedirs(MODEL_PATH, exist_ok=True)\n\
        \n\
        # Download model and tokenizer\n\
        tokenizer = MBart50TokenizerFast.from_pretrained(MODEL_ID)\n\
        model = MBartForConditionalGeneration.from_pretrained(MODEL_ID)\n\
        \n\
        # Save to disk\n\
        tokenizer.save_pretrained(MODEL_PATH)\n\
        model.save_pretrained(MODEL_PATH)\n\
        \n\
        logger.info(f"Successfully downloaded model files to {MODEL_PATH}")\n\
        return True\n\
    except Exception as e:\n\
        logger.error(f"Error downloading model: {e}")\n\
        return False\n\
\n\
if __name__ == "__main__":\n\
    download_model()\n\
' > /app/download_model.py && chmod +x /app/download_model.py

# Create a runpod_start.sh script to properly handle both API and serverless modes
RUN echo '#!/bin/bash\n\
\n\
# Function to check if model exists locally\n\
check_model() {\n\
  if [[ ! -f "${MODEL_PATH}/config.json" ]]; then\n\
    echo "Model files not found in ${MODEL_PATH}. Attempting to download..."\n\
    python /app/download_model.py\n\
  else\n\
    echo "Model files found in ${MODEL_PATH}"\n\
  fi\n\
}\n\
\n\
# Ensure the volume directories exist\n\
mkdir -p /runpod-volume/models /runpod-volume/cache /runpod-volume/huggingface /runpod-volume/torch\n\
\n\
# Check for model files\n\
check_model\n\
\n\
# Start the appropriate service\n\
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
VOLUME ["/runpod-volume/models", "/runpod-volume/cache", "/runpod-volume/huggingface", "/runpod-volume/torch"]

# Set entrypoint
ENTRYPOINT ["/app/runpod_start.sh"]
