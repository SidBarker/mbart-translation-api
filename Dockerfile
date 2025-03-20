# Use the official Python 3.10 image as a parent image
FROM python:3.10-slim

# Set environment variables
ENV MODEL_PATH=/data/models
ENV DEVICE=cuda
ENV PYTHONUNBUFFERED=1

# Install system dependencies for GPU support
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make sure xlm-roberta model is downloaded for language detection
RUN python -c "from transformers import pipeline; pipeline('text-classification', model='papluca/xlm-roberta-base-language-detection')"

# Expose port 23129 for FastAPI
EXPOSE 23129

# Add volume mounting point for models
VOLUME ["/data/models", "/app"]

# Command to run on container start
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "23129"]
