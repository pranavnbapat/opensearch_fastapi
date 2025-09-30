# syntax=docker/dockerfile:1.7

# Use official Python image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy dependency list first (improves caching)
COPY requirements.txt .

# Install system dependencies, upgrade pip, install Python packages, download NLTK stopwords
#RUN apt-get update && apt-get install -y \
#    g++ cmake libffi-dev libssl-dev wget nano \
#    && rm -rf /var/lib/apt/lists/* \
#    && pip install --upgrade pip \
#    && pip install --no-cache-dir -r requirements.txt \
#    && pip install --no-cache-dir 'huggingface_hub[hf_xet]' \
#    && python -m nltk.downloader stopwords

# Install OS deps, then Python deps with a persistent pip cache
RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ cmake libffi-dev libssl-dev wget nano \
  && rm -rf /var/lib/apt/lists/* \
  && python -m pip install --upgrade pip

# Python deps with pip cache (BuildKit)  ← the --mount must be here
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt 'huggingface_hub[hf_xet]'

# NLTK data (network fetch at build time)
RUN python -m nltk.downloader stopwords


# Copy the rest of the project files
COPY . .

# Expose FastAPI port
EXPOSE 10000

# Run the FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]
