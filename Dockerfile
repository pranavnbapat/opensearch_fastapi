# Use official Python image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy dependency list first (improves caching)
COPY requirements.txt .

# Install system dependencies, upgrade pip, install Python packages, download NLTK stopwords
RUN apt-get update && apt-get install -y \
    g++ cmake libffi-dev libssl-dev wget nano \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir 'huggingface_hub[hf_xet]' \
    && python -m nltk.downloader stopwords

# Copy the rest of the project files
COPY . .

# Expose FastAPI port
EXPOSE 10000

# Run the FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]
