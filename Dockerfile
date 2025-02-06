# Use official Python image as base
FROM python:3.11-slim

# Set environment variables for models (optional)
ENV MODEL_MINILM=sentence-transformers/all-MiniLM-L6-v2
ENV MODEL_MPNET=sentence-transformers/all-mpnet-base-v2

# Set working directory
WORKDIR /app

# Copy dependency list first (improves caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# ✅ Preload models separately to cache them effectively
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('${MODEL_MINILM}'); \
    SentenceTransformer('${MODEL_MPNET}')"

# Copy the rest of the project files
COPY . .

# Preload models during build
# RUN python preload_models.py

# Expose FastAPI port
EXPOSE 10000

# Run the FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]
