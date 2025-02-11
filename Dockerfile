# Use official Python image as base
FROM python:3.11-slim

# Set environment variables for models
ENV MODEL_MPNET=sentence-transformers/all-mpnet-base-v2
ENV MODEL_MINILM=sentence-transformers/all-MiniLM-L6-v2
ENV MODEL_MXBAI=mixedbread-ai/mxbai-embed-large-v1
ENV MODEL_SENTENCE_T5=sentence-transformers/sentence-t5-base
ENV MODEL_MULTILINGUAL_E5=intfloat/multilingual-e5-large

# Set working directory
WORKDIR /app

# Copy dependency list first (improves caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Preload all models separately to cache them effectively
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('${MODEL_MPNET}'); \
    SentenceTransformer('${MODEL_MINILM}'); \
    SentenceTransformer('${MODEL_MXBAI}'); \
    SentenceTransformer('${MODEL_SENTENCE_T5}'); \
    SentenceTransformer('${MODEL_MULTILINGUAL_E5}')"

# Copy the rest of the project files
COPY . .

# Preload models during build (optional, if using a separate script)
# RUN python preload_models.py

# Expose FastAPI port
EXPOSE 10000

# Run the FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]
