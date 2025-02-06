from sentence_transformers import SentenceTransformer
import os

# Load environment variables
MODEL_MINILM = os.getenv("MODEL_MINILM", "sentence-transformers/all-MiniLM-L6-v2")
MODEL_MPNET = os.getenv("MODEL_MPNET", "sentence-transformers/all-mpnet-base-v2")

# Preload models to ensure they're cached
print(f"Downloading {MODEL_MINILM}...")
SentenceTransformer(MODEL_MINILM)

print(f"Downloading {MODEL_MPNET}...")
SentenceTransformer(MODEL_MPNET)

print("✅ Models downloaded and cached successfully.")
