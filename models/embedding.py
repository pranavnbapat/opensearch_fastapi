# models/embedding.py
from functools import lru_cache
from sentence_transformers import SentenceTransformer
import os


# Cached Model Loader
@lru_cache(maxsize=5)  # Cache up to 5 models
def get_model(name: str):
    return SentenceTransformer(name)


# Define available models & corresponding index names
MODEL_CONFIG = {
    "mpnet": {"name": os.getenv("MODEL_MPNET", "sentence-transformers/all-mpnet-base-v2"),
              "index": os.getenv("INDEX_NAME_MPNET", "mpnet_index")},
    "minilm": {"name": os.getenv("MODEL_MINILM", "sentence-transformers/all-MiniLM-L6-v2"),
               "index": os.getenv("INDEX_NAME_MINILM", "minilm_index")},
    "mxbai": {"name": os.getenv("MODEL_MXBAI", "mixedbread-ai/mxbai-embed-large-v1"),
              "index": os.getenv("INDEX_NAME_MXBAI", "mxbai_index")},
    "sentence_t5": {"name": os.getenv("MODEL_T5", "sentence-transformers/sentence-t5-base"),
                    "index": os.getenv("INDEX_NAME_T5", "sentence_t5_index")},
    "multilingual_e5": {"name": os.getenv("MODEL_E5", "intfloat/multilingual-e5-large"),
                        "index": os.getenv("INDEX_NAME_E5", "multilingual_e5_index")},
}


def select_model(model_name: str):
    if model_name not in MODEL_CONFIG:
        raise ValueError(f"Invalid model: {model_name}. Choose from {list(MODEL_CONFIG.keys())}")

    model = get_model(MODEL_CONFIG[model_name]["name"])
    index_name = MODEL_CONFIG[model_name]["index"]

    return model, index_name


def generate_vector(model, query: str):
    """Generate vector embeddings for the given query."""
    return model.encode(query).tolist()
