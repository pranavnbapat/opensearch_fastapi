# models/embedding.py

import numpy as np

from functools import lru_cache
from sentence_transformers import SentenceTransformer
import os


# Cached Model Loader
@lru_cache(maxsize=5)  # Cache up to 5 models
def get_model(name: str):
    return SentenceTransformer(name)

def select_model(selected_model: str):
    model_mapping = {
        "minilm": ("MODEL_MINILM", "INDEX_NAME_MINILM"),
        "mpnet": ("MODEL_MPNET", "INDEX_NAME_MPNET"),
        "mxbai": ("MODEL_MXBAI", "INDEX_NAME_MXBAI"),
        "sentence_t5": ("MODEL_SENTENCE_T5", "INDEX_NAME_SENTENCE_T5"),
        "multilingual_e5": ("MODEL_MULTILINGUAL_E5", "INDEX_NAME_MULTILINGUAL_E5"),

        "msmarco": ("MODEL_MSMARCO", "INDEX_NAME_MSMARCO"),
    }

    if selected_model in model_mapping:
        model_env, index_env = model_mapping[selected_model]
        model = get_model(os.getenv(model_env, "default_model"))
        index_name = os.getenv(index_env, "default_index")

        if index_name == "default_index":
            raise ValueError(f"Missing index name for model '{selected_model}'. Check your environment variables.")

        return model, index_name

    raise ValueError("Invalid model selected")


def generate_vector(model, query: str):
    """Generate vector embeddings for the given query."""
    return model.encode(query).tolist()


def generate_vector_neural_search(model, query: str):
    """Generate vector embeddings for the given query."""
    vector = model.encode(query)

    if isinstance(vector, np.ndarray):
        vector = vector.tolist()

    if not all(isinstance(v, (float, int)) for v in vector):
        raise ValueError(f"Invalid query vector format! Expected list of floats, got: {vector}")

    return vector

    # if hasattr(vector, "tolist"):  # ✅ Convert NumPy array or Tensor to list
    #     vector = vector.tolist()
    #
    # # ✅ Final Check
    # if not isinstance(vector, list) or not all(isinstance(v, (float, int)) for v in vector):
    #     raise ValueError(f"Invalid query vector format: {vector}")
    #
    # return vector  # Ensure it's a list of floats

