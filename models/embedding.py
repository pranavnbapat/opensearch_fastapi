from functools import lru_cache
from sentence_transformers import SentenceTransformer
import os


# Cached Model Loader
@lru_cache(maxsize=2)
def get_model(name: str):
    return SentenceTransformer(name)


def select_model(selected_model: str):
    if selected_model == "minilm":
        model = get_model(os.getenv("MODEL_MINILM", "sentence-transformers/all-MiniLM-L6-v2"))
        index_name = os.getenv("INDEX_NAME_MINILM", "minilm_index")
    else:
        model = get_model(os.getenv("MODEL_MPNET", "sentence-transformers/all-mpnet-base-v2"))
        index_name = os.getenv("INDEX_NAME_MPNET", "mpnet_index")

    return model, index_name


def generate_vector(model, query: str):
    return model.encode(query).tolist()
