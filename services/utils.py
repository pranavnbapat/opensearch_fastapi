# services/utils.py

import base64
import nltk
import numpy as np
import os

from datetime import datetime
from dotenv import load_dotenv
from functools import lru_cache
from nltk.corpus import stopwords
from opensearchpy import OpenSearch, RequestsHttpConnection
from sentence_transformers import SentenceTransformer
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from starlette.requests import Request
from starlette.status import HTTP_401_UNAUTHORIZED
from typing import Dict, Any

load_dotenv()

def get_stopwords(lang="english"):
    try:
        return set(stopwords.words(lang))
    except LookupError:
        nltk.download("stopwords")
        return set(stopwords.words(lang))

STOPWORDS = get_stopwords()

K_VALUE = 10

PAGE_SIZE = 10

# recomm_model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
# recomm_sys_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

RECOMM_SYS_SUPPORTED_MODELS = {
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "e5": "intfloat/e5-base",
    "bge": "BAAI/bge-base-en-v1.5",
    "distilbert": "distilbert-base-nli-stsb-mean-tokens",
}

MODEL_CONFIG = {
    "minilml12v2": {
        "index": "neural_search_index_minilml12v2",
        "model_id": "nabLu5YBflT08yLW0hsI"
    },
    "mpnetv2": {
        "index": "neural_search_index_mpnetv2",
        "model_id": "VfYasJYBO4HQYgJp9pB7"
    },
    "msmarco": {
        "index": "neural_search_index_msmarco_distilbert",
        "model_id": "Vvbwu5YBO4HQYgJpFZCZ"
    }
}

model_id = os.getenv("OPENSEARCH_MSMARCO_MODEL_ID", "Vvbwu5YBO4HQYgJpFZCZ")  # Fallback to default if not set

# Fetch OpenSearch credentials
OPENSEARCH_API = os.getenv("OPENSEARCH_API")
OPENSEARCH_USR = os.getenv("OPENSEARCH_USR")
OPENSEARCH_PWD = os.getenv("OPENSEARCH_PWD")

BASIC_AUTH_USER = os.getenv("BASIC_AUTH_USER")
BASIC_AUTH_PASS = os.getenv("BASIC_AUTH_PASS")

if not all([OPENSEARCH_API, OPENSEARCH_USR, OPENSEARCH_PWD]):
    raise EnvironmentError("Missing OpenSearch environment variables!")

# OpenSearch Client
client = OpenSearch(
    hosts=[{"host": OPENSEARCH_API, "port": 443}],
    http_auth=(OPENSEARCH_USR, OPENSEARCH_PWD),
    use_ssl=True,
    verify_certs=True,
    http_compress=True,
    connection_class=RequestsHttpConnection,
    timeout=60,
    max_retries=3,
    retry_on_timeout=True
)


class MultiUserTimedAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, users: dict):
        super().__init__(app)
        self.users = users

    async def dispatch(self, request: Request, call_next):
        auth = request.headers.get("Authorization")
        if not auth or not auth.startswith("Basic "):
            return Response(status_code=HTTP_401_UNAUTHORIZED, headers={"WWW-Authenticate": "Basic"},
                            content="Auth required")

        try:
            encoded = auth.split(" ")[1]
            decoded = base64.b64decode(encoded).decode("utf-8")
            username, password = decoded.split(":", 1)
        except Exception:
            return Response(status_code=HTTP_401_UNAUTHORIZED, headers={"WWW-Authenticate": "Basic"},
                            content="Invalid auth")

        user = self.users.get(username)
        if not user or user["password"] != password:
            return Response(status_code=HTTP_401_UNAUTHORIZED, headers={"WWW-Authenticate": "Basic"},
                            content="Invalid credentials")

        # Time-based access restriction
        expires = user.get("expires")
        if expires and datetime.now() > expires:
            return Response(status_code=HTTP_401_UNAUTHORIZED, headers={"WWW-Authenticate": "Basic"},
                            content="Access expired")

        return await call_next(request)


# Convert all filters to lowercase to match OpenSearch indexing
def lowercase_list(values):
    return [v.lower() for v in values] if values else []


def remove_stopwords_from_query(query: str) -> str:
    tokens = query.lower().split()
    return ' '.join([word for word in tokens if word not in STOPWORDS])


def normalise_scores(hits):
    valid_scores = [hit["_score"] for hit in hits if isinstance(hit["_score"], (int, float)) and hit["_score"] < 0]

    print(f"Raw negative scores: {valid_scores}")  # Debug: see what you're working with

    if not valid_scores:
        print("No negative scores found. Skipping normalisation.")
        return hits

    max_neg = max(valid_scores)  # closest to zero, e.g. -1
    min_neg = min(valid_scores)  # most negative, e.g. -9549512000
    score_range = max_neg - min_neg or 1

    print(f"Normalising scores from min: {min_neg}, max: {max_neg}, range: {score_range}")

    for hit in hits:
        score = hit["_score"]
        if isinstance(score, (int, float)) and score < 0:
            hit["_score_normalised"] = (score - min_neg) / score_range
        else:
            hit["_score_normalised"] = 0.0  # this can be adjusted if needed

    return hits


def knn_search_on_field(field: str, query_vector: list, index_name: str, filters: list, k: int = 10):
    return client.search(index=index_name, body={
        "_source": {
            "excludes": [
                "title_embedding", "summary_embedding", "keywords_embedding", "topics_embedding", "content_embedding",
                "project_embedding", "project_acronym_embedding", "content_embedding_input", "topics_embedding_input",
                "keywords_embedding_input", "content_pages", "_orig_id"
            ]
        },
        "track_total_hits": False,
        "size": k,
        "query": {
            "bool": {
                "filter": filters
            }
        },
        "knn": {
            field: {
                "vector": query_vector,
                "k": k
            }
        }
    })["hits"]["hits"]


@lru_cache(maxsize=5)  # Cache up to 5 models
def get_model(name: str):
    return SentenceTransformer(name)


def generate_vector_neural_search(model, query: str):
    """Generate vector embeddings for the given query."""
    vector = model.encode(query)

    if isinstance(vector, np.ndarray):
        vector = vector.tolist()

    if not all(isinstance(v, (float, int)) for v in vector):
        raise ValueError(f"Invalid query vector format! Expected list of floats, got: {vector}")

    return vector



_model_cache = {}

def get_recomm_sys_model(model_key: str) -> SentenceTransformer:
    if model_key not in RECOMM_SYS_SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model: {model_key}. Choose from: {list(RECOMM_SYS_SUPPORTED_MODELS)}")
    if model_key not in _model_cache:
        _model_cache[model_key] = SentenceTransformer(RECOMM_SYS_SUPPORTED_MODELS[model_key])
    return _model_cache[model_key]


def format_results_neural_search(hit: Dict[str, Any]) -> Dict[str, Any]:
    source = hit["_source"]

    # Convert dateCreated to DD-MM-YYYY format
    date_created = source.get("dateCreated", "N/A")
    try:
        formatted_date = datetime.strptime(date_created, "%Y-%m-%d").strftime("%d-%m-%Y")
    except ValueError:
        formatted_date = date_created

    source["dateCreated"] = formatted_date
    source["_tags"] = source.get("keywords", [])

    return {
        "_id": hit["_id"],
        "_score": hit["_score"],
        "title": source.get("title"),
        "summary": source.get("summary"),
        "projectAcronym": source.get("projectAcronym"),
        "projectName": source.get("projectName"),
        "project_type": source.get("project_type"),
        "project_id": source.get("project_id"),
        "topics": source.get("topics"),
        "subtopics": source.get("subtopics"),
        "keywords": source.get("keywords"),
        "languages": source.get("languages"),
        "locations": source.get("locations"),
        "fileType": source.get("fileType"),
        "creators": source.get("creators"),
        "dateCreated": source.get("dateCreated"),
        "@id": source.get("@id"),
        "_tags": source.get("_tags"),
    }
