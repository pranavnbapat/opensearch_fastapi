# services/utils.py

import base64
import nltk
import numpy as np
import os

from collections import defaultdict
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
    date_created = source.get("date_of_completion", "N/A")
    try:
        formatted_date = datetime.strptime(date_created, "%Y-%m-%d").strftime("%d-%m-%Y")
    except ValueError:
        formatted_date = date_created

    source["date_of_completion"] = formatted_date
    source["_tags"] = source.get("keywords", [])

    return {
        "_id": hit["_id"],
        "_score": hit["_score"],
        "title": source.get("title"),
        "subtitle": source.get("subtitle"),
        "description": source.get("description"),
        "projectAcronym": source.get("projectAcronym"),
        "projectName": source.get("projectName"),
        "project_type": source.get("project_type"),
        "project_id": source.get("project_id"),
        "topics": source.get("topics"),
        "themes": source.get("themes"),
        "keywords": source.get("keywords"),
        "languages": source.get("languages"),
        "locations": source.get("locations"),
        "category": source.get("category"),
        "subcategories": source.get("subcategories"),
        "creators": source.get("creators"),
        "dateCreated": source.get("date_of_completion"),
        "ko_content_flat": source.get("ko_content_flat"),
        "@id": source.get("@id"),
        "_orig_id": source.get("_orig_id"),
        "_tags": source.get("_tags"),
    }


def group_hits_by_parent(hits, parents_size=PAGE_SIZE, top_k_snippets=3, include_snippets=False):
    grouped = {}

    for h in hits:
        src = h.get("_source", {})
        pid = src.get("parent_id") or src.get("_orig_id")
        if not pid:
            continue

        entry = grouped.setdefault(pid, {
            "parent_id": pid,
            "project_name": src.get("project_name"),
            "project_acronym": src.get("project_acronym"),
            "title": src.get("title"),
            "subtitle": src.get("subtitle"),
            "description": src.get("description"),
            "keywords": src.get("keywords"),
            "topics": src.get("topics"),
            "themes": src.get("themes"),
            "locations": src.get("locations"),
            "languages": src.get("languages"),
            "category": src.get("category"),
            "subcategories": src.get("subcategories"),
            "date_of_completion": src.get("date_of_completion"),
            "creators": src.get("creators"),
            "intended_purposes": src.get("intended_purposes"),
            "project_id": src.get("project_id"),
            "project_type": src.get("project_type"),
            "project_url": src.get("project_url"),
            "@id": src.get("@id"),
            "_orig_id": src.get("_orig_id"),
            "max_score": 0.0,
            **({"snippets": []} if include_snippets else {})
        })

        score_raw = h.get("_score")
        score = score_raw if isinstance(score_raw, (int, float)) else 0.0

        if include_snippets:
            hi = h.get("highlight", {}).get("content_chunk", [])
            snippet = hi[0] if hi else (src.get("content_chunk") or "")
            entry["snippets"].append({
                "chunk_index": src.get("chunk_index"),
                "snippet": snippet,
                "score": score,
                "_id": h.get("_id")
            })

        if score > entry["max_score"]:
            entry["max_score"] = score

    parents_ranked = sorted(grouped.values(), key=lambda x: x["max_score"], reverse=True)
    top_parents = parents_ranked[:parents_size]

    # if include_snippets:
    #     for p in top_parents:
    #         p["snippets"] = sorted(p["snippets"], key=lambda s: s["score"], reverse=True)[:top_k_snippets]

    return {"total_parents": len(parents_ranked), "parents": top_parents}

def fetch_chunks_for_parents(index_name: str, parent_ids: list[str]) -> dict[str, list[dict]]:
    """
    Return all chunks for each parent_id, ordered by chunk_index.
    Shape per chunk: {"chunk_index": int, "content": str, "_id": str}
    """
    if not parent_ids:
        return {}

    body = {
        "_source": ["parent_id", "content_chunk", "chunk_index"],
        "size": 10000,
        "query": {
            "bool": {
                "filter": [{"terms": {"parent_id": parent_ids}}]
            }
        },
        "sort": [
            {"parent_id": "asc"},
            {"chunk_index": "asc"}
        ]
    }
    resp = client.search(index=index_name, body=body)

    by_parent: dict[str, list[dict]] = defaultdict(list)
    for h in resp["hits"]["hits"]:
        src = h["_source"]
        # skip meta docs or empties
        if src.get("chunk_index", -1) < 0:
            continue
        txt = src.get("content_chunk", "")
        if not isinstance(txt, str) or not txt.strip():
            continue

        by_parent[src["parent_id"]].append({
            "chunk_index": src.get("chunk_index"),
            "content": txt,
            "_id": h.get("_id"),
        })

    return by_parent
