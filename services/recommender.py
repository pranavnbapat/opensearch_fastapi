# services/recommender.py

from fastapi import HTTPException
from pydantic import BaseModel, Field
from services.utils import client, get_recomm_sys_model
from sklearn.metrics.pairwise import cosine_similarity
from typing import Literal


class RecommenderRequest(BaseModel):
    text: str = Field(default="Education and training of the farmers")
    top_k: int = 3
    model_name: Literal["mpnet", "minilm", "e5", "bge", "distilbert"] = Field(
        default="distilbert",
        description="Choose from: mpnet, minilm, e5, bge, distilbert"
    )

def recommend_similar(text: str, top_k: int, model_name: str):
    model = get_recomm_sys_model(model_name)
    vector = model.encode(text).tolist()
    index_name = f"recomm_sys_{model_name}"

    # Check if index exists
    if not client.indices.exists(index=index_name):
        raise HTTPException(
            status_code=404,
            detail=f"Index '{index_name}' does not exist. Please ensure documents are indexed for model '{model_name}'.")

    res = client.search(index=index_name, body={
        "size": top_k,
        "query": {
            "knn": {
                "embedding": {
                    "vector": vector,
                    "k": top_k
                }
            }
        }
    })

    return [
        {
            "title": hit["_source"].get("title", ""),
            "summary": hit["_source"].get("summary", ""),
            "keywords": hit["_source"].get("keywords", []),
            "file_type": hit["_source"].get("file_type", ""),
            "project_name": hit["_source"].get("project_name", ""),
            "project_acronym": hit["_source"].get("project_acronym", "")
        }
        for hit in res["hits"]["hits"]
    ]


# Local Reranking Using Cosine Similarity
def recommend_similar_cos(text: str, top_k: int, model_name: str):
    model = get_recomm_sys_model(model_name)
    vector = model.encode(text).tolist()
    index_name = f"recomm_sys_{model_name}"

    # Check if index exists
    if not client.indices.exists(index=index_name):
        raise HTTPException(
            status_code=404,
            detail=f"Index '{index_name}' does not exist. Please index documents using model '{model_name}' first."
        )

    search_size = top_k * 10

    res = client.search(index=index_name, body={
        "size": search_size,
        "_source": ["title", "summary", "keywords", "file_type", "project_name", "project_acronym", "embedding"],
        "query": {
            "knn": {
                "embedding": {
                    "vector": vector,
                    "k": search_size
                }
            }
        }
    })

    hits = res["hits"]["hits"]
    if not hits:
        return []

    embeddings = [hit["_source"]["embedding"] for hit in hits]
    scores = cosine_similarity([vector], embeddings)[0]

    scored_hits = sorted(zip(hits, scores), key=lambda x: -x[1])

    return [
        {
            "title": hit["_source"].get("title", ""),
            "summary": hit["_source"].get("summary", ""),
            "keywords": hit["_source"].get("keywords", []),
            "file_type": hit["_source"].get("file_type", ""),
            "project_name": hit["_source"].get("project_name", ""),
            "project_acronym": hit["_source"].get("project_acronym", ""),
            "score": float(score)
        }
        for hit, score in scored_hits[:top_k]
    ]
