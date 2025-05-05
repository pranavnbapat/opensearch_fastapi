# services/recommender.py

from pydantic import BaseModel, Field
from services.utils import client, recomm_model
from sklearn.metrics.pairwise import cosine_similarity


class RecommenderRequest(BaseModel):
    text: str = Field(default="Education and training of the farmers")
    top_k: int = 3

def recommend_similar(text: str, top_k: int = 3, index_name: str = "test_recomm"):
    vector = recomm_model.encode(text).tolist()

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
def recommend_similar_cos(text: str, top_k: int = 3, index_name: str = "test_recomm"):
    # 1. Encode input query text
    vector = recomm_model.encode(text).tolist()

    # 2. Query more documents than needed to allow reranking
    search_size = top_k * 5

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

    # 3. Extract document vectors and rerank using cosine similarity
    hits = res["hits"]["hits"]
    if not hits:
        return []

    embeddings = [hit["_source"]["embedding"] for hit in hits]
    scores = cosine_similarity([vector], embeddings)[0]

    # 4. Sort results by similarity score
    scored_hits = sorted(zip(hits, scores), key=lambda x: -x[1])

    # 5. Prepare final top-k results
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
