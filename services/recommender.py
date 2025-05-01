# services/recommender.py

from pydantic import BaseModel, Field
from services.utils import client, recomm_model

class RecommenderRequest(BaseModel):
    text: str = Field(default="Education and training of the farmers")
    top_k: int = 3

def recommend_similar(text: str, top_k: int = 5, index_name: str = "test_recomm"):
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