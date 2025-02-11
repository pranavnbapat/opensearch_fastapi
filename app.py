# app.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware

from models.embedding import select_model, generate_vector
from services.opensearch_service import search_opensearch

app = FastAPI(title="OpenSearch API", version="1.0")

origins = [
    "http://127.0.0.1:8000",
    "https://api.opensearch.nexavion.com",
    "https://backend-admin.dev.farmbook.ugent.be",
    "https://backend-admin.prd.farmbook.ugent.be",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Request Model
class SearchRequest(BaseModel):
    search_term: str
    model: str

@app.post("/search")
async def search_endpoint(request: SearchRequest):
    query = request.search_term.strip()
    selected_model = request.model

    if not query:
        raise HTTPException(status_code=400, detail="No search term provided")

    model, index_name = select_model(selected_model)
    query_vector = generate_vector(model, query)

    # Hybrid Query (k-NN + BM25)
    search_query = {
        "query": {
            "bool": {
                "should": [
                    # Vector Search
                    {
                        "knn": {
                            "vector_embedding": {
                                "vector": query_vector,
                                "k": 10
                            }
                        }
                    },
                    # BM25 Search (Keyword Filtering)
                    {"match": {"keywords.raw": {"query": query, "boost": 5}}},
                    {"match": {"projectAcronym": {"query": query, "boost": 4}}},
                    {"match": {"locations": {"query": query, "boost": 3}}},
                    {"match": {"languages": {"query": query, "boost": 2}}},

                    {"match": {"title": {"query": query, "fuzziness": "AUTO"}}},
                    {"match": {"summary": {"query": query, "fuzziness": "AUTO"}}},
                    {"match": {"content.content_pages": {"query": query, "fuzziness": "AUTO"}}}
                ],
                "minimum_should_match": 1
            }
        }
    }

    response = search_opensearch(index_name, search_query)

    # Extract raw scores
    scores = [hit["_score"] for hit in response["hits"]["hits"]]
    min_score = min(scores) if scores else 0
    max_score = max(scores) if scores else 1  # Avoid division by zero

    results = []
    for hit in response["hits"]["hits"]:
        raw_score = hit["_score"]

        # Normalize the score
        if max_score != min_score:
            normalized_score = (raw_score - min_score) / (max_score - min_score)
        else:
            normalized_score = 1.0  # All scores are the same, assign 1.0

        results.append({
            "title": hit["_source"]["title"],
            "summary": hit["_source"].get("summary", "N/A"),
            "url": hit["_source"].get("URL", "N/A"),
            "raw_score": round(raw_score, 4),
            "normalized_score": round(normalized_score, 4)  # 🔹 Added normalized score
        })

    return {"query": query, "results": results}

