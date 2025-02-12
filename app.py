# app.py

import json

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
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
    topics: Optional[List[str]] = None
    subtopics: Optional[List[str]] = None
    languages: Optional[List[str]] = None
    fileType: Optional[str] = None
    projectAcronym: Optional[str] = None
    locations: Optional[List[str]] = None

@app.post("/search")
async def search_endpoint(request: SearchRequest):
    query = request.search_term.strip()
    selected_model = request.model

    if not query:
        raise HTTPException(status_code=400, detail="No search term provided")

    model, index_name = select_model(selected_model)

    # print(f"\n🔍 Selected Index for Model {selected_model}: {index_name}")
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
        },
        "highlight": {
            "pre_tags": ["<mark>"],
            "post_tags": ["</mark>"],
            "fields": {
                "title": {},
                "summary": {},
                "content.content_pages": {}
            }
        },
        "size": 10,  # Increase size to fetch everything
        "sort": [{"_score": "desc"}]
    }

    # Print the OpenSearch query (for debugging)
    # print("\n--- OpenSearch Query Sent ---")
    # print(json.dumps(search_query, indent=4))
    # print(f"Searching in Index: {index_name}")
    # print(f"🔍 Searching in index: {index_name}")
    response = search_opensearch(index_name, search_query)

    all_results = response["hits"]["hits"]

    # Extract raw scores
    scores = [hit["_score"] for hit in all_results]
    min_score = min(scores) if scores else 0
    max_score = max(scores) if scores else 1  # Avoid division by zero

    results = []
    for hit in all_results:
        raw_score = hit["_score"]

        normalised_score = (raw_score - min_score) / (max_score - min_score) if max_score != min_score else 1.0

        # Extract highlighted text if available
        highlighted_title = hit.get("highlight", {}).get("title", [hit["_source"]["title"]])[0]
        highlighted_summary = hit.get("highlight", {}).get("summary", [hit["_source"].get("summary", "N/A")])[0]
        highlighted_content = hit.get("highlight", {}).get("content.content_pages",
                                                           [hit["_source"].get("content.content_pages", "N/A")])[0]

        results.append({
            "title": highlighted_title,
            "acronym": hit["_source"].get("projectAcronym", "N/A"),
            "summary": highlighted_summary,
            "highlighted_content": highlighted_content,
            "url": hit["_source"].get("URL", "N/A"),
            "topics": hit["_source"].get("topics", []) if isinstance(hit["_source"].get("topics"), list) else [],
            "subtopics": hit["_source"].get("subtopics", []) if isinstance(hit["_source"].get("subtopics"), list) else [],
            "languages": hit["_source"].get("languages", []) if isinstance(hit["_source"].get("languages"), list) else [],
            "fileType": hit["_source"].get("fileType", "N/A"),
            "locations": hit["_source"].get("locations", []) if isinstance(hit["_source"].get("locations"), list) else [],
            "raw_score": round(raw_score, 4),
            "normalised_score": round(normalised_score, 4)
        })

    return {
        "query": query,
        "total_results": len(results),
        "results": results
    }

