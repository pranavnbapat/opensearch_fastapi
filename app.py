from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from starlette.middleware.cors import CORSMiddleware

from models.embedding import select_model, generate_vector
from services.opensearch_service import search_opensearch

app = FastAPI(title="OpenSearch API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Model
class SearchRequest(BaseModel):
    search_term: str
    model: str

@app.post("/search", response_model=Dict[str, Any])
async def search_endpoint(request: SearchRequest):
    query = request.search_term.strip()
    selected_model = request.model

    if not query:
        raise HTTPException(status_code=400, detail="No search term provided")

    model, index_name = select_model(selected_model)
    query_vector = generate_vector(model, query)

    # OpenSearch Query
    search_query = {
        "query": {
            "bool": {
                "should": [
                    {
                        "knn": {
                            "vector_embedding": {
                                "vector": query_vector,
                                "k": 10
                            }
                        }
                    },
                    {
                        "bool": {
                            "should": [
                                # Exact Match (Boosted)
                                {"term": {"title": {"value": query, "boost": 5}}},
                                {"term": {"projectAcronym": {"value": query, "boost": 4}}},
                                {"term": {"projectName.raw": {"value": query, "boost": 3}}},
                                {"term": {"keywords.raw": {"value": query, "boost": 3}}},
                                {"term": {"topics.raw": {"value": query, "boost": 2}}},
                                {"term": {"subtopics.raw": {"value": query, "boost": 2}}},
                                {"term": {"object_name": {"value": query, "boost": 2}}},

                                # Full-Text Match with Fuzziness (Handles Typos)
                                {"match": {"title": {"query": query, "fuzziness": "AUTO"}}},
                                {"match": {"summary": {"query": query, "fuzziness": "AUTO"}}},
                                {"match": {"content_pages": {"query": query, "fuzziness": "AUTO"}}},
                                {"match": {"projectName": {"query": query, "fuzziness": "AUTO"}}},

                                # N-Gram Matching (Autocomplete & Partial Words)
                                {"match": {"title.ngram": {"query": query}}},
                                {"match": {"summary.ngram": {"query": query}}},
                                {"match": {"content_pages.ngram": {"query": query}}},
                                {"match": {"projectName.ngram": {"query": query}}},
                                {"match": {"topics.ngram": {"query": query}}},
                                {"match": {"subtopics.ngram": {"query": query}}},

                                # Wildcard (Prefix Match)
                                {"wildcard": {"title.raw": {"value": f"{query.lower()}*"}}},
                                {"wildcard": {"summary.raw": {"value": f"{query.lower()}*"}}},

                                # Keyword Exact Match for Filters
                                {"term": {"fileTypeCategories": {"value": query}}},
                                {"term": {"fileType": {"value": query}}},
                                {"term": {"locations": {"value": query}}},
                                {"term": {"languages": {"value": query}}},
                            ],
                            "minimum_should_match": 1  # Ensure at least one match type works
                        }
                    }
                ]
            }
        },
        "highlight": {
            "fields": {
                "title": {},
                "summary": {},
                "content_pages": {},
                "projectName": {},
                "topics": {},
                "subtopics": {}
            },
            "pre_tags": ["<mark>"],
            "post_tags": ["</mark>"]
        },
        "sort": [
            {"_score": "desc"},  # Sort by highest relevance
            # {"startDate": "desc"}  # Then sort by latest start date
        ]
    }

    # Execute search
    response = search_opensearch(index_name, search_query)

    # Process Results
    results = []
    scores = [hit["_score"] for hit in response["hits"]["hits"]]

    # Min-Max Normalisation
    min_score = min(scores) if scores else 0
    max_score = max(scores) if scores else 1  # Avoid division by zero



    for hit in response["hits"]["hits"]:
        highlighted_title = hit.get("highlight", {}).get("title", [hit["_source"]["title"]])[0]
        highlighted_summary = hit.get("highlight", {}).get("summary", [hit["_source"].get("summary", "N/A")])[0]
        raw_score = hit["_score"]

        # Normalise the score
        if max_score != min_score:
            normalised_score = (raw_score - min_score) / (max_score - min_score)
        else:
            normalised_score = 1.0  # All scores are the same

        results.append({
            "title": highlighted_title,
            "acronym": hit["_source"].get("projectAcronym", "N/A"),
            "summary": highlighted_summary,
            "url": hit["_source"].get("URL", "N/A"),
            "raw_score": round(raw_score, 4),
            "normalised_score": round(normalised_score, 4)
        })

    return {"query": query, "results": results}
