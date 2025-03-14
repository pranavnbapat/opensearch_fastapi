# app.py

import datetime
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from starlette.middleware.cors import CORSMiddleware

from models.embedding import select_model, generate_vector, generate_vector_neural_search
from services.language_detect import detect_language, translate_text_with_backoff
from services.opensearch_service import search_opensearch, client
from services.neural_search_relevant import neural_search_relevant, RelevantSearchRequest
from services.neural_search_knn import neural_search_knn, KNNSearchRequest, MODEL_IDS_FOR_NS
from services.validate_and_analyse_results import analyze_search_results
from services.utils import lowercase_list, PAGE_SIZE

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
    fileType: Optional[List[str]] = None
    projectAcronym: Optional[str] = None
    locations: Optional[List[str]] = None
    page: Optional[int] = 1  # Default to page 1

@app.post("/search")
async def search_endpoint(request: SearchRequest):
    start_time = time.time()

    query = request.search_term.strip()
    selected_model = request.model

    page_number = max(request.page, 1)  # Ensure page number is always 1 or higher

    if not query:
        raise HTTPException(status_code=400, detail="No search term provided")

    # Detect language
    detected_lang = detect_language(query)

    # If not English, translate to English
    if detected_lang != "en":
        query = translate_text_with_backoff(query, "en")

    model, index_name = select_model(selected_model)
    query_vector = generate_vector(model, query)

    # Calculate pagination offset
    from_offset = (page_number - 1) * PAGE_SIZE

    # Add filters (AND conditions)
    filter_conditions = []

    if request.topics:
        filter_conditions.append({"terms": {"topics": lowercase_list(request.topics)}})

    if request.subtopics:
        filter_conditions.append({"terms": {"subtopics": lowercase_list(request.subtopics)}})

    if request.languages:
        filter_conditions.append({"terms": {"languages": lowercase_list(request.languages)}})

    if request.fileType:
        filter_conditions.append({"terms": {"fileType": lowercase_list(request.fileType)}})

    if request.locations:
        filter_conditions.append({"terms": {"locations": lowercase_list(request.locations)}})

    knn_query = {
        "knn": {
            "vector_embedding": {
                "vector": query_vector,
                "k": 10,  # Get top 10 nearest neighbors
                # **({"filter": {"bool": {"must": filter_conditions}}} if filter_conditions else {})
            }
        }
    }

    # BM25 bool query
    bm25_query = {
        "bool": {
            "should": [
                # BM25 Search (Keyword Filtering)
                {"term": {"keywords.raw": {"value": query, "boost": 6}}},  # Exact match
                {"match": {"keywords": {"query": query, "boost": 5}}},  # Full-text search
                {"term": {"projectAcronym.raw": {"value": query, "boost": 4}}},  # Exact match
                {"match": {"projectAcronym": {"query": query, "boost": 3}}},  # Full-text search
                {"term": {"locations.raw": {"value": query, "boost": 3}}},  # Exact match
                {"match": {"locations": {"query": query, "boost": 2}}},  # Full-text search
            ],
        },
        # "minimum_should_match": 1,
    }

    # Hybrid Query (k-NN + BM25)
    search_query = {
        "track_total_hits": True,
        "size": PAGE_SIZE,
        "from": from_offset,
        "sort": [{"_score": "desc"}],
        "query": {
            "bool": {
                "must": filter_conditions,
                "should": [
                    knn_query,  # k-NN Search
                    bm25_query  # BM25 Search
                ],
                "minimum_should_match": 1
            }
        }
    }

    response = search_opensearch(index_name, search_query)

    total_results = response["hits"]["total"]["value"]
    total_pages = (total_results + PAGE_SIZE - 1) // PAGE_SIZE  # Calculate total pages
    response_time = round((time.time() - start_time) * 1000, 2)  # Convert to ms

    # Log Search Request to OpenSearch
    log_entry = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "search_term": query,
        "model": selected_model,
        "filters": {
            "topics": request.topics,
            "subtopics": request.subtopics,
            "languages": request.languages,
            "fileType": request.fileType,
            "projectAcronym": request.projectAcronym,
            "locations": request.locations
        },
        "total_results": total_results,
        "response_time_ms": response_time
    }

    try:
        client.index(index="search_logs", body=log_entry)
    except Exception as e:
        print(f"Error logging search query: {e}")

    all_results = response["hits"]["hits"]

    # Extract raw scores
    scores = [hit["_score"] for hit in all_results]
    min_score = min(scores) if scores else 0
    max_score = max(scores) if scores else 1  # Avoid division by zero

    results = []
    for hit in all_results:
        raw_score = hit["_score"]

        normalised_score = (raw_score - min_score) / (max_score - min_score) if max_score != min_score else 1.0

        results.append({
            "_id": hit.get("_id", "N/A"),
            "title": hit["_source"].get("title_display", hit["_source"].get("title", "Untitled")),
            "summary": hit["_source"].get("summary_display", hit["_source"].get("summary", "No summary available")),
            "projectAcronym": hit["_source"].get("projectAcronym_display", hit["_source"].get("projectAcronym", "N/A")),
            "projectName": hit["_source"].get("projectName_display", hit["_source"].get("projectName", "N/A")),
            "keywords": hit["_source"].get("keywords_display", hit["_source"].get("keywords", [])),
            "locations": hit["_source"].get("locations_display", hit["_source"].get("locations", [])),
            "topics": hit["_source"].get("topics_display", hit["_source"].get("topics", [])),
            "subtopics": hit["_source"].get("subtopics_display", hit["_source"].get("subtopics", [])),
            "languages": hit["_source"].get("languages_display", hit["_source"].get("languages", [])),
            "fileType": hit["_source"].get("fileType_display", hit["_source"].get("fileType", "N/A")),
            "dateCreated": hit["_source"].get("dateCreated", "N/A"),
            "creators": hit["_source"].get("creators", []),
            "@id": hit["_source"].get("@id", "N/A"),
            "raw_score": round(raw_score, 4),
            "normalised_score": round(normalised_score, 4)
        })

    response_json = {
        "data": results,
        "pagination": {
            "total_records": total_results,
            "current_page": page_number,
            "total_pages": total_pages,
            "next_page": page_number + 1 if page_number < total_pages else None,
            "prev_page": page_number - 1 if page_number > 1 else None
        }
    }

    return response_json


@app.post("/neural_search_relevant")
async def neural_search_relevant_endpoint(request: RelevantSearchRequest):
    """Search using only BM25 (multi_match) with filters."""
    # if not request.search_term:
    #     raise HTTPException(status_code=400, detail="No search term provided")

    page_number = max(request.page, 1)

    query = request.search_term.strip()
    detected_lang = detect_language(query)
    print("detected_lang: ", detected_lang)

    filters = {
        "topics": request.topics,
        "subtopics": request.subtopics,
        "languages": request.languages,
        "fileType": request.fileType,
        "locations": request.locations
    }

    response_original = neural_search_relevant(
        index_name="neural_search_index",
        query=query,
        filters=filters,
        page=page_number
    )

    original_results = response_original["hits"]["hits"]

    english_results = []
    if detected_lang != "en":
        translated_query = translate_text_with_backoff(query, "en")
        print("translated_query: ", translated_query)

        response_english = neural_search_relevant(
            index_name="neural_search_index",
            query=translated_query,
            filters=filters,
            page=page_number
        )
        english_results = response_english["hits"]["hits"]

    all_results = original_results + english_results

    total_results = len(all_results)
    total_pages = (total_results + PAGE_SIZE - 1) // PAGE_SIZE

    # Perform Analysis on Search Results
    # analysis = analyze_search_results(results)

    formatted_results = []
    for hit in all_results:
        source = hit["_source"]

        # Convert dateCreated from YYYY-MM-DD to DD-MM-YYYY
        date_created = source.get("dateCreated", "N/A")
        try:
            formatted_date = datetime.datetime.strptime(date_created, "%Y-%m-%d").strftime("%d-%m-%Y")
        except ValueError:
            formatted_date = date_created  # Keep as is if conversion fails

        # Update result entry with formatted date
        source["dateCreated"] = formatted_date

        # Copy keywords to _tags
        source["_tags"] = source.get("keywords", [])

        # Flatten structure: move _score and _id to the same level as _source contents
        flattened_result = {"_id": hit["_id"], "_score": hit["_score"], **source}
        formatted_results.append(flattened_result)

    response_json = {
        "data": formatted_results,
        "pagination": {
            "total_records": total_results,
            "current_page": page_number,
            "total_pages": total_pages,
            "next_page": page_number + 1 if page_number < total_pages else None,
            "prev_page": page_number - 1 if page_number > 1 else None
        }
    }

    return response_json


@app.post("/neural_search_knn")
async def neural_search_knn_endpoint(request: KNNSearchRequest):
    """Search using k-NN (semantic search) with filters."""
    if not request.search_term:
        raise HTTPException(status_code=400, detail="No search term provided")

    model_name = request.model

    # Validate model
    if model_name not in MODEL_IDS_FOR_NS:
        raise HTTPException(status_code=400,
                            detail=f"Invalid model '{model_name}'. Available models: {list(MODEL_IDS_FOR_NS.keys())}")

    model_id = MODEL_IDS_FOR_NS[model_name]

    filters = {
        "topics": request.topics,
        "subtopics": request.subtopics,
        "languages": request.languages,
        "fileType": request.fileType,
        "locations": request.locations
    }

    try:
        response = neural_search_knn(
            query=request.search_term,
            model_name=model_name,
            filters=filters,
            page=request.page,
            model_id=model_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    total_results = response["hits"]["total"]["value"]
    results = response["hits"]["hits"]
    # Perform Analysis on Search Results
    analysis = analyze_search_results(results)

    return {
        "total_results": total_results,
        "results": results,
        "analysis": analysis,
        "current_page": request.page
    }

