# app.py

import datetime
import os
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Optional
from starlette.middleware.cors import CORSMiddleware

#from models.embedding import select_model, generate_vector, generate_vector_neural_search
from services.language_detect import detect_language, translate_text_with_backoff, DEEPL_SUPPORTED_LANGUAGES
from services.neural_search_relevant import neural_search_relevant, RelevantSearchRequest
from services.project_search import project_search, ProjectSearchRequest
# from services.neural_search_knn import neural_search_knn, KNNSearchRequest, MODEL_IDS_FOR_NS
# from services.validate_and_analyse_results import analyze_search_results
from services.utils import PAGE_SIZE

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

@app.post("/neural_search_relevant")
async def neural_search_relevant_endpoint(request_temp: Request, request: RelevantSearchRequest):
    # client_host = request_temp.client.host
    # user_agent = request_temp.headers.get("user-agent")
    # referer = request_temp.headers.get("referer")
    # origin = request_temp.headers.get("origin")
    # full_url = str(request_temp.url)
    #
    # print("###############")
    # logger.info(f"Search request from IP: {client_host}")
    # logger.info(f"User-Agent: {user_agent}")
    # logger.info(f"Referer: {referer}")
    # logger.info(f"Origin: {origin}")
    # logger.info(f"Full request URL: {full_url}")
    # print("\n\n")

    # logger.info(
    #     f"Search Query: '{request.search_term}', Semantic: {request.use_semantic}, Index: {'neural_search_index_dev' if request.dev else 'neural_search_index'}, Page: {max(request.page, 1)}")

    page_number = max(request.page, 1)

    query = request.search_term.strip()
    # detected_lang = detect_language(query)

    # Translate if not English
    # if detected_lang != "en" and detected_lang.upper() in DEEPL_SUPPORTED_LANGUAGES:
    #     try:
    #         query = translate_text_with_backoff(query, target_language="EN")
    #         logger.info(f"Translated query to English: {query}")
    #     except Exception as e:
    #         logger.error(f"Failed to translate non-English query: {e}")
    # else:
    #     logger.info(f"Skipping translation for language: {detected_lang}")

    filters = {
        "topics": request.topics,
        "subtopics": request.subtopics,
        "languages": request.languages,
        "fileType": request.fileType,
        "project_type": request.project_type,
        "projectAcronym": request.projectAcronym,
        "locations": request.locations
    }

    index_name = "neural_search_index_dev" if request.dev else "neural_search_index"

    # Smart fallback to BM25 if query is short
    if len(query.split()) <= 5:
        logger.info("Short query detected, switching to BM25")
        use_semantic = False
    else:
        use_semantic = True

    response = neural_search_relevant(
        index_name=index_name,
        query=query,
        filters=filters,
        page=page_number,
        use_semantic=use_semantic
    )

    total_results = response["hits"]["total"]["value"]
    total_pages = (total_results + PAGE_SIZE - 1) // PAGE_SIZE
    results = response["hits"]["hits"]

    #################### Code below lists top 3 projects from this page
    # Build project aggregation from search results
    project_counter = {}

    for hit in results:
        source = hit["_source"]
        project_acronym = source.get("projectAcronym", "Unknown Acronym")

        project_counter[project_acronym] = project_counter.get(project_acronym, 0) + 1

    # Sort by count, descending
    sorted_projects = sorted(project_counter.items(), key=lambda x: x[1], reverse=True)

    # Take top 3 only
    top_3_projects = sorted_projects[:3]
    ####################

    ##################### Code below lists top 3 projects from the entire resultset
    aggregations = response.get("aggregations", {})
    top_projects_buckets = aggregations.get("top_projects", {}).get("buckets", [])

    related_projects_2 = []
    for bucket in top_projects_buckets:
        acronym = bucket["key"]
        related_projects_2.append({
            "project_acronym": acronym,
            "count": bucket["doc_count"]
        })
    ####################

    # Perform Analysis on Search Results
    # analysis = analyze_search_results(results)

    formatted_results = []
    for hit in results:
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
        "related_projects_from_this_page": [
            {
                "project_acronym": acronym,
                "count": count
            }
            for acronym, count in top_3_projects
        ],
        "related_projects_from_entire_resultset": related_projects_2,
        "pagination": {
            "total_records": total_results,
            "current_page": page_number,
            "total_pages": total_pages,
            "next_page": page_number + 1 if page_number < total_pages else None,
            "prev_page": page_number - 1 if page_number > 1 else None
        }
    }

    logger.info(f"Search Query: '{query}', Semantic: {use_semantic}, Index: {index_name}, Page: {page_number}")

    return response_json


@app.post("/project_search")
async def project_search_endpoint(request: ProjectSearchRequest):
    query = request.search_term.strip()
    page_number = max(request.page, 1)

    if not query:
        raise HTTPException(status_code=400, detail="Search term cannot be empty.")

    index_name = "projects_index_dev" if request.dev else "projects_index"

    response = project_search(
        index_name=index_name,
        query=query,
        page=page_number
    )

    results = response["hits"]["hits"]
    total_results = response["hits"]["total"]["value"]
    total_pages = (total_results + PAGE_SIZE - 1) // PAGE_SIZE

    formatted = [
        {
            "_id": hit["_id"],
            "_score": hit["_score"],
            "projectName": hit["_source"].get("projectName", ""),
            "projectAcronym": hit["_source"].get("projectAcronym", "")
        }
        for hit in results
    ]

    return {
        "data": formatted,
        "pagination": {
            "total_records": total_results,
            "current_page": page_number,
            "total_pages": total_pages,
            "next_page": page_number + 1 if page_number < total_pages else None,
            "prev_page": page_number - 1 if page_number > 1 else None
        }
    }

# @app.post("/neural_search_knn")
# async def neural_search_knn_endpoint(request: KNNSearchRequest):
#     """Search using k-NN (semantic search) with filters."""
#     if not request.search_term:
#         raise HTTPException(status_code=400, detail="No search term provided")
#
#     model_name = request.model
#
#     # Validate model
#     if model_name not in MODEL_IDS_FOR_NS:
#         raise HTTPException(status_code=400,
#                             detail=f"Invalid model '{model_name}'. Available models: {list(MODEL_IDS_FOR_NS.keys())}")
#
#     model_id = MODEL_IDS_FOR_NS[model_name]
#
#     filters = {
#         "topics": request.topics,
#         "subtopics": request.subtopics,
#         "languages": request.languages,
#         "fileType": request.fileType,
#         "locations": request.locations
#     }
#
#     try:
#         response = neural_search_knn(
#             query=request.search_term,
#             model_name=model_name,
#             filters=filters,
#             page=request.page,
#             model_id=model_id
#         )
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
#
#     total_results = response["hits"]["total"]["value"]
#     results = response["hits"]["hits"]
#     # Perform Analysis on Search Results
#     analysis = analyze_search_results(results)
#
#     return {
#         "total_results": total_results,
#         "results": results,
#         "analysis": analysis,
#         "current_page": request.page
#     }

