# app.py

import logging

from datetime import datetime
from fastapi import FastAPI, Request
from starlette.middleware.cors import CORSMiddleware

#from models.embedding import select_model, generate_vector, generate_vector_neural_search
# from services.language_detect import detect_language, translate_text_with_backoff, DEEPL_SUPPORTED_LANGUAGES
from services.neural_search_relevant import neural_search_relevant, RelevantSearchRequest
from services.neural_search_relevant_new import neural_search_relevant_new, RelevantSearchRequestNew
# from services.project_search import project_search, ProjectSearchRequest
from services.recommender import recommend_similar, RecommenderRequest, recommend_similar_cos
from services.hybrid_search import hybrid_search_local, hybrid_search
# from services.validate_and_analyse_results import analyze_search_results
from services.utils import (PAGE_SIZE, BASIC_AUTH_PASS, BASIC_AUTH_USER, MODEL_CONFIG, MultiUserTimedAuthMiddleware,
                            format_results_neural_search, fetch_chunks_for_parents)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ALLOWED_USERS = {
    BASIC_AUTH_USER: {
        "password": BASIC_AUTH_PASS,
        "expires": None
    },
    "reviewer": {
        "password": "ItRWu8Y4jX1L",
        "expires": datetime(2025, 7, 31, 23, 59, 59)
    }
}

app = FastAPI(title="OpenSearch API", version="1.0")
# app.add_middleware(BasicAuthMiddleware, username=BASIC_AUTH_USER, password=BASIC_AUTH_PASS)
app.add_middleware(MultiUserTimedAuthMiddleware, users=ALLOWED_USERS)

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

@app.post("/neural_search_relevant", tags=["Search"],
          summary="Context-aware neural search with smart fallback",
          description="""Performs semantic relevance-based search using one of the supported models and retrieves 
          contextually matched documents. Supported semantic models:
          - `msmarco` (default)
          - `mpnetv2`
          - `minilml12v2`
          You can optionally pass `model` (default is `msmarco`) and `k` to get only the top-k ranked results 
          (no pagination).""")
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
        "themes": request.themes,
        "languages": request.languages,
        "category": request.category,
        "project_type": request.project_type,
        "projectAcronym": request.projectAcronym,
        "locations": request.locations
    }

    # Smart fallback to BM25 if query is short
    if len(query.split()) <= 5:
        logger.info("Short query detected, switching to BM25")
        use_semantic = False
    else:
        use_semantic = True

    model_key = request.model.lower().strip() if request.model else "msmarco"
    model_config = MODEL_CONFIG.get(model_key, MODEL_CONFIG["msmarco"])
    index_name = model_config["index"]

    if request.dev:
        index_name += "_dev"

    model_id = model_config["model_id"]

    response = neural_search_relevant(
        index_name=index_name,
        query=query,
        filters=filters,
        page=page_number,
        model_id=model_id,
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
        project_id = source.get("project_id", "Unknown ID")

        project_counter[project_id] = project_counter.get(project_id, 0) + 1

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
            "project_id": acronym,
            "count": bucket["doc_count"]
        })
    ####################

    # Perform Analysis on Search Results
    # analysis = analyze_search_results(results)

    formatted_results = [format_results_neural_search(hit) for hit in results]

    # If k is provided, override pagination and return top-k only
    if request.k is not None and request.k > 0:
        formatted_results = formatted_results[:request.k]
        pagination = {
            "total_records": len(formatted_results),
            "current_page": 1,
            "total_pages": 1,
            "next_page": None,
            "prev_page": None
        }
    else:
        pagination = {
            "total_records": total_results,
            "current_page": page_number,
            "total_pages": total_pages,
            "next_page": page_number + 1 if page_number < total_pages else None,
            "prev_page": page_number - 1 if page_number > 1 else None
        }

    response_json = {
        "data": formatted_results,
        "related_projects_from_this_page": [
            {
                "project_id": pid,
                "count": count
            }
            for pid, count in top_3_projects
        ],
        "related_projects_from_entire_resultset": related_projects_2,
        "pagination": pagination
    }

    logger.info(f"Search Query: '{query}', Semantic: {use_semantic}, Index: {index_name}, Page: {page_number}")

    return response_json


@app.post("/neural_search_relevant_new", tags=["Search"],
          summary="Context-aware neural search with smart fallback",
          description="""Performs semantic relevance-based search using one of the supported models and retrieves 
          contextually matched documents. Supported semantic models:
          - `msmarco` (default)
          - `mpnetv2`
          - `minilml12v2`
          You can optionally pass `model` (default is `msmarco`) and `k` to get only the top-k ranked results 
          (no pagination).""")
async def neural_search_relevant_endpoint_new(request_temp: Request, request: RelevantSearchRequestNew):
    page_number = max(request.page, 1)

    query = request.search_term.strip()

    filters = {
        "topics": request.topics,
        "themes": request.themes,
        "languages": request.languages,
        "category": request.category,
        "project_type": request.project_type,
        "projectAcronym": request.projectAcronym,
        "locations": request.locations
    }

    # Smart fallback to BM25 if query is short
    if len(query.split()) <= 5:
        logger.info("Short query detected, switching to BM25")
        use_semantic = False
    else:
        use_semantic = True

    model_key = request.model.lower().strip() if request.model else "msmarco"
    model_config = MODEL_CONFIG.get(model_key, MODEL_CONFIG["msmarco"])
    index_name = model_config["index"]

    if request.dev:
        index_name += "_dev"

    model_id = model_config["model_id"]

    response = neural_search_relevant_new(
        index_name=index_name,
        query=query,
        filters=filters,
        page=page_number,
        model_id=model_id,
        use_semantic=use_semantic
    )

    grouped = response.get("grouped", {})
    parents = grouped.get("parents", [])
    total_parents = grouped.get("total_parents", len(parents))

    # Get full texts for these parents
    # parent_ids = [p["parent_id"] for p in parents]
    # chunks_map = fetch_chunks_for_parents(index_name, parent_ids) if parent_ids else {}

    # Parent-level “formatted” result objects
    formatted_results = []
    for p in parents:
        pid = p.get("parent_id")
        # Convert chunks -> array of strings (like ko_content_flat was originally)
        # ko_chunks = [c["content"] for c in chunks_map.get(pid, [])]

        formatted_results.append({
            "_id": pid,
            "_score": p.get("max_score"),
            "title": p.get("title"),
            "subtitle": p.get("subtitle") or "",
            "description": p.get("description"),
            "projectAcronym": p.get("projectAcronym"),
            "projectName": p.get("projectName"),
            "project_type": p.get("project_type"),
            "project_id": p.get("project_id"),
            "topics": p.get("topics") or [],
            "themes": p.get("themes") or [],
            "keywords": p.get("keywords") or [],
            "languages": p.get("languages") or [],
            "locations": p.get("locations") or [],
            "category": p.get("category"),
            "subcategories": p.get("subcategories") or [],
            "creators": p.get("creators") or [],
            "date_of_completion": p.get("date_of_completion"),
            # "ko_content_flat": ko_chunks,
            "@id": p.get("@id"),
            "_tags": p.get("keywords") or []
        })

    # k override still applies (now to parent results)
    if request.k is not None and request.k > 0:
        formatted_results = formatted_results[:request.k]
        pagination = {
            "total_records": len(formatted_results),
            "current_page": 1,
            "total_pages": 1,
            "next_page": None,
            "prev_page": None
        }
    else:
        total_pages = (total_parents + PAGE_SIZE - 1) // PAGE_SIZE
        pagination = {
            "total_records": total_parents,
            "current_page": page_number,
            "total_pages": total_pages,
            "next_page": page_number + 1 if page_number < total_pages else None,
            "prev_page": page_number - 1 if page_number > 1 else None
        }

    page_counts = {}
    for item in formatted_results:  # use 'parents' if you want pre-k truncation
        pid = item.get("project_id")
        if pid:
            page_counts[pid] = page_counts.get(pid, 0) + 1

    related_projects_from_this_page = [
        {"project_id": k, "count": v}
        for k, v in sorted(page_counts.items(), key=lambda x: x[1], reverse=True)[:3]  # drop [:3] to return all
    ]

    aggs = response.get("aggregations", {})
    buckets = aggs.get("top_projects", {}).get("buckets", [])
    related_projects_all = [
        {
            "project_id": b.get("key"),
            "count": b.get("unique_parents", {}).get("value", 0)
        }
        for b in buckets
    ]

    # project counts: your earlier logic can stay, but now work off parent-level data if you wish.
    response_json = {
        "data": formatted_results,
        "related_projects_from_this_page": related_projects_from_this_page,
        "related_projects_from_entire_resultset": related_projects_all,
        "pagination": pagination
    }

    logger.info(f"Search Query: '{query}', Semantic: {use_semantic}, Index: {index_name}, Page: {page_number}")

    return response_json


@app.post("/recommend", tags=["Recommender System"],
          summary="Vector-based content recommender",
          description="Returns top-k most similar knowledge objects using vector similarity search. Embeddings are "
                      "generated from textual metadata (title, summary, keywords, topics, etc.) and stored in "
                      "OpenSearch. Recommendations are retrieved using KNN search over pre-computed embeddings." 
                      "Models supported: mpnet, minilm, e5, bge, distilbert.")
def recommend_endpoint(data: RecommenderRequest):
    return recommend_similar(text=data.text, top_k=data.top_k, model_name=data.model_name)


@app.post("/recommend_cos", tags=["Recommender System"],
          summary="Recommender with cosine similarity reranking",
          description="Improves recommendation quality by reranking OpenSearch results using cosine similarity. "
                      "First retrieves a larger set of candidates via KNN, then ranks them locally using a "
                      "transformer-based embedding model to return the most semantically similar items. "
                      "Best used when precision matters. "
                      "Models supported: mpnet, minilm, e5, bge, distilbert.")
def recommend_cos_endpoint(data: RecommenderRequest):
    return recommend_similar_cos(text=data.text, top_k=data.top_k, model_name=data.model_name)


@app.post("/hybrid_search_local", tags=["Hybrid Search"],
          summary="Hybrid search with local reranking (MiniLM, MPNet, MS MARCO)",
          description="""Performs hybrid search by retrieving results using BM25 keyword matching, then reranks them 
          locally using cosine similarity with a transformer model. You can choose from the following reranking models:
          - `msmarco` (default)
          - `mpnetv2`
          - `minilml12v2`
          These models rerank top BM25 candidates based on semantic similarity. If no model is specified, 
          MS MARCO is used by default.""")
async def hybrid_search_local_endpoint(request: RelevantSearchRequest):
    page_number = max(request.page, 1)
    query = request.search_term.strip()

    filters = {
        "topics": request.topics,
        "subtopics": request.subtopics,
        "languages": request.languages,
        "fileType": request.fileType,
        "project_type": request.project_type,
        "projectAcronym": request.projectAcronym,
        "locations": request.locations
    }

    # index_name = "neural_search_index_dev" if request.dev else "neural_search_index"

    model_key = request.model.lower().strip() if request.model else "msmarco"
    model_config = MODEL_CONFIG.get(model_key, MODEL_CONFIG["msmarco"])
    index_name = model_config["index"]

    result = hybrid_search_local(
        index_name=index_name,
        query=query,
        filters=filters,
        page=page_number,
        model_key=model_key
    )

    return {
        "data": result["results"],
        "pagination": {
            "total_records": result["total"],
            "current_page": result["page"],
            "total_pages": result["total_pages"],
            "next_page": result["page"] + 1 if result["page"] < result["total_pages"] else None,
            "prev_page": result["page"] - 1 if result["page"] > 1 else None
        }
    }


@app.post("/hybrid_search", tags=["Hybrid Search"],
          summary="Hybrid search using OpenSearch BM25 + neural",
          description="""Performs hybrid search using OpenSearch's built-in hybrid scoring mechanism that combines BM25 
          (lexical keyword match) with neural semantic similarity.
          You can select one of the following semantic embedding models:
          - `msmarco` (default)
          - `mpnetv2`
          - `minilml12v2`
          If no model is specified in the request, the system defaults to MS MARCO.
          """)
async def hybrid_search_endpoint(request: RelevantSearchRequest):
    page_number = max(request.page, 1)
    query = request.search_term.strip()

    filters = {
        "topics": request.topics,
        "subtopics": request.subtopics,
        "languages": request.languages,
        "fileType": request.fileType,
        "project_type": request.project_type,
        "projectAcronym": request.projectAcronym,
        "locations": request.locations
    }

    # index_name = "neural_search_index_dev" if request.dev else "neural_search_index"

    model_key = request.model.lower().strip() if request.model else "msmarco"
    model_config = MODEL_CONFIG.get(model_key, MODEL_CONFIG["msmarco"])
    index_name = model_config["index"]
    model_id = model_config["model_id"]

    result = hybrid_search(
        index_name=index_name,
        query=query,
        filters=filters,
        page=page_number,
        model_id=model_id
    )

    return {
        "data": result["results"],
        "pagination": {
            "total_records": result["total"],
            "current_page": result["page"],
            "total_pages": result["total_pages"],
            "next_page": result["page"] + 1 if result["page"] < result["total_pages"] else None,
            "prev_page": result["page"] - 1 if result["page"] > 1 else None
        }
    }
