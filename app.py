# app.py

import datetime
import logging

from fastapi import FastAPI, HTTPException, Request
from starlette.middleware.cors import CORSMiddleware

#from models.embedding import select_model, generate_vector, generate_vector_neural_search
# from services.language_detect import detect_language, translate_text_with_backoff, DEEPL_SUPPORTED_LANGUAGES
from services.neural_search_relevant import neural_search_relevant, RelevantSearchRequest
from services.project_search import project_search, ProjectSearchRequest
from services.recommender import recommend_similar, RecommenderRequest, recommend_similar_cos
from services.hybrid_search import hybrid_search_local, hybrid_search
# from services.validate_and_analyse_results import analyze_search_results
from services.utils import PAGE_SIZE, BasicAuthMiddleware, BASIC_AUTH_PASS, BASIC_AUTH_USER

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="OpenSearch API", version="1.0")
app.add_middleware(BasicAuthMiddleware, username=BASIC_AUTH_USER, password=BASIC_AUTH_PASS)

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
          description="Performs relevance-based search using semantic embedding models (e.g. DistilBERT/MS MARCO) "
                      "to retrieve and rank documents based on contextual similarity to the input query. Automatically "
                      "falls back to BM25 keyword search for short or ambiguous queries. Also includes topic-based "
                      "filtering and project aggregation for enhanced insight.")
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


@app.post("/project_search", tags=["Project Search"],
          summary="Simple keyword-based project search",
          description="Performs a straightforward keyword search over indexed project documents using BM25 scoring. "
                      "Supports pagination and basic project metadata retrieval, such as project name and acronym. "
                      "Best used when searching by known keywords, acronyms, or exact terms.")
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


@app.post("/recommend", tags=["Recommender System"],
          summary="Vector-based content recommender",
          description="Returns top-k most similar knowledge objects using vector similarity search. Embeddings are "
                      "generated from textual metadata (title, summary, keywords, topics, etc.) and stored in "
                      "OpenSearch. Recommendations are retrieved using KNN search over pre-computed embeddings.")
def recommend_endpoint(data: RecommenderRequest):
    return recommend_similar(text=data.text, top_k=data.top_k)


@app.post("/recommend_cos", tags=["Recommender System"],
          summary="Recommender with cosine similarity reranking",
          description="Improves recommendation quality by reranking OpenSearch results using cosine similarity. "
                      "First retrieves a larger set of candidates via KNN, then ranks them locally using a "
                      "transformer-based embedding model to return the most semantically similar items. "
                      "Best used when precision matters.")
def recommend_cos_endpoint(data: RecommenderRequest):
    return recommend_similar_cos(text=data.text, top_k=data.top_k)


@app.post("/hybrid_search_local", tags=["Hybrid Search"],
          summary="Hybrid search with local reranking",
          description="Performs hybrid search by retrieving results using BM25 keyword matching, then reranks them "
                      "locally using cosine similarity with a transformer model (MS MARCO DistilBERT). This ensures "
                      "high recall with precise semantic ranking, ideal for longer queries with domain-specific "
                      "context.")
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

    index_name = "neural_search_index_dev" if request.dev else "neural_search_index"

    result = hybrid_search_local(
        index_name=index_name,
        query=query,
        filters=filters,
        page=page_number
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
          description= "Performs hybrid search using OpenSearch's built-in support for combining lexical (BM25) and "
                       "semantic (neural) search in a single query. This server-side hybrid scoring retrieves "
                       "contextually relevant documents based on both exact keyword matches and semantic understanding "
                       "of the query.")
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

    index_name = "neural_search_index_dev" if request.dev else "neural_search_index"

    result = hybrid_search(
        index_name=index_name,
        query=query,
        filters=filters,
        page=page_number
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
