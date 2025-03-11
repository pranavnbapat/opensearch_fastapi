# services/opensearch_service.py

import datetime
import os

from dotenv import load_dotenv
from opensearchpy import OpenSearch, RequestsHttpConnection
from models.embedding import select_model, generate_vector
from services.language_detect import detect_language, translate_text_with_backoff

load_dotenv()

# Fetch OpenSearch credentials
OPENSEARCH_API = os.getenv("OPENSEARCH_API")
OPENSEARCH_USR = os.getenv("OPENSEARCH_USR")
OPENSEARCH_PWD = os.getenv("OPENSEARCH_PWD")

if not all([OPENSEARCH_API, OPENSEARCH_USR, OPENSEARCH_PWD]):
    raise EnvironmentError("Missing OpenSearch environment variables!")

# OpenSearch Client
client = OpenSearch(
    hosts=[{"host": OPENSEARCH_API, "port": 443}],
    http_auth=(OPENSEARCH_USR, OPENSEARCH_PWD),
    use_ssl=True,
    verify_certs=True,
    http_compress=True,
    connection_class=RequestsHttpConnection,
    max_retries=3,
    retry_on_timeout=True
)

def search_opensearch(index_name: str, query_body: dict):
    try:
        response = client.search(index=index_name, body=query_body)
        return response
    except Exception as e:
        raise RuntimeError(f"OpenSearch query failed: {e}")

# Convert all filters to lowercase to match OpenSearch indexing
def lowercase_list(values):
    return [v.lower() for v in values] if values else []

# def apply_filters(request):
#     """
#     Convert all filters to lowercase and return formatted filter conditions for OpenSearch.
#     """
#     def lowercase_list(values):
#         return [v.lower() for v in values] if values else []
#
#     filter_conditions = []
#
#     if request.topics:
#         filter_conditions.append({"terms": {"topics": lowercase_list(request.topics)}})
#
#     if request.subtopics:
#         filter_conditions.append({"terms": {"subtopics": lowercase_list(request.subtopics)}})
#
#     if request.languages:
#         filter_conditions.append({"terms": {"languages": lowercase_list(request.languages)}})
#
#     if request.fileType:
#         filter_conditions.append({"terms": {"fileType": lowercase_list(request.fileType)}})
#
#     if request.locations:
#         filter_conditions.append({"terms": {"locations": lowercase_list(request.locations)}})
#
#     return filter_conditions
#
#
# def perform_search(index_name, query_vector, query_text, filter_conditions, from_offset=0, size=10):
#     """
#     Perform search in OpenSearch using both vector search and keyword-based BM25.
#     """
#     bool_query = {
#         "must": [],
#         "should": [
#             {
#                 "script_score": {
#                     "query": {"exists": {"field": "vector_embedding"}},
#                     "script": {
#                         "source": """
#                             return cosineSimilarity(params.query_vector, doc['vector_embedding']) + 1.0;
#                         """,
#                         "params": {"query_vector": query_vector}
#                     }
#                 }
#             },
#             {"term": {"keywords.raw": {"value": query_text, "boost": 6}}},
#             {"match": {"keywords": {"query": query_text, "boost": 5}}},
#             {"term": {"projectAcronym.raw": {"value": query_text, "boost": 4}}},
#             {"match": {"projectAcronym": {"query": query_text, "boost": 3}}},
#             {"term": {"locations.raw": {"value": query_text, "boost": 3}}},
#             {"match": {"locations": {"query": query_text, "boost": 2}}},
#         ],
#         "minimum_should_match": 1,
#     }
#
#     # Apply filters if present
#     if filter_conditions:
#         bool_query.setdefault("filter", []).extend(filter_conditions)
#
#     search_query = {
#         "track_total_hits": True,
#         "size": size,
#         "from": from_offset,
#         "sort": [{"_score": "desc"}],
#         "query": {"bool": bool_query},
#     }
#
#     return client.search(index=index_name, body=search_query)
#
#
# def extract_results(response):
#     """
#     Extract relevant information from OpenSearch response.
#     """
#     all_results = response["hits"]["hits"]
#
#     # Extract raw scores
#     scores = [hit["_score"] for hit in all_results]
#     min_score = min(scores) if scores else 0
#     max_score = max(scores) if scores else 1  # Avoid division by zero
#
#     results = []
#     for hit in all_results:
#         raw_score = hit["_score"]
#         normalised_score = (raw_score - min_score) / (max_score - min_score) if max_score != min_score else 1.0
#
#         results.append({
#             "title": hit["_source"].get("title_display", hit["_source"].get("title", "Untitled")),
#             "summary": hit["_source"].get("summary_display", hit["_source"].get("summary", "No summary available")),
#             "acronym": hit["_source"].get("projectAcronym_display", hit["_source"].get("projectAcronym", "N/A")),
#             "projectName": hit["_source"].get("projectName_display", hit["_source"].get("projectName", "N/A")),
#             "keywords": hit["_source"].get("keywords_display", hit["_source"].get("keywords", [])),
#             "locations": hit["_source"].get("locations_display", hit["_source"].get("locations", [])),
#             "topics": hit["_source"].get("topics_display", hit["_source"].get("topics", [])),
#             "subtopics": hit["_source"].get("subtopics_display", hit["_source"].get("subtopics", [])),
#             "languages": hit["_source"].get("languages_display", hit["_source"].get("languages", [])),
#             "fileType": hit["_source"].get("fileType_display", hit["_source"].get("fileType", "N/A")),
#             "dateCreated": hit["_source"].get("dateCreated", "N/A"),
#             "creator_name": hit["_source"].get("creator_name", "N/A"),
#             "url": hit["_source"].get("URL", "N/A"),
#             "raw_score": round(raw_score, 4),
#             "normalised_score": round(normalised_score, 4)
#         })
#
#     return results
#
#
# def search_with_fallback(request, threshold=50):
#     """
#     Perform search using the original query first. If results are below threshold, translate and retry in English.
#     """
#     query = request.search_term.strip()
#     selected_model = request.model
#     detected_lang = detect_language(query)
#
#     # Select model and index
#     model, index_name = select_model(selected_model)
#     query_vector = generate_vector(model, query)
#
#     # Get filter conditions (this will be used for both original and translated queries)
#     filter_conditions = apply_filters(request)
#
#     # ✅ Step 1: Perform search in the original language
#     response_original = perform_search(index_name, query_vector, query, filter_conditions)
#     total_results_original = response_original["hits"]["total"]["value"]
#
#     # ✅ Step 2: Log search query **ONLY ONCE** before translation
#     log_entry = {
#         "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
#         "search_term": query,
#         "translated_query": None,  # No translation yet
#         "translated_search_performed": False,
#         "model": selected_model,
#         "filters": request.dict(exclude_none=True),
#         "total_results": total_results_original
#     }
#
#     try:
#         client.index(index="search_logs", body=log_entry)
#     except Exception as e:
#         print(f"Error logging search query: {e}")
#
#     translated_results = []
#     translated_query = None
#     response_translated = None  # ✅ Ensure it's always defined
#
#     # ✅ Step 3: If results are too low, translate and search in English
#     if detected_lang != "en" and total_results_original < threshold:
#         translated_query = translate_text_with_backoff(query, "en")
#         query_vector_translated = generate_vector(model, translated_query)
#
#         # ✅ Step 4: Apply the same filters to the translated query
#         response_translated = perform_search(index_name, query_vector_translated, translated_query, filter_conditions)
#         translated_results = response_translated["hits"]["hits"]
#
#     # ✅ Step 5: Extract and merge results (prioritizing native language results first)
#     original_results = extract_results(response_original)
#     translated_results = extract_results(response_translated) if response_translated else []
#
#     final_results = original_results + translated_results
#
#     return {
#         "results": final_results,
#         "translated_search_performed": detected_lang != "en" and total_results_original < threshold,
#         "original_query": query,
#         "translated_query": translated_query
#     }
#
