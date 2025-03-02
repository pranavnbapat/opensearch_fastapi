# services/opensearch_service.py

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
#             {"term": {"keywords.raw": {"value": query_text, "boost": 6}}},  # Exact match
#             {"match": {"keywords": {"query": query_text, "boost": 5}}},  # Full-text search
#             {"term": {"projectAcronym.raw": {"value": query_text, "boost": 4}}},  # Exact match
#             {"match": {"projectAcronym": {"query": query_text, "boost": 3}}},  # Full-text search
#             {"term": {"locations.raw": {"value": query_text, "boost": 3}}},  # Exact match
#             {"match": {"locations": {"query": query_text, "boost": 2}}},  # Full-text search
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
# def search_with_fallback(request, threshold=10):
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
#     # Get filter conditions
#     filter_conditions = apply_filters(request)
#
#     # Perform search in the original language
#     response_original = perform_search(index_name, query_vector, query, filter_conditions)
#     total_results_original = response_original["hits"]["total"]["value"]
#
#     translated_results = []
#     translated_query = None
#
#     # If results are too low, translate and search again
#     if detected_lang != "en" and total_results_original < threshold:
#         translated_query = translate_text_with_backoff(query, "en")
#         query_vector_translated = generate_vector(model, translated_query)
#         response_translated = perform_search(index_name, query_vector_translated, translated_query, filter_conditions)
#         translated_results = response_translated["hits"]["hits"]
#
#     # Merge results: Prioritize original search results
#     final_results = response_original["hits"]["hits"] + translated_results
#
#     return {
#         "results": final_results,
#         "translated_search_performed": detected_lang != "en" and total_results_original < threshold,
#         "original_query": query,
#         "translated_query": translated_query if translated_query else None
#     }
