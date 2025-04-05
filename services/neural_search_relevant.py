# neural_search_relevant.py

from typing import List, Optional, Dict, Any
from services.utils import lowercase_list, PAGE_SIZE, remove_stopwords_from_query, K_VALUE
from pydantic import BaseModel
from services.opensearch_service import client
import os


class RelevantSearchRequest(BaseModel):
    search_term: str
    topics: Optional[List[str]] = None
    subtopics: Optional[List[str]] = None
    languages: Optional[List[str]] = None
    fileType: Optional[List[str]] = None
    project_type: Optional[List[str]] = None
    projectAcronym: Optional[List[str]] = None
    locations: Optional[List[str]] = None
    page: Optional[int] = 1
    dev: Optional[bool] = False
    use_semantic: Optional[bool] = True


def neural_search_relevant(index_name: str, query: str, filters: Dict[str, Any], page: int, use_semantic: bool = True):
# def neural_search_relevant(index_name, query, filters, page):
    """
    Perform semantic or BM25-based neural search against OpenSearch.

    :param index_name: Name of the OpenSearch index.
    :param query: The raw user query string.
    :param filters: Dictionary of filters (topics, subtopics, etc.).
    :param page: Page number for pagination.
    :param use_semantic: Whether to use OpenSearch's neural search.
    :return: OpenSearch response as JSON.
    """
    # Pagination offset
    from_offset = (page - 1) * PAGE_SIZE

    # Filter conditions
    filter_conditions = []
    if filters.get("topics"):
        filter_conditions.append({"terms": {"topics": filters["topics"]}})
    if filters.get("subtopics"):
        filter_conditions.append({"terms": {"subtopics": lowercase_list(filters["subtopics"])}})
    if filters.get("languages"):
        filter_conditions.append({"terms": {"languages": lowercase_list(filters["languages"])}})
    if filters.get("locations"):
        filter_conditions.append({"terms": {"locations": lowercase_list(filters["locations"])}})
    if filters.get("fileType"):
        filter_conditions.append({"terms": {"fileType": filters["fileType"]}})
    if filters.get("project_type"):
        filter_conditions.append({"terms": {"project_type": filters["project_type"]}})
    if filters.get("projectAcronym"):
        filter_conditions.append({"terms": {"projectAcronym": filters["projectAcronym"]}})

    # Decide query type based on whether search_term is provided
    if not query:
        query_part = {"match_all": {}}  # Retrieve all documents
    elif use_semantic:
        # Neural semantic search
        model_id = os.getenv("OPENSEARCH_MSMARCO_MODEL_ID", "LciGfZUBVa2ERaFSUEya")  # Fallback to default if not set
        query_part = {
            "bool": {
                "should": [
                    {
                        "neural": {
                            "title_embedding": {
                                "query_text": query,
                                "model_id": model_id,
                                "k": K_VALUE
                            }
                        }
                    },
                    {
                        "neural": {
                            "summary_embedding": {
                                "query_text": query,
                                "model_id": model_id,
                                "k": K_VALUE
                            }
                        }
                    },
                    {
                        "neural": {
                            "keywords_embedding": {
                                "query_text": query,
                                "model_id": model_id,
                                "k": K_VALUE
                            }
                        }
                    },
                    {
                        "neural": {
                            "content_embedding": {
                                "query_text": query,
                                "model_id": model_id,
                                "k": K_VALUE
                            }
                        }
                    },
                    {
                        "neural": {
                            "project_embedding": {
                                "query_text": query,
                                "model_id": model_id,
                                "k": K_VALUE
                            }
                        }
                    },
                ],
                "minimum_should_match": 1  # At least one should match
            }
        }
    else:
        filtered_query = remove_stopwords_from_query(query)

        query_part = {
            "multi_match": {
                "query": filtered_query,
                "fields": [
                    "title^9",
                    "content_pages^8",
                    "summary^7",
                    "keywords^6"
                ]
            }
        }


    # Final OpenSearch query
    search_query = {
        "_source": {
            "excludes": [
                "title_embedding",
                "summary_embedding",
                "keywords_embedding",
                "topics_embedding",
                "content_embedding",
                "project_embedding",
                "project_acronym_embedding",
                "content_embedding_input",
                "topics_embedding_input",
                "keywords_embedding_input",
                "content_pages",
                "_orig_id",
            ]
        },
        "track_total_hits": True,
        "size": PAGE_SIZE,
        "from": from_offset,
        "sort": [{"_score": "desc"}],
        "query": {
            "bool": {
                "must": query_part,
                "filter": filter_conditions
            }
        }
    }

    response = client.search(index=index_name, body=search_query)
    return response
