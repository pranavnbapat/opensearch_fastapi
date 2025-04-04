# neural_search_relevant.py

from typing import List, Optional
from services.utils import lowercase_list, PAGE_SIZE, remove_stopwords_from_query
from pydantic import BaseModel
from services.opensearch_service import client


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


def neural_search_relevant(index_name, query, filters, page):
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

    # BM25 Search Query
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
