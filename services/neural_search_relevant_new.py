# services/neural_search_relevant_new.py

from pydantic import BaseModel
from services.utils import (PAGE_SIZE, remove_stopwords_from_query, K_VALUE, client,
                            group_hits_by_parent)
from typing import List, Optional, Dict, Any


class RelevantSearchRequestNew(BaseModel):
    search_term: str
    topics: Optional[List[str]] = None
    themes: Optional[List[str]] = None
    languages: Optional[List[str]] = None
    category: Optional[List[str]] = None
    project_type: Optional[List[str]] = None
    project_acronym: Optional[List[str]] = None
    locations: Optional[List[str]] = None
    page: Optional[int] = 1
    dev: Optional[bool] = True
    k: Optional[int] = None
    model: Optional[str] = "msmarco"
    include_fulltext: Optional[bool] = False


def neural_search_relevant_new(
        index_name: str,
        query: str,
        filters: Dict[str, Any],
        page: int,
        model_id: str,
        use_semantic: bool = True
    ):
    """
    Perform semantic (neural) or BM25-based search against OpenSearch.
    - Semantic branch: multiple neural clauses + a light BM25 safety net over TEXT fields.
    - BM25 branch: multi_match over TEXT fields + small neural assist on content.
    - Exact acronym matches are handled via a boosted term query on project_acronym (keyword field).
    """

    # Pagination offset
    from_offset = (page - 1) * PAGE_SIZE

    # Filter conditions
    filter_conditions = []
    if filters.get("topics"):
        filter_conditions.append({"terms": {"topics": filters["topics"]}})
    if filters.get("themes"):
        filter_conditions.append({"terms": {"themes": (filters["themes"])}})
    if filters.get("languages"):
        filter_conditions.append({"terms": {"languages": (filters["languages"])}})
    if filters.get("locations"):
        filter_conditions.append({"terms": {"locations": filters["locations"]}})
    if filters.get("category"):
        filter_conditions.append({"terms": {"category": filters["category"]}})
    if filters.get("project_type"):
        filter_conditions.append({"terms": {"project_type": filters["project_type"]}})
    if filters.get("project_acronym"):
        filter_conditions.append({"terms": {"project_acronym": filters["project_acronym"]}})

    # Decide query type based on whether search_term is provided
    if not query:
        query_part: Dict[str, Any] = {"match_all": {}}
    elif use_semantic:
        # Neural semantic search
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
                            "subtitle_embedding": {
                                "query_text": query,
                                "model_id": model_id,
                                "k": K_VALUE
                            }
                        }
                    },
                    {
                        "neural": {
                            "description_embedding": {
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
                    {
                        "multi_match": {
                            "query": remove_stopwords_from_query(query),
                            "fields": [
                                # TEXT fields only
                                "project_name^9",
                                "title^8",
                                "subtitle^7",
                                "keywords^7",
                                "description^6",
                                "content_chunk^5"
                            ],
                            "operator": "and",
                            "type": "best_fields",
                            "boost": 0.3
                        }
                    },
                    # Optional: exact acronym “term” on keyword field
                    {"term": {"project_acronym": {"value": query, "boost": 6.0}}},
                ],
                "minimum_should_match": 1  # At least one should match
            }
        }
    else:
        filtered_query = remove_stopwords_from_query(query)

        query_part = {
            "bool": {
                "should": [
                    {
                        "multi_match": {
                            "query": filtered_query,
                            # TEXT fields only
                            "fields": [
                                "project_name^9",
                                "title^8",
                                "subtitle^7",
                                "keywords^7",
                                "description^6",
                                "content_chunk^5"
                            ],
                            "operator": "and",
                            "type": "best_fields",
                            "boost": 1.0
                        }
                    },
                    # OPTIONAL: exact acronym boost (only helps if the user typed the exact acronym)
                    {"term": {"project_acronym": {"value": filtered_query, "boost": 6.0}}},
                    {
                        "neural": {
                            "content_embedding": {
                                "query_text": query,
                                "model_id": model_id,
                                "k": K_VALUE,
                                "boost": 0.3
                            }
                        }
                    }
                ],
                "minimum_should_match": 1
            }
        }


    # Final OpenSearch query
    search_query = {
        "_source": {
            "includes": [
                "parent_id", "project_name", "project_acronym", "title", "subtitle", "description",
                "keywords", "topics", "themes", "locations", "languages", "category", "subcategories",
                "date_of_completion", "creators", "intended_purposes", "project_id", "project_type",
                "project_url", "@id", "_orig_id",
            ],
            "excludes": [
                # vector fields
                "title_embedding",
                "subtitle_embedding",
                "description_embedding",
                "keywords_embedding",
                "locations_embedding",
                "topics_embedding",
                "content_embedding",
                "project_embedding",

                # raw embedding inputs (not needed in responses)
                "content_embedding_input",
                "keywords_embedding_input",
                "description_embedding_input",
                "title_embedding_input",
                "subtitle_embedding_input",
                "project_embedding_input",

                "content_chunk"
            ]
        },
        "track_total_hits": True,
        "size": PAGE_SIZE,
        "from": from_offset,
        "sort": [{"_score": "desc"}],
        "query": {
            "bool": {
                "must": query_part,
                "filter": filter_conditions,
                "must_not": [
                    {"term": {"chunk_index": -1}}
                ]
            }
        },
        "aggs": {
            "top_projects": {
                "terms": {
                    "field": "project_id",
                    "size": 3,
                    "order": { "unique_parents": "desc" }
                },
                "aggs": {
                    "unique_parents": {
                        "cardinality": {
                            "field": "parent_id",
                            "precision_threshold": 40000
                        }
                    }
                }
            }
        },
    }

    raw_fetch_size = PAGE_SIZE * 40
    search_query["size"] = raw_fetch_size
    search_query["from"] = 0

    response = client.search(index=index_name, body=search_query)

    hits = response["hits"]["hits"]
    grouped = group_hits_by_parent(hits, parents_size=PAGE_SIZE, top_k_snippets=3)

    # Overwrite the original response shape to your controller’s expectations
    response["grouped"] = grouped

    return response
