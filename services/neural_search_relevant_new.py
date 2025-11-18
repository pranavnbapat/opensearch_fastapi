# services/neural_search_relevant_new.py

import re

from enum import Enum
from typing import List, Optional, Dict, Any, Union

from pydantic import BaseModel

from services.utils import (PAGE_SIZE, remove_stopwords_from_query, K_VALUE, client,
                            group_hits_by_parent, build_sort)


WORD_RE = re.compile(r"\w+", re.UNICODE)

def tokenize(text: str) -> List[str]:
    """
    Very simple tokenizer: lowercase + word characters.
    """
    if not text:
        return []
    return WORD_RE.findall(text.lower())

def split_query_into_fragments(query: str) -> List[str]:
    """
    Split user query into fragments on basic separators (and, &, +, comma, semicolon).
    Fallback to the whole query if nothing useful appears.
    """
    if not query:
        return []

    q = query.strip()

    parts = re.split(r"\b(?:and|&|\+|,|;)\b", q, flags=re.IGNORECASE)
    fragments = []
    for p in parts:
        frag = p.strip()
        if len(frag) >= 3:
            fragments.append(frag)

    return fragments or [q]

def score_chunk_for_fragments(chunk_text: str, fragments: List[str]) -> dict:
    """
    Compute simple lexical overlap-based scores between a chunk and all fragments.

    Returns:
        {
            "coverage": fraction of fragments with any overlap (0..1),
            "avg_score": average overlap score per fragment (0..1),
            "max_score": best overlap score among fragments (0..1),
        }
    """
    chunk_tokens = tokenize(chunk_text)
    if not chunk_tokens or not fragments:
        return {"coverage": 0.0, "avg_score": 0.0, "max_score": 0.0}

    chunk_token_set = set(chunk_tokens)

    frag_scores = []
    for frag in fragments:
        frag_tokens = tokenize(frag)
        if not frag_tokens:
            continue

        # overlap fraction of fragment tokens appearing in the chunk
        overlap = sum(1 for t in frag_tokens if t in chunk_token_set)
        score = overlap / len(frag_tokens)
        frag_scores.append(score)

    if not frag_scores:
        return {"coverage": 0.0, "avg_score": 0.0, "max_score": 0.0}

    coverage = sum(1 for s in frag_scores if s > 0) / len(fragments)
    avg_score = sum(frag_scores) / len(frag_scores)
    max_score = max(frag_scores)

    return {"coverage": coverage, "avg_score": avg_score, "max_score": max_score}

class SortBy(str, Enum):
    score_desc = "score_desc"
    score_asc  = "score_asc"

    ko_created_at_desc  = "ko_created_at_desc"
    ko_created_at_asc   = "ko_created_at_asc"
    ko_updated_at_desc  = "ko_updated_at_desc"
    ko_updated_at_asc   = "ko_updated_at_asc"

    proj_created_at_desc = "proj_created_at_desc"
    proj_created_at_asc  = "proj_created_at_asc"
    proj_updated_at_desc = "proj_updated_at_desc"
    proj_updated_at_asc  = "proj_updated_at_asc"

NUMERIC_SORT_MAP = {
    1: SortBy.score_desc,          2: SortBy.score_asc,
    3: SortBy.ko_created_at_desc,  4: SortBy.ko_created_at_asc,
    5: SortBy.ko_updated_at_desc,  6: SortBy.ko_updated_at_asc,
    7: SortBy.proj_created_at_desc,8: SortBy.proj_created_at_asc,
    9: SortBy.proj_updated_at_desc,10: SortBy.proj_updated_at_asc,
}

def coerce_sort(raw: Union[None, int, str, SortBy]) -> str:
    """
    Accepts Enum | int | str | None and returns a canonical sort key string.
    Falls back to 'score_desc' on unknown values.
    """
    if raw is None:
        return SortBy.score_desc.value
    if isinstance(raw, SortBy):
        return raw.value
    # numerals (e.g., "3" or 3)
    try:
        i = int(raw)
        return NUMERIC_SORT_MAP.get(i, SortBy.score_desc).value
    except (TypeError, ValueError):
        pass
    s = str(raw).strip().lower()
    for member in SortBy:
        if s == member.value:
            return member.value
    return SortBy.score_desc.value

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
    sort_by: Optional[Union[SortBy, int, str]] = SortBy.score_desc


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
        query_part = {"match_all": {}}
    elif use_semantic:
        query_part = {
            "bool": {
                "should": [
                    {
                        "dis_max": {
                            "tie_breaker": 0.2,
                            "queries": [
                                {"neural": {
                                    "content_embedding": {
                                        "query_text": query,
                                        "model_id": model_id,
                                        "k": K_VALUE
                                    }
                                }
                                },
                                {"multi_match": {
                                    "query": remove_stopwords_from_query(query),
                                    "fields": [
                                        "project_name^9", "title^8", "subtitle^7",
                                        "keywords^7", "description^6", "content_chunk^5"
                                    ],
                                    "operator": "and",
                                    "type": "best_fields"
                                }
                                }
                            ]
                        }
                    },
                    {"neural": {
                        "title_embedding": {
                            "query_text": query,
                            "model_id": model_id,
                            "k": min(K_VALUE, 100)
                        }
                    }
                    },
                    {"term": {"project_acronym": {"value": query, "boost": 6.0}}}
                ],
                "minimum_should_match": 1
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

    if not query:
        must_not = []  # include meta docs so ALL KOs can show up
    else:
        must_not = [{"term": {"chunk_index": -1}}]

    raw_sort = (filters or {}).get("sort_by")
    sort_key = coerce_sort(raw_sort)

    # Final OpenSearch query
    search_query = {
        "_source": {
            "includes": [
                "parent_id", "project_name", "project_acronym", "title", "subtitle", "description",
                "keywords", "topics", "themes", "locations", "languages", "category", "subcategories",
                "date_of_completion", "creators", "intended_purposes", "project_id", "project_type",
                "project_url", "@id", "_orig_id", "ko_created_at", "ko_updated_at", "proj_created_at",
                "proj_updated_at",
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
        "query": {
            "bool": {
                "must": query_part,
                "filter": filter_conditions,
                "must_not": must_not
            }
        },
        "collapse": {"field": "parent_id"},
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

    search_query["sort"] = build_sort(sort_key, has_query=bool(query))

    raw_fetch_size = PAGE_SIZE * 40
    search_query["size"] = raw_fetch_size
    search_query["from"] = 0

    response = client.search(index=index_name, body=search_query)

    hits = response["hits"]["hits"]
    grouped = group_hits_by_parent(hits, parents_size=PAGE_SIZE)

    response["grouped"] = grouped

    return response
