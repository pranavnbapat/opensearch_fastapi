# neural_search_knn

import datetime

from pydantic import BaseModel
from typing import List, Optional

from collections import defaultdict
from services.utils import (client, lowercase_list, PAGE_SIZE, knn_search_on_field, generate_vector_neural_search,
                            get_model)


class KNNSearchRequest(BaseModel):
    search_term: str
    model: str = "msmarco"
    topics: Optional[List[str]] = None
    subtopics: Optional[List[str]] = None
    languages: Optional[List[str]] = None
    fileType: Optional[List[str]] = None
    project_type: Optional[List[str]] = None
    projectAcronym: Optional[List[str]] = None
    locations: Optional[List[str]] = None
    page: Optional[int] = 1
    dev: Optional[bool] = False

def neural_search_knn(index_name, query, filters, page):
    """Performs a k-NN (semantic search) with optional filters."""

    # Generate vector embedding for the query
    model = get_model("sentence-transformers/msmarco-distilbert-base-tas-b")
    query_vector = generate_vector_neural_search(model, query)

    # Convert the vector explicitly to JSON-compatible format
    if not isinstance(query_vector, list) or not all(isinstance(v, (float, int)) for v in query_vector):
        raise ValueError(f"Final Validation Failed! Query vector is not a valid list of floats: {query_vector}")

    # Pagination offset
    from_offset = (page - 1) * PAGE_SIZE

    filter_conditions = []
    if filters.get("topics"):
        filter_conditions.append({"terms": {"topics.raw": filters["topics"]}})
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

    fields = [
        "title_embedding",
        "summary_embedding",
        "keywords_embedding",
        "content_embedding",
        "project_embedding"
    ]

    all_results = []
    for field in fields:
        all_results.extend(knn_search_on_field(field, query_vector, index_name, filter_conditions, k=20))

    # Merge and rank
    merged = defaultdict(lambda: {"_source": None, "_score": 0.0})
    for hit in all_results:
        doc_id = hit["_id"]
        score = hit["_score"]
        if merged[doc_id]["_source"] is None:
            merged[doc_id]["_source"] = hit["_source"]
        merged[doc_id]["_score"] += score

    sorted_hits = sorted(merged.items(), key=lambda x: x[1]["_score"], reverse=True)
    total_results = len(sorted_hits)
    total_pages = (total_results + PAGE_SIZE - 1) // PAGE_SIZE
    paginated = sorted_hits[from_offset:from_offset + PAGE_SIZE]

    # Format results
    formatted_results = []
    project_counter = {}

    for doc_id, doc_data in paginated:
        source = doc_data["_source"]

        # Format dateCreated
        date_created = source.get("dateCreated", "N/A")
        try:
            formatted_date = datetime.datetime.strptime(date_created, "%Y-%m-%d").strftime("%d-%m-%Y")
        except ValueError:
            formatted_date = date_created
        source["dateCreated"] = formatted_date

        # Copy keywords to _tags
        source["_tags"] = source.get("keywords", [])

        # Flatten result
        flattened = {"_id": doc_id, "_score": doc_data["_score"], **source}
        formatted_results.append(flattened)

        # Track project acronyms
        project_acronym = source.get("projectAcronym", "Unknown Acronym")
        project_counter[project_acronym] = project_counter.get(project_acronym, 0) + 1

    # Top 3 from this page
    top_3_projects = sorted(project_counter.items(), key=lambda x: x[1], reverse=True)[:3]

    # Re-run aggregations for entire result set
    agg_query = {
        "size": 0,
        "query": {
            "bool": {
                "filter": filter_conditions
            }
        },
        "aggs": {
            "top_projects": {
                "terms": {
                    "field": "projectAcronym",
                    "size": 3,
                    "order": {"_count": "desc"}
                }
            }
        }
    }

    aggregation_response = client.search(index=index_name, body=agg_query)
    buckets = aggregation_response.get("aggregations", {}).get("top_projects", {}).get("buckets", [])

    related_projects_2 = [
        {"project_acronym": b["key"], "count": b["doc_count"]}
        for b in buckets
    ]

    return {
        "data": formatted_results,
        "related_projects_from_this_page": [
            {"project_acronym": acronym, "count": count}
            for acronym, count in top_3_projects
        ],
        "related_projects_from_entire_resultset": related_projects_2,
        "pagination": {
            "total_records": total_results,
            "current_page": page,
            "total_pages": total_pages,
            "next_page": page + 1 if page < total_pages else None,
            "prev_page": page - 1 if page > 1 else None
        }
    }
