# neural_search_knn

from typing import List, Optional
from pydantic import BaseModel
from services.utils import lowercase_list, PAGE_SIZE
from models.embedding import select_model, generate_vector_neural_search
from services.opensearch_service import client


class KNNSearchRequest(BaseModel):
    search_term: str
    model: str = "msmarco"
    topics: Optional[List[str]] = None
    subtopics: Optional[List[str]] = None
    languages: Optional[List[str]] = None
    fileType: Optional[List[str]] = None
    locations: Optional[List[str]] = None
    page: Optional[int] = 1

MODEL_IDS_FOR_NS = {
    "msmarco": "LciGfZUBVa2ERaFSUEya",
}


def neural_search_knn(query, model_name, filters, page, model_id):
    """Performs a k-NN (semantic search) with optional filters."""

    # Generate vector embedding for the query
    model, index_name = select_model(model_name)
    query_vector = generate_vector_neural_search(model, query)

    # Convert the vector explicitly to JSON-compatible format
    if not isinstance(query_vector, list) or not all(isinstance(v, (float, int)) for v in query_vector):
        raise ValueError(f"Final Validation Failed! Query vector is not a valid list of floats: {query_vector}")

    # print(f"Sending k-NN Query with vector (length {len(query_vector)})")

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

    # k-NN Query (Semantic Search)
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
                "_orig_id"
            ]
        },
        "track_total_hits": True,
        "size": PAGE_SIZE,
        "from": from_offset,
        "query": {
            "bool": {
                "should": [  # Use multiple embeddings for better relevance
                    {
                        "neural": {
                            "content_embedding": {
                                "query_text": query,
                                "model_id": model_id,
                                "k": 20
                            }
                        }
                    },
                    {
                        "neural": {
                            "summary_embedding": {
                                "query_text": query,
                                "model_id": model_id,
                                "k": 20
                            }
                        }
                    },
                    {
                        "neural": {
                            "title_embedding": {
                                "query_text": query,
                                "model_id": model_id,
                                "k": 20
                            }
                        }
                    },
                    {
                        "neural": {
                            "keywords_embedding": {
                                "query_text": query,
                                "model_id": model_id,
                                "k": 20
                            }
                        }
                    },
                    {
                        "neural": {
                            "topics_embedding": {
                                "query_text": query,
                                "model_id": model_id,
                                "k": 20
                            }
                        }
                    }
                ],
                "minimum_should_match": 1,
                "filter": filter_conditions
            }
        }
    }

    response = client.search(index=index_name, body=search_query)
    return response
