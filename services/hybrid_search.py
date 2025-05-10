# services/hybrid_search.py

import datetime
import torch

from sentence_transformers import SentenceTransformer, util
from services.utils import (PAGE_SIZE, remove_stopwords_from_query, lowercase_list, client, normalise_scores)

rerank_model = SentenceTransformer("sentence-transformers/msmarco-distilbert-base-tas-b")
LOCAL_MODEL_PATHS = {
    "msmarco": "sentence-transformers/msmarco-distilbert-base-tas-b",
    "mpnetv2": "sentence-transformers/all-mpnet-base-v2",
    "minilml12v2": "sentence-transformers/all-MiniLM-L12-v2"
}

def hybrid_search_local(index_name: str, query: str, filters: dict, page: int, model_key: str):
    from_offset = 0
    rerank_limit = 100  # Number of docs to rerank

    rerank_model_path = LOCAL_MODEL_PATHS.get(model_key, LOCAL_MODEL_PATHS["msmarco"])
    rerank_model = SentenceTransformer(rerank_model_path)

    # Build filter conditions
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

    filtered_query = remove_stopwords_from_query(query)

    bm25_query = {
        "_source": {
            "excludes": [
                "title_embedding", "summary_embedding", "keywords_embedding", "topics_embedding", "content_embedding",
                "project_embedding", "project_acronym_embedding", "content_embedding_input", "topics_embedding_input",
                "keywords_embedding_input", "content_pages", "_orig_id",
            ]
        },
        "size": rerank_limit,
        "from": from_offset,
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": filtered_query,
                        "fields": [
                            "projectAcronym^9",
                            "projectName^9",
                            "title^8",
                            "content_pages^7",
                            "keywords^6",
                            "summary^5"
                        ]
                    }
                },
                "filter": filter_conditions
            }
        }
    }

    response = client.search(index=index_name, body=bm25_query)
    hits = response["hits"]["hits"]
    total = response["hits"]["total"]["value"]

    if not hits:
        return {"total": 0, "results": [], "page": page}

    # Prepare documents for embedding
    doc_texts = []
    for hit in hits:
        src = hit["_source"]
        combined_text = " ".join([
            src.get("title", ""),
            src.get("summary", ""),
            " ".join(src.get("keywords", [])),
            " ".join(src.get("content_pages", [])) if isinstance(src.get("content_pages"), list) else ""
        ])
        doc_texts.append(combined_text)

    query_embedding = rerank_model.encode(query, convert_to_tensor=True)
    doc_embeddings = rerank_model.encode(doc_texts, convert_to_tensor=True)

    scores = util.cos_sim(query_embedding, doc_embeddings)[0]
    reranked_indices = torch.argsort(scores, descending=True)

    start = (page - 1) * PAGE_SIZE
    end = start + PAGE_SIZE
    selected = [hits[i] for i in reranked_indices[start:end]]

    formatted_results = []
    for hit in selected:
        src = hit["_source"]
        try:
            src["dateCreated"] = datetime.datetime.strptime(src.get("dateCreated", "N/A"), "%Y-%m-%d").strftime("%d-%m-%Y")
        except Exception:
            pass
        src["_tags"] = src.get("keywords", [])
        formatted_results.append({
            "_id": hit["_id"],
            "_score": hit["_score"],
            **src
        })

    total_pages = (total + PAGE_SIZE - 1) // PAGE_SIZE

    return {
        "total": total,
        "results": formatted_results,
        "page": page,
        "total_pages": total_pages
    }


def hybrid_search(index_name: str, query: str, filters: dict, page: int, model_id: str):
    from_offset = (page - 1) * PAGE_SIZE
    filtered_query = remove_stopwords_from_query(query)

    # Build filter conditions
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

    search_query = {
        "_source": {
            "excludes": [
                "title_embedding", "summary_embedding", "keywords_embedding", "topics_embedding", "content_embedding",
                "project_embedding", "project_acronym_embedding", "content_embedding_input", "topics_embedding_input",
                "keywords_embedding_input", "content_pages", "_orig_id",
            ]
        },
        "from": from_offset,
        "size": PAGE_SIZE,
        "query": {
            "bool": {
                "filter": filter_conditions,
                "must": {
                    "hybrid": {
                        "queries": [
                            {
                                "multi_match": {
                                    "query": filtered_query,
                                    "fields": [
                                        "projectAcronym^9",
                                        "projectName^9",
                                        "title^8",
                                        "content_pages^7",
                                        "keywords^6",
                                        "summary^5"
                                    ]
                                }
                            },
                            {
                                "neural": {
                                    "content_embedding": {
                                        "query_text": query,
                                        "model_id": model_id,
                                        "k": 100
                                    }
                                }
                            }
                        ]
                    }
                }
            }
        }
    }

    response = client.search(index=index_name, body=search_query)
    hits = response["hits"]["hits"]
    total = response["hits"]["total"]["value"]

    formatted_results = []
    for hit in hits:
        src = hit["_source"]
        try:
            src["dateCreated"] = datetime.datetime.strptime(src.get("dateCreated", "N/A"), "%Y-%m-%d").strftime("%d-%m-%Y")
        except Exception:
            pass
        src["_tags"] = src.get("keywords", [])
        formatted_results.append({
            "_id": hit["_id"],
            "_score": hit["_score"],
            **src
        })

    total_pages = (total + PAGE_SIZE - 1) // PAGE_SIZE

    return {
        "total": total,
        "results": formatted_results,
        "page": page,
        "total_pages": total_pages
    }
