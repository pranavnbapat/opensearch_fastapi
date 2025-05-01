# services/recommender.py

import re

from collections import Counter
from nltk.corpus import stopwords
from pydantic import BaseModel, Field
from services.utils import client, recomm_model

stop_words = set(stopwords.words('english'))

class RecommenderRequest(BaseModel):
    text: str = Field(default="Education and training of the farmers")
    top_k: int = 3

def extract_keywords(text):
    """
    Extract significant words from a user query by filtering out stopwords and short tokens.
    """
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    keywords = [w for w in words if w not in stop_words and len(w) > 2]
    return set(keywords)

def recommend_similar(text: str, top_k: int = 5, index_name: str = "test_recomm"):
    # Step 1: Convert input text to embedding vector
    vector = recomm_model.encode(text).tolist()

    # Step 2: Extract keywords from input text for explanation
    input_keywords = extract_keywords(text)

    # Step 3: Perform a kNN vector search in OpenSearch
    res = client.search(index=index_name, body={
        "size": top_k,
        "query": {
            "knn": {
                "embedding": {
                    "vector": vector,
                    "k": top_k
                }
            }
        }
    })

    # Step 4: Parse the results and generate explanations
    results = []
    for hit in res['hits']['hits']:
        src = hit["_source"]

        doc_keywords = src.get("keywords", [])
        doc_keywords_lower = set([kw.lower() for kw in doc_keywords])
        overlapping_keywords = sorted(input_keywords & doc_keywords_lower)

        if overlapping_keywords:
            explanation = (
                    "This project is recommended because it shares keywords like: "
                    + ", ".join(overlapping_keywords[:-1])
                    + (" and " if len(overlapping_keywords) > 1 else "")
                    + overlapping_keywords[-1]
                    + " with your input."
            )
        else:
            explanation = "This project is recommended based on semantic similarity."

        results.append({
            "title": src.get("title", ""),
            "summary": src.get("summary", ""),
            "keywords": doc_keywords,
            "file_type": src.get("file_type", ""),
            "project_name": src.get("project_name", ""),
            "project_acronym": src.get("project_acronym", ""),
            "explanation_keywords": overlapping_keywords,
            "explanation_text": explanation
        })

    return results

    # return [
    #     {
    #         "title": hit["_source"].get("title", ""),
    #         "summary": hit["_source"].get("summary", ""),
    #         "keywords": hit["_source"].get("keywords", []),
    #         "file_type": hit["_source"].get("file_type", ""),
    #         "project_name": hit["_source"].get("project_name", ""),
    #         "project_acronym": hit["_source"].get("project_acronym", "")
    #     }
    #     for hit in res["hits"]["hits"]
    # ]