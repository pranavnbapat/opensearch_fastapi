import numpy as np

from collections import Counter
from typing import Dict


def analyze_search_results(results: list) -> Dict:
    """Analyze the quality and distribution of search results."""

    if not results:
        return {"message": "No results found."}

    # 1. Extract Scores
    scores = [hit["_score"] for hit in results]
    avg_score = np.mean(scores)
    median_score = np.median(scores)
    min_score = min(scores)
    max_score = max(scores)
    std_dev = np.std(scores)

    # 2. Score Distribution Buckets
    score_buckets = Counter([round(score, 1) for score in scores])

    # 3. Top Field Contribution
    field_contributions = Counter()
    for hit in results:
        for field in ["title", "summary", "content_pages", "keywords"]:
            if field in hit["_source"]:
                field_contributions[field] += 1

    # 4. Check for Duplicate Results (by ID)
    unique_ids = {hit["_id"] for hit in results}
    duplicate_count = len(results) - len(unique_ids)

    # 5. Count Different Topics in Results
    topics_found = Counter()
    for hit in results:
        if "topics" in hit["_source"]:
            topics_found.update(hit["_source"]["topics"])

    # 6. Identify Outliers (Low-Scoring Results)
    threshold = avg_score - 2 * std_dev
    low_quality_results = [hit for hit in results if hit["_score"] < threshold]

    return {
        "score_stats": {
            "avg_score": round(avg_score, 4),
            "median_score": round(median_score, 4),
            "min_score": round(min_score, 4),
            "max_score": round(max_score, 4),
            "std_dev": round(std_dev, 4),
        },
        "score_distribution": dict(score_buckets),
        "field_contributions": dict(field_contributions),
        "duplicates_found": duplicate_count,
        "topics_distribution": dict(topics_found),
        "low_quality_results": len(low_quality_results),
        "message": "Search analysis completed."
    }
