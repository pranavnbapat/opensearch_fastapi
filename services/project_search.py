# services/project_search.py

from pydantic import BaseModel
from typing import Optional, Dict, Any

from services.utils import PAGE_SIZE, client

class ProjectSearchRequest(BaseModel):
    search_term: str
    page: Optional[int] = 1
    dev: Optional[bool] = False


def project_search(index_name: str, query: str, page: int = 1) -> Dict[str, Any]:
    from_offset = (page - 1) * PAGE_SIZE

    search_query = {
        "track_total_hits": True,
        "from": from_offset,
        "size": PAGE_SIZE,
        "sort": [{ "_score": "desc" }],
        "query": {
            "multi_match": {
                "query": query,
                "fields": ["projectName^2", "projectAcronym"]
            }
        }
    }

    response = client.search(index=index_name, body=search_query)
    return response
