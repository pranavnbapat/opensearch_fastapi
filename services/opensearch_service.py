# services/opensearch_service.py

import json
import os
from opensearchpy import OpenSearch, RequestsHttpConnection
from dotenv import load_dotenv

load_dotenv()

# Fetch OpenSearch credentials
OPENSEARCH_API = os.getenv("OPENSEARCH_API")
OPENSEARCH_USR = os.getenv("OPENSEARCH_USR")
OPENSEARCH_PWD = os.getenv("OPENSEARCH_PWD")

if not all([OPENSEARCH_API, OPENSEARCH_USR, OPENSEARCH_PWD]):
    raise EnvironmentError("Missing OpenSearch environment variables!")

# OpenSearch Client
client = OpenSearch(
    hosts=[{"host": OPENSEARCH_API, "port": 443}],
    http_auth=(OPENSEARCH_USR, OPENSEARCH_PWD),
    use_ssl=True,
    verify_certs=True,
    http_compress=True,
    connection_class=RequestsHttpConnection,
    max_retries=3,
    retry_on_timeout=True
)

def search_opensearch(index_name: str, query_body: dict):
    try:
        # print(f"\n🚀 Querying OpenSearch at: {index_name}")
        # print(json.dumps(query_body, indent=2))  # Print formatted query

        response = client.search(index=index_name, body=query_body)
        return response
    except Exception as e:
        raise RuntimeError(f"OpenSearch query failed: {e}")

