import os
from opensearchpy import OpenSearch, RequestsHttpConnection
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Ensure environment variables are present, else raise an error
OPENSEARCH_API = os.getenv("OPENSEARCH_API")
OPENSEARCH_USR = os.getenv("OPENSEARCH_USR")
OPENSEARCH_PWD = os.getenv("OPENSEARCH_PWD")

# Validate critical environment variables
if not all([OPENSEARCH_API, OPENSEARCH_USR, OPENSEARCH_PWD]):
    raise EnvironmentError("Missing one or more OpenSearch environment variables (OPENSEARCH_API, OPENSEARCH_USR, OPENSEARCH_PWD)")

# OpenSearch Client Setup
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
        response = client.search(index=index_name, body=query_body)
        return response
    except Exception as e:
        raise RuntimeError(f"OpenSearch query failed: {e}")
