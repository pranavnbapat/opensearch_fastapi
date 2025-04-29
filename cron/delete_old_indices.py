# cron/delete_old_indices.py

import os
from dotenv import load_dotenv
from opensearchpy import OpenSearch
from requests.auth import HTTPBasicAuth
import requests

load_dotenv()

OPENSEARCH_URL = os.getenv("OPENSEARCH_URL")
AUTH = (os.getenv("OPENSEARCH_USR"), os.getenv("OPENSEARCH_PWD"))

INDEX_PATTERNS = [
    "top_queries-*",
    "security-auditlog-*"
]

if not all(AUTH) or not OPENSEARCH_URL:
    raise ValueError("Missing OpenSearch credentials or URL. Check your .env file!")

client = OpenSearch(
    hosts=[OPENSEARCH_URL],
    http_auth=AUTH,
    use_ssl=True,
    timeout=60,
    max_retries=3,
    retry_on_timeout=True,
    verify_certs=False
)

def delete_indices():
    for pattern in INDEX_PATTERNS:
        url = f"{OPENSEARCH_URL}/{pattern}"
        response = requests.delete(
            url,
            auth=HTTPBasicAuth(AUTH[0], AUTH[1]),
            headers={"Content-Type": "application/json"},
            verify=False
        )

        if response.status_code == 200:
            print(f"Successfully deleted index pattern: {pattern}")
        elif response.status_code == 404:
            print(f"No indices found matching: {pattern}")
        else:
            print(f"Failed to delete {pattern}. Status {response.status_code}: {response.text}")

if __name__ == "__main__":
    delete_indices()
