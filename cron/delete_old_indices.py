# cron/delete_old_indices.py

import os
import requests
import sys

from datetime import datetime
from dotenv import load_dotenv
from opensearchpy import OpenSearch
from requests.auth import HTTPBasicAuth

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

def setup_logging():
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join(log_dir, f"delete_old_indices_{timestamp}.log")

    # Redirect stdout and stderr to log file
    sys.stdout = open(log_path, "w")
    sys.stderr = sys.stdout
    return log_path

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
    # Set up log file at the start of main execution
    log_file = setup_logging()
    try:
        delete_indices()
    except Exception as e:
        import traceback

        print("❌ Exception occurred:")
        traceback.print_exc()
    finally:
        print(f"\nLog saved to: {log_file}")
