# services/utils.py

import nltk
import os

from dotenv import load_dotenv
from nltk.corpus import stopwords
from opensearchpy import OpenSearch, RequestsHttpConnection
from sentence_transformers import SentenceTransformer

load_dotenv()

def get_stopwords(lang="english"):
    try:
        return set(stopwords.words(lang))
    except LookupError:
        nltk.download("stopwords")
        return set(stopwords.words(lang))

STOPWORDS = get_stopwords()

K_VALUE = 10

PAGE_SIZE = 10

recomm_model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

model_id = os.getenv("OPENSEARCH_MSMARCO_MODEL_ID", "LciGfZUBVa2ERaFSUEya")  # Fallback to default if not set

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
    timeout=60,
    max_retries=3,
    retry_on_timeout=True
)


# Convert all filters to lowercase to match OpenSearch indexing
def lowercase_list(values):
    return [v.lower() for v in values] if values else []


def remove_stopwords_from_query(query: str) -> str:
    tokens = query.lower().split()
    return ' '.join([word for word in tokens if word not in STOPWORDS])


def normalise_scores(hits):
    valid_scores = [hit["_score"] for hit in hits if isinstance(hit["_score"], (int, float)) and hit["_score"] < 0]

    print(f"Raw negative scores: {valid_scores}")  # Debug: see what you're working with

    if not valid_scores:
        print("No negative scores found. Skipping normalisation.")
        return hits

    max_neg = max(valid_scores)  # closest to zero, e.g. -1
    min_neg = min(valid_scores)  # most negative, e.g. -9549512000
    score_range = max_neg - min_neg or 1

    print(f"Normalising scores from min: {min_neg}, max: {max_neg}, range: {score_range}")

    for hit in hits:
        score = hit["_score"]
        if isinstance(score, (int, float)) and score < 0:
            hit["_score_normalised"] = (score - min_neg) / score_range
        else:
            hit["_score_normalised"] = 0.0  # this can be adjusted if needed

    return hits

