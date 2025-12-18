# services/summariser.py

import asyncio
import os
import re

from typing import Any, Dict, List, Optional

import httpx

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://opensearch_fastapi_ollama:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

SUMMARY_TIMEOUT_S = float(os.getenv("SUMMARY_TIMEOUT_S", "4.0"))
SUMMARY_MAX_CONCURRENCY = int(os.getenv("SUMMARY_MAX_CONCURRENCY", "1"))
SUMMARY_MAX_DESC_CHARS = int(os.getenv("SUMMARY_MAX_DESC_CHARS", "450"))

_sem = asyncio.Semaphore(SUMMARY_MAX_CONCURRENCY)

timeout = httpx.Timeout(connect=1.0, read=SUMMARY_TIMEOUT_S, write=5.0, pool=5.0)
# _client = httpx.AsyncClient(timeout=timeout)
_client: Optional[httpx.AsyncClient] = None
_client_lock = asyncio.Lock()

def _clip(text: Optional[str], limit: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit] + "…"

def _tokens(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (s or "").lower()))

def _hit_relevance_score(query: str, h: Dict[str, Any]) -> int:
    q = _tokens(query)
    text = " ".join([
        h.get("title") or "",
        h.get("subtitle") or "",
        h.get("description") or "",
        " ".join(h.get("keywords") or []),
        " ".join(h.get("topics") or []),
        " ".join(h.get("themes") or []),
    ])
    t = _tokens(text)
    return len(q & t)

def build_prompt(query: str, hits: List[Dict[str, Any]]) -> str:
    # Compact, structured input reduces hallucinations + keeps CPU fast
    lines = [
        f"Query: {query}",
        "",
        "Task: Write ONE neutral summary (150–200 words) about the QUERY TOPIC using only the evidence below.",
        "",
        "Hard rules:",
        "- Do NOT mention the number of results.",
        "- Do NOT mention that some results are irrelevant or unrelated.",
        "- Do NOT describe the set of documents (no meta commentary).",
        "- If the evidence is insufficient to summarise the topic, output exactly: null",
        "",
        "Evidence (snippets):"
    ]
    for i, h in enumerate(hits, start=1):
        lines.append(
            f"{i}) {h.get('title','').strip()}\n"
            f"   Keywords: {', '.join((h.get('keywords') or [])[:8])}\n"
            f"   Description: {_clip(h.get('description'), SUMMARY_MAX_DESC_CHARS)}"
        )
    return "\n".join(lines)

async def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        async with _client_lock:
            if _client is None:
                _client = httpx.AsyncClient(timeout=timeout)
    return _client

async def aclose() -> None:
    """Close the shared AsyncClient (called from FastAPI lifespan shutdown)."""
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None

async def summarise_top5(query: str, hits: List[Dict[str, Any]]) -> Optional[str]:
    if not hits:
        return None

    scored = sorted(
        ((h, _hit_relevance_score(query, h)) for h in hits),
        key=lambda x: x[1],
        reverse=True,
    )

    filtered = [h for (h, s) in scored if s > 0][:5]
    if not filtered:
        return None  # nothing looks relevant; don't fabricate
    prompt = build_prompt(query, filtered)

    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": "You write concise factual summaries."},
            {"role": "user", "content": prompt},
        ],
        "options": {
            "num_predict": 200,
            "temperature": 0.1,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
            "num_ctx": 2048,
            "num_thread": 18,
        },
    }

    async with _sem:
        try:
            client = await _get_client()
            r = await client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
            r.raise_for_status()
            data = r.json()
            text = ((data.get("message") or {}).get("content") or "").strip()
            if text.lower() == "null":
                return None

            # Treat tiny/truncated outputs as failure (common when timing out / overloaded)
            if len(text) < 40:  # tweak threshold if you want
                return None

            return text
        except Exception as e:
            import traceback
            print("OLLAMA SUMMARY FAILED:", repr(e))
            traceback.print_exc()
            return None

