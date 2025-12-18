# services/summariser.py

import asyncio
import os

from typing import Any, Dict, List, Optional

import httpx

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://opensearch_fastapi_ollama:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

SUMMARY_TIMEOUT_S = float(os.getenv("SUMMARY_TIMEOUT_S", "4.0"))
SUMMARY_MAX_CONCURRENCY = int(os.getenv("SUMMARY_MAX_CONCURRENCY", "1"))
SUMMARY_MAX_DESC_CHARS = int(os.getenv("SUMMARY_MAX_DESC_CHARS", "450"))

_sem = asyncio.Semaphore(SUMMARY_MAX_CONCURRENCY)

def _clip(text: Optional[str], limit: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit] + "…"

def build_prompt(query: str, hits: List[Dict[str, Any]]) -> str:
    # Compact, structured input reduces hallucinations + keeps CPU fast
    lines = [
        f"Query: {query}",
        "",
        "Task: Write ONE neutral summary of the results in 100–200 words.",
        "Rules: Use only information present below; do not invent details; avoid bullet points.",
        "",
        "Top results:"
    ]
    for i, h in enumerate(hits, start=1):
        lines.append(
            f"{i}) {h.get('title','').strip()}\n"
            f"   Topics: {', '.join((h.get('topics') or [])[:2])}; Themes: {', '.join((h.get('themes') or [])[:4])}\n"
            f"   Keywords: {', '.join((h.get('keywords') or [])[:8])}\n"
            f"   Description: {_clip(h.get('description'), SUMMARY_MAX_DESC_CHARS)}"
        )
    return "\n".join(lines)

async def summarise_top5(query: str, hits: List[Dict[str, Any]]) -> Optional[str]:
    if not hits:
        return None

    prompt = build_prompt(query, hits[:5])

    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": "You write concise factual summaries."},
            {"role": "user", "content": prompt},
        ],
        "options": {
            # Keep it short and stable
            "num_predict": 220,
            "temperature": 0.2,
        },
    }

    async with _sem:
        try:
            async with httpx.AsyncClient(timeout=SUMMARY_TIMEOUT_S) as client:
                r = await client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
                r.raise_for_status()
                data = r.json()
                return (data.get("message") or {}).get("content")
        except Exception as e:
                import traceback
                print("OLLAMA SUMMARY FAILED:", repr(e))
                traceback.print_exc()
                return None

