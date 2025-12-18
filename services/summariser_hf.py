# services/summariser_hf.py

import os
import asyncio
from typing import Any, Dict, List, Optional

import httpx

HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
HF_BASE_URL = os.getenv("HF_BASE_URL", "https://router.huggingface.co/v1").rstrip("/")
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "Qwen/Qwen2.5-3B-Instruct")

SUMMARY_TIMEOUT_S = float(os.getenv("SUMMARY_TIMEOUT_S", "15.0"))
SUMMARY_MAX_DESC_CHARS = int(os.getenv("SUMMARY_MAX_DESC_CHARS", "250"))
SUMMARY_MAX_CONCURRENCY = int(os.getenv("SUMMARY_MAX_CONCURRENCY", "4"))

_sem = asyncio.Semaphore(SUMMARY_MAX_CONCURRENCY)

def _clip(text: Optional[str], limit: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit] + "…"

def build_prompt(query: str, hits: List[Dict[str, Any]]) -> str:
    lines = [
        f"Query: {query}",
        "Write ONE neutral summary of the results in 100–200 words.",
        "Use only information below. Do not invent details.",
        "",
        "Top results:"
    ]
    for i, h in enumerate(hits[:5], start=1):
        lines.append(
            f"{i}) {h.get('title','').strip()} — "
            f"{_clip(h.get('description'), SUMMARY_MAX_DESC_CHARS)}"
        )
    return "\n".join(lines)

async def summarise_top5_hf(query: str, hits: List[Dict[str, Any]]) -> Optional[str]:
    if not hits:
        return None

    if not HF_TOKEN:
        return None

    prompt = build_prompt(query, hits)

    url = f"{HF_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": HF_MODEL_ID,
        "messages": [
            {"role": "system", "content": "You write concise factual summaries."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 220,
        "temperature": 0.2,
    }

    timeout = httpx.Timeout(connect=5.0, read=SUMMARY_TIMEOUT_S, write=10.0, pool=5.0)

    try:
        async with _sem:
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.post(url, headers=headers, json=payload)
                if r.status_code >= 400:
                    import logging
                    logging.getLogger(__name__).error(
                        "HF Router error status=%s body=%s",
                        r.status_code,
                        r.text[:2000],
                    )

                r.raise_for_status()
                data = r.json()
    except Exception as e:
        import logging
        logging.getLogger(__name__).exception("HF summary failed: %s", e)
        return None

    # Common response shapes:
    # - [{"generated_text": "..."}]
    # - {"generated_text": "..."}  (less common)
    if isinstance(data, dict):
        choices = data.get("choices") or []
        if choices and isinstance(choices[0], dict):
            msg = choices[0].get("message") or {}
            return msg.get("content")
    return None
