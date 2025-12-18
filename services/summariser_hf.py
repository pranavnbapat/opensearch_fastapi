# services/summariser_hf.py

import asyncio
import os
import re

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

def _tokens(s: str) -> set[str]:
    # Simple, fast tokeniser; good enough for gating.
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

    # Score = token overlap count (simple + surprisingly effective)
    return len(q & t)

def build_prompt(query: str, hits: List[Dict[str, Any]]) -> str:
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

    # Keep only hits that have *some* overlap with the query.
    scored = sorted(
        ((h, _hit_relevance_score(query, h)) for h in hits),
        key=lambda x: x[1],
        reverse=True,
    )
    filtered = [h for (h, s) in scored if s > 0][:5]

    # If nothing looks relevant, don't fabricate a topic summary.
    if not filtered:
        return None

    prompt = build_prompt(query, filtered)

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
