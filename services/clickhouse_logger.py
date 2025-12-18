# services/clickhouse_logger.py

import asyncio
import hashlib
import json
import logging
import os
import time
import uuid

from datetime import datetime, timezone
from typing import Any, Dict, Optional, List

import httpx

logger = logging.getLogger(__name__)


def utc_ts_ms() -> str:
    # ClickHouse expects DateTime64(3). Format: "YYYY-MM-DD HH:MM:SS.mmm"
    dt = datetime.now(timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ClickHouseEventLogger:
    """
    Non-blocking event logger:
    - endpoint code enqueues events (very fast)
    - background task batches and flushes to ClickHouse
    """

    def __init__(
        self,
        base_url: str,
        user: str,
        password: str,
        database: str,
        table: str = "search_events",
        flush_interval_s: float = 1.0,
        batch_size: int = 200,
        queue_maxsize: int = 10000,
        timeout_s: float = 3.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.user = user
        self.password = password
        self.database = database
        self.table = table
        self.flush_interval_s = flush_interval_s
        self.batch_size = batch_size
        self.queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(maxsize=queue_maxsize)
        self._task: Optional[asyncio.Task] = None
        self._client: Optional[httpx.AsyncClient] = None
        self.timeout_s = timeout_s
        self._stop_event = asyncio.Event()

    async def start(self) -> None:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout_s)
        if self._task is None:
            self._stop_event.clear()
            self._task = asyncio.create_task(self._run())
            logger.info("ClickHouseEventLogger started")

    async def stop(self) -> None:
        self._stop_event.set()
        if self._task:
            await self._task
            self._task = None
        if self._client:
            await self._client.aclose()
            self._client = None
        logger.info("ClickHouseEventLogger stopped")

    def log_event(self, event: Dict[str, Any]) -> None:
        """
        Non-blocking: if the queue is full, drop the event instead of slowing search.
        """
        try:
            self.queue.put_nowait(event)
        except asyncio.QueueFull:
            # Avoid blowing up your search endpoint under spikes
            logger.warning("ClickHouse event queue full; dropping event")

    async def _flush_batch(self, batch: List[Dict[str, Any]]) -> None:
        if not batch:
            return
        assert self._client is not None

        # ClickHouse JSONEachRow expects one JSON per line
        payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in batch) + "\n"

        # Use query param for INSERT (simple and fast)
        url = (
            f"{self.base_url}/"
            f"?database={self.database}"
            f"&query=INSERT%20INTO%20{self.table}%20FORMAT%20JSONEachRow"
        )

        try:
            r = await self._client.post(
                url,
                content=payload.encode("utf-8"),
                auth=(self.user, self.password),
                headers={"Content-Type": "application/json"},
            )
            r.raise_for_status()
        except Exception as e:
            # Important: do NOT raise (don’t crash the background task)
            logger.error("ClickHouse insert failed (%d events): %s", len(batch), e)

    async def _run(self) -> None:
        """
        Flush loop: batch_size or flush_interval_s, whichever comes first.
        """
        batch: List[Dict[str, Any]] = []
        last_flush = time.monotonic()

        while not self._stop_event.is_set():
            timeout = max(0.0, self.flush_interval_s - (time.monotonic() - last_flush))
            try:
                item = await asyncio.wait_for(self.queue.get(), timeout=timeout)
                batch.append(item)
                # flush if batch size reached
                if len(batch) >= self.batch_size:
                    await self._flush_batch(batch)
                    batch.clear()
                    last_flush = time.monotonic()
            except asyncio.TimeoutError:
                # flush on interval
                if batch:
                    await self._flush_batch(batch)
                    batch.clear()
                last_flush = time.monotonic()

        # final flush on shutdown
        if batch:
            await self._flush_batch(batch)

    def stats(self) -> dict:
        return {
            "task_running": bool(self._task and not self._task.done()),
            "task_done": bool(self._task and self._task.done()),
            "queue_size": self.queue.qsize(),
            "base_url": self.base_url,
            "db": self.database,
            "table": self.table,
        }


def build_search_event(
    *,
    endpoint: str,
    original_query: str,
    final_query: str,
    ip: str,
    user_agent: str,
    accept_language: str,
    referer: str,
    user_id: Optional[int] = None,
    result_orig_ids: Optional[List[str]] = None,
    page: int,
    k: Optional[int],
    model_key: str,
    index_name: str,
    use_semantic: bool,
    translation_allowed: bool,
    detected_lang: str,
    translated: bool,
    filters: Dict[str, Any],
    status_code: int,
    latency_ms: int,
    results_count: int,
    total_records: Optional[int],
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    rid = request_id or str(uuid.uuid4())
    return {
        "ts": utc_ts_ms(),
        "endpoint": endpoint,
        "request_id": rid,
        "ip": ip,
        "user_agent": user_agent or "",
        "accept_language": accept_language or "",
        "referer": referer or "",
        "user_id": int(user_id) if user_id is not None else None,
        "result_orig_ids": result_orig_ids or [],
        "original_query": original_query or "",
        "final_query": final_query or "",
        "query_hash": sha256_hex(final_query or ""),
        "page": int(page),
        "k": int(k) if k is not None else None,
        "model_key": model_key or "",
        "index_name": index_name or "",
        "use_semantic": 1 if use_semantic else 0,
        "translation_allowed": 1 if translation_allowed else 0,
        "detected_lang": detected_lang or "en",
        "translated": 1 if translated else 0,
        "filters_json": json.dumps(filters or {}, ensure_ascii=False),
        "status_code": int(status_code),
        "latency_ms": int(latency_ms),
        "results_count": int(results_count),
        "total_records": int(total_records) if total_records is not None else None,
    }


def make_default_clickhouse_logger() -> ClickHouseEventLogger:
    base_url = os.getenv("CLICKHOUSE_URL", "http://euf_search_clickhouse:8123")
    user = os.getenv("CLICKHOUSE_USER", "eager_beaver")
    password = os.getenv("CLICKHOUSE_PASSWORD", "")
    db = os.getenv("CLICKHOUSE_DB", "euf_search_analytics")
    return ClickHouseEventLogger(
        base_url=base_url,
        user=user,
        password=password,
        database=db,
        flush_interval_s=float(os.getenv("CLICKHOUSE_FLUSH_INTERVAL_S", "1.0")),
        batch_size=int(os.getenv("CLICKHOUSE_BATCH_SIZE", "200")),
        queue_maxsize=int(os.getenv("CLICKHOUSE_QUEUE_MAXSIZE", "10000")),
    )
