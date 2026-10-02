"""Optional cache and advisory reservation leases; PostgreSQL stays authoritative."""

import hashlib
import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from typing import Any
from uuid import UUID, uuid4

from redis import Redis
from redis.backoff import NoBackoff
from redis.exceptions import (
    ConnectionError,
    InvalidResponse,
    ResponseError,
    TimeoutError,
)
from redis.retry import Retry

from app.config import get_redis_url

CACHE_SCHEMA_VERSION = "v1"
SEARCH_TTL_SECONDS = 300
RESERVATION_TTL_MS = 5_000
MAX_CACHE_BYTES = 512 * 1024
_REDIS_FAILURES = (ConnectionError, TimeoutError, ResponseError, InvalidResponse)
_RELEASE = (
    "if redis.call('get', KEYS[1]) == ARGV[1] then "
    "return redis.call('del', KEYS[1]) else return 0 end"
)
logger = logging.getLogger(__name__)


def search_key(repository_id: UUID, generation: int, query: str, limit: int) -> str:
    # Fixed deployment model identity: bump schema version for model revisions
    # or semantic changes even if a model keeps the same public name.
    inputs = {
        "operation": "semantic_search",
        "q": query,
        "limit": limit,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "embedding_dimension": 384,
    }
    digest = hashlib.sha256(_encode(inputs)).hexdigest()
    return (
        f"repomind:{CACHE_SCHEMA_VERSION}:repo:{repository_id}:"
        f"snapshot:{generation}:search:{digest}"
    )


def _encode(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


class RedisCache:
    def __init__(self, client: Redis) -> None:
        self.client = client

    def read(self, key: str) -> dict[str, Any] | None:
        try:
            raw = self.client.get(key)
        except _REDIS_FAILURES as error:
            logger.debug("Redis read unavailable category=%s", type(error).__name__)
            return None
        if not isinstance(raw, bytes) or len(raw) > MAX_CACHE_BYTES:
            return None
        try:
            envelope = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError, RecursionError):
            return None
        if (
            not isinstance(envelope, dict)
            or set(envelope) != {"schema", "key", "response"}
            or envelope["schema"] != CACHE_SCHEMA_VERSION
            or envelope["key"] != key
            or not isinstance(envelope["response"], dict)
        ):
            return None
        return envelope["response"]

    def write(self, key: str, response: dict[str, Any]) -> None:
        raw = _encode(
            {"schema": CACHE_SCHEMA_VERSION, "key": key, "response": response}
        )
        if len(raw) > MAX_CACHE_BYTES:
            return
        try:
            self.client.set(key, raw, ex=SEARCH_TTL_SECONDS)
        except _REDIS_FAILURES as error:
            logger.debug("Redis write unavailable category=%s", type(error).__name__)

    @contextmanager
    def reservation(self, repository_id: UUID) -> Iterator[bool]:
        key = f"repomind:{CACHE_SCHEMA_VERSION}:repo:{repository_id}:reservation"
        token = uuid4().hex
        try:
            acquired = self.client.set(key, token, nx=True, px=RESERVATION_TTL_MS)
        except _REDIS_FAILURES as error:
            logger.debug("Redis lease unavailable category=%s", type(error).__name__)
            yield True  # Fall back to PostgreSQL, without claiming Redis ownership.
            return
        if not acquired:
            yield False
            return
        try:
            yield True
        finally:
            try:
                self.client.eval(_RELEASE, 1, key, token)
            except _REDIS_FAILURES as error:
                logger.debug(
                    "Redis release unavailable category=%s", type(error).__name__
                )


@lru_cache(maxsize=1)
def _configured_cache(url: str) -> RedisCache:
    return RedisCache(
        Redis.from_url(
            url,
            decode_responses=False,
            socket_connect_timeout=0.2,
            socket_timeout=0.2,
            retry=Retry(NoBackoff(), 0),
            max_connections=20,
            protocol=2,
        )
    )


def get_redis_cache() -> RedisCache | None:
    url = get_redis_url()
    return None if url is None else _configured_cache(url)
