"""Opt-in protocol tests: TEST_REDIS_URL must point to a disposable instance."""

import os
from collections.abc import Iterator
from uuid import uuid4

import pytest
from redis import Redis

from app.redis_cache import _RELEASE, RedisCache


@pytest.fixture
def client() -> Iterator[Redis]:
    url = os.getenv("TEST_REDIS_URL")
    if not url:
        pytest.skip("TEST_REDIS_URL disposable instance is not configured")
    value = Redis.from_url(url, socket_timeout=1, socket_connect_timeout=1)
    try:
        assert value.ping()
        yield value
    finally:
        value.close()


def test_real_json_ttl_and_lease_contention(client: Redis) -> None:
    cache = RedisCache(client)
    key = f"repomind:test:{uuid4()}"
    try:
        cache.write(key, {"items": [], "limit": 10})
        assert 0 < client.ttl(key) <= 300
        assert cache.read(key) == {"items": [], "limit": 10}
        rid = uuid4()
        with cache.reservation(rid) as first:
            assert first
            with cache.reservation(rid) as second:
                assert not second
            with cache.reservation(uuid4()) as other:
                assert other
        with cache.reservation(rid) as after:
            assert after
    finally:
        client.delete(key)


def test_expired_owner_cannot_release_reacquired_lease(client: Redis) -> None:
    key = f"repomind:test:{uuid4()}"
    try:
        assert client.set(key, "old", nx=True, px=5000)
        assert 0 < client.pttl(key) <= 5000
        # Force expiry of only this test-owned key, without timing/sleep races.
        client.pexpire(key, 0)
        assert client.set(key, "new", nx=True, px=5000)
        assert client.eval(_RELEASE, 1, key, "old") == 0
        assert client.get(key) == b"new"
        assert client.eval(_RELEASE, 1, key, "new") == 1
        assert client.get(key) is None
    finally:
        client.delete(key)
