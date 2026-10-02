import json
from unittest.mock import Mock
from uuid import uuid4

import pytest
from redis import Redis
from redis.exceptions import (
    ConnectionError,
    InvalidResponse,
    ResponseError,
    TimeoutError,
)

from app import redis_cache as module
from app.config import get_redis_url
from app.redis_cache import RedisCache, search_key


def test_keys_are_canonical_and_separate_all_inputs() -> None:
    rid = uuid4()
    key = search_key(rid, 3, "  日本語\n", 10)
    assert key == search_key(rid, 3, "  日本語\n", 10)
    assert "日本語" not in key
    assert len(key.rsplit(":", 1)[1]) == 64
    assert (
        len(
            {
                key,
                search_key(uuid4(), 3, "  日本語\n", 10),
                search_key(rid, 4, "  日本語\n", 10),
                search_key(rid, 3, "日本語", 10),
                search_key(rid, 3, "  日本語\n", 1),
            }
        )
        == 5
    )


def test_round_trip_ttl_and_identity() -> None:
    client = Mock(spec=Redis)
    cache = RedisCache(client)
    cache.write("key", {"items": [], "limit": 10})
    args, kwargs = client.set.call_args
    assert kwargs == {"ex": 300}
    assert isinstance(args[1], bytes)
    client.get.return_value = args[1]
    assert cache.read("key") == {"items": [], "limit": 10}
    assert cache.read("different") is None


@pytest.mark.parametrize(
    "raw",
    [
        None,
        b"\xff",
        b"not-json",
        b"null",
        b"[]",
        b"{}",
        b'{"schema":"v0","key":"key","response":{}}',
        b'{"schema":"v1","key":"key","response":[]}',
        b'{"schema":"v1","key":"key","response":{},"extra":1}',
        b"x" * (module.MAX_CACHE_BYTES + 1),
    ],
)
def test_bad_values_miss(raw: bytes | None) -> None:
    client = Mock(spec=Redis)
    client.get.return_value = raw
    assert RedisCache(client).read("key") is None


def test_oversize_write_is_skipped() -> None:
    client = Mock(spec=Redis)
    RedisCache(client).write("key", {"content": "x" * module.MAX_CACHE_BYTES})
    client.set.assert_not_called()


@pytest.mark.parametrize(
    "error", [ConnectionError, TimeoutError, ResponseError, InvalidResponse]
)
def test_redis_errors_fail_open_without_secrets(
    error: type[Exception], caplog: pytest.LogCaptureFixture
) -> None:
    client = Mock(spec=Redis)
    client.get.side_effect = error("SECRET")
    client.set.side_effect = error("SECRET")
    cache = RedisCache(client)
    assert cache.read("key") is None
    cache.write("key", {})
    with cache.reservation(uuid4()) as allowed:
        assert allowed
    client.eval.assert_not_called()
    assert "SECRET" not in caplog.text


def test_programming_errors_propagate() -> None:
    client = Mock(spec=Redis)
    client.get.side_effect = TypeError("defect")
    with pytest.raises(TypeError, match="defect"):
        RedisCache(client).read("key")


@pytest.mark.parametrize("acquired", [True, None])
def test_lease_ownership_and_expiry(acquired: bool | None) -> None:
    client = Mock(spec=Redis)
    client.set.return_value = acquired
    cache = RedisCache(client)
    rid = uuid4()
    with cache.reservation(rid) as allowed:
        assert allowed is bool(acquired)
        client.eval.assert_not_called()
    args, kwargs = client.set.call_args
    assert str(rid) in args[0]
    assert len(args[1]) == 32
    assert kwargs == {"nx": True, "px": 5000}
    if acquired:
        client.eval.assert_called_once_with(module._RELEASE, 1, args[0], args[1])
    else:
        client.eval.assert_not_called()


def test_release_failure_does_not_mask_body_error() -> None:
    client = Mock(spec=Redis)
    client.set.return_value = True
    client.eval.side_effect = TimeoutError("SECRET")
    with pytest.raises(ValueError, match="body"):
        with RedisCache(client).reservation(uuid4()):
            raise ValueError("body")


def test_tokens_are_unique() -> None:
    client = Mock(spec=Redis)
    client.set.return_value = True
    cache = RedisCache(client)
    for _ in range(2):
        with cache.reservation(uuid4()):
            pass
    assert client.set.call_args_list[0].args[1] != client.set.call_args_list[1].args[1]


@pytest.mark.parametrize("value", ["", None])
def test_absent_redis_is_disabled(
    monkeypatch: pytest.MonkeyPatch, value: str | None
) -> None:
    if value is None:
        monkeypatch.delenv("REDIS_URL", raising=False)
    else:
        monkeypatch.setenv("REDIS_URL", value)
    assert module.get_redis_cache() is None


@pytest.mark.parametrize(
    "value",
    [
        "https://localhost",
        "redis:///0",
        "redis://x:bad/0",
        "redis://x/invalid",
        "redis://x/0?q=1",
        "redis://x/0#f",
        "redis://local\nhost/0",
        *[
            f"redis://user:SECRET@localhost:6379/{selector}"
            for selector in ("-1", "abc", "²", "١", "１", "1.0", "+1", " ", "9" * 5000)
        ],
    ],
)
def test_invalid_config_is_sanitized(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("REDIS_URL", value)
    factory = Mock()
    monkeypatch.setattr(module, "_configured_cache", factory)
    with pytest.raises(ValueError, match="REDIS_URL is invalid") as error:
        module.get_redis_cache()
    assert str(error.value) == "REDIS_URL is invalid"
    assert value not in str(error.value)
    assert "SECRET" not in str(error.value)
    factory.assert_not_called()


@pytest.mark.parametrize("path", ["", "/", "/0", "/1", "/15"])
def test_valid_database_selectors_preserve_url(
    monkeypatch: pytest.MonkeyPatch, path: str
) -> None:
    value = f"redis://localhost:6379{path}"
    monkeypatch.setenv("REDIS_URL", value)
    assert get_redis_url() == value


def test_client_is_cached_with_bounded_timeout_and_no_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = Mock(return_value=Mock(spec=Redis))
    monkeypatch.setattr(module.Redis, "from_url", factory)
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    module._configured_cache.cache_clear()
    try:
        assert module.get_redis_cache() is module.get_redis_cache()
        assert factory.call_count == 1
        kwargs = factory.call_args.kwargs
        assert kwargs["socket_timeout"] == kwargs["socket_connect_timeout"] == 0.2
        assert kwargs["retry"].get_retries() == 0
        assert kwargs["decode_responses"] is False
    finally:
        module._configured_cache.cache_clear()


def test_response_validation_rejects_invalid_cached_dtos() -> None:
    from app.search_api import _cached_response

    client = Mock(spec=Redis)
    cache = RedisCache(client)
    for response in (
        {"items": [], "limit": "10"},
        {"items": [], "limit": 1},
        {"items": [], "limit": 10, "extra": True},
    ):
        client.get.return_value = json.dumps(
            {"schema": "v1", "key": "key", "response": response}
        ).encode()
        assert _cached_response(cache, "key", 10) is None


@pytest.mark.parametrize(
    "defect", ["uuid", "start", "end", "nan", "infinity", "duplicate", "count", "extra"]
)
def test_malformed_cached_items_are_misses(defect: str) -> None:
    from app.search_api import _cached_response

    item = {
        "code_unit_id": str(uuid4()),
        "file_id": str(uuid4()),
        "path": "a.py",
        "kind": "function",
        "content": "def a(): pass",
        "language": "python",
        "start_line": 2,
        "end_line": 3,
        "symbol_name": "a",
        "cosine_distance": 0.0,
    }
    if defect == "uuid":
        item["file_id"] = "invalid"
    elif defect == "start":
        item["start_line"] = 0
    elif defect == "end":
        item["end_line"] = 1
    elif defect in {"nan", "infinity"}:
        item["cosine_distance"] = float("nan" if defect == "nan" else "inf")
    elif defect == "extra":
        item["unexpected"] = "not part of DTO"
    items = [item] * (2 if defect in {"duplicate", "count"} else 1)
    limit = 1 if defect == "count" else 10
    client = Mock(spec=Redis)
    client.get.return_value = json.dumps(
        {"schema": "v1", "key": "key", "response": {"items": items, "limit": limit}}
    ).encode()
    assert _cached_response(RedisCache(client), "key", limit) is None
