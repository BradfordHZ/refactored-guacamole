from __future__ import annotations

import pytest

from perplexity_gateway import client, errors


def test_get_api_key_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
    with pytest.raises(errors.PerplexityConfigError):
        client.get_api_key()


def test_get_api_key_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PERPLEXITY_API_KEY", "test-key")
    assert client.get_api_key() == "test-key"


def test_chat_completions_client_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PERPLEXITY_API_KEY", "test-key")
    c = client.chat_completions_client()
    assert str(c.base_url).rstrip("/") == client.CHAT_COMPLETIONS_BASE_URL


def test_messages_client_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PERPLEXITY_API_KEY", "test-key")
    c = client.messages_client()
    assert str(c.base_url).rstrip("/") == client.MESSAGES_BASE_URL


def test_call_with_retry_retries_on_429_then_succeeds() -> None:
    calls = {"count": 0}

    class FakeError(Exception):
        status_code = 429
        response = type("R", (), {"headers": {"retry-after": "0"}})()

    def flaky() -> str:
        calls["count"] += 1
        if calls["count"] < 2:
            raise FakeError()
        return "ok"

    result = client._call_with_retry(flaky, retryable_errors=(FakeError,))
    assert result == "ok"
    assert calls["count"] == 2


def test_call_with_retry_reraises_non_retryable_status() -> None:
    class FakeError(Exception):
        status_code = 400
        response = None

    def always_fails() -> str:
        raise FakeError()

    with pytest.raises(FakeError):
        client._call_with_retry(always_fails, retryable_errors=(FakeError,))
