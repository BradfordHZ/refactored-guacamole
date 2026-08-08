"""Perplexity Gateway clients.

The Gateway API (https://docs.perplexity.ai/docs/gateway/quickstart) exposes
frontier models from Anthropic, OpenAI, Google, xAI, and Perplexity behind a
single API key, through either an OpenAI-compatible Chat Completions schema
or an Anthropic-compatible Messages schema. This module wires the official
``openai`` and ``anthropic`` SDKs to the Gateway base URLs — the Perplexity
SDK is not used here, since it targets the Agent/Search/Embeddings APIs.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterable
from typing import Any, TypeVar

import anthropic
import openai
from anthropic.types import MessageParam
from openai.types.chat import ChatCompletionMessageParam

from perplexity_gateway.errors import PerplexityConfigError

CHAT_COMPLETIONS_BASE_URL = "https://api.perplexity.ai/router/v1"
# The Anthropic SDK appends "/v1/messages" to base_url, so this omits "/v1".
MESSAGES_BASE_URL = "https://api.perplexity.ai/router"

_T = TypeVar("_T")

_RETRYABLE_STATUS = {429}
_MAX_RETRIES = 3
_DEFAULT_RETRY_AFTER_SECONDS = 1.0


def get_api_key() -> str:
    """Read the Perplexity API key from the environment.

    Never hardcode the key — create one at https://console.perplexity.ai and
    export it as PERPLEXITY_API_KEY in your own shell before running this code.
    """
    api_key = os.environ.get("PERPLEXITY_API_KEY")
    if not api_key:
        raise PerplexityConfigError(
            "PERPLEXITY_API_KEY is not set. Create a key at "
            "https://console.perplexity.ai and export it, e.g.\n"
            "  export PERPLEXITY_API_KEY=your_key_here"
        )
    return api_key


def chat_completions_client(**kwargs: Any) -> openai.OpenAI:
    """OpenAI-compatible client pointed at the Gateway's chat/completions schema."""
    return openai.OpenAI(api_key=get_api_key(), base_url=CHAT_COMPLETIONS_BASE_URL, **kwargs)


def messages_client(**kwargs: Any) -> anthropic.Anthropic:
    """Anthropic-compatible client pointed at the Gateway's messages schema."""
    return anthropic.Anthropic(api_key=get_api_key(), base_url=MESSAGES_BASE_URL, **kwargs)


def _retry_after_seconds(response: Any) -> float:
    header = response.headers.get("retry-after") if response is not None else None
    if header is None:
        return _DEFAULT_RETRY_AFTER_SECONDS
    try:
        return max(float(header), 0.0)
    except ValueError:
        return _DEFAULT_RETRY_AFTER_SECONDS


def _call_with_retry(
    fn: Callable[[], _T],
    retryable_errors: tuple[type[BaseException], ...],
    max_retries: int = _MAX_RETRIES,
) -> _T:
    """Call ``fn``, honoring Retry-After on 429s (rate limit or model overload)."""
    attempt = 0
    while True:
        try:
            return fn()
        except retryable_errors as exc:
            status_code = getattr(exc, "status_code", None)
            if status_code not in _RETRYABLE_STATUS or attempt >= max_retries:
                raise
            delay = _retry_after_seconds(getattr(exc, "response", None))
            time.sleep(delay)
            attempt += 1


def list_models() -> list[str]:
    """Return the Gateway model catalog (also the allowlist for model ids)."""
    client = chat_completions_client()
    page = _call_with_retry(
        lambda: client.models.list(),
        retryable_errors=(openai.RateLimitError,),
    )
    return [model.id for model in page.data]


def chat_completion(
    model: str,
    messages: Iterable[ChatCompletionMessageParam],
    *,
    stream: bool = False,
    **kwargs: Any,
) -> Any:
    """Call the Gateway via the OpenAI-compatible /chat/completions schema.

    ``model`` must be a "creator/model-name" slug from ``list_models()``
    (e.g. "anthropic/claude-sonnet-5") — unlisted models fail with a 400.
    """
    client = chat_completions_client()
    return _call_with_retry(
        lambda: client.chat.completions.create(
            model=model, messages=list(messages), stream=stream, **kwargs
        ),
        retryable_errors=(openai.RateLimitError,),
    )


def create_message(
    model: str,
    messages: Iterable[MessageParam],
    *,
    max_tokens: int,
    stream: bool = False,
    **kwargs: Any,
) -> Any:
    """Call the Gateway via the Anthropic-compatible /messages schema.

    ``model`` must be a "creator/model-name" slug from ``list_models()``
    (e.g. "openai/gpt-5.6-terra") — unlisted models fail with a 400.
    """
    client = messages_client()
    return _call_with_retry(
        lambda: client.messages.create(
            model=model,
            messages=list(messages),
            max_tokens=max_tokens,
            stream=stream,
            **kwargs,
        ),
        retryable_errors=(anthropic.RateLimitError,),
    )
