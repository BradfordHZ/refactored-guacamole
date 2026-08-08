"""Minimal live smoke test for the Perplexity Gateway integration.

Requires PERPLEXITY_API_KEY to be set in the environment. Prints only
the HTTP status / response shape — never the key or full response body.

Usage:
    PERPLEXITY_API_KEY=... python scripts/smoke_test.py
"""

from __future__ import annotations

import sys

import anthropic
import openai

from perplexity_gateway import chat_completion, create_message, get_api_key, list_models
from perplexity_gateway.errors import PerplexityConfigError


def main() -> int:
    try:
        get_api_key()
    except PerplexityConfigError as exc:
        print(f"CONFIG ERROR: {exc}")
        return 1

    try:
        models = list_models()
        print(f"GET /router/v1/models -> 200, {len(models)} models")
    except openai.APIStatusError as exc:
        print(f"GET /router/v1/models -> {exc.status_code}")
        return 1

    try:
        response = chat_completion(
            model="anthropic/claude-sonnet-5",
            messages=[{"role": "user", "content": "Reply with the single word: pong"}],
            max_tokens=16,
        )
        print(
            "POST /router/v1/chat/completions -> 200, "
            f"choices={len(response.choices)}, "
            f"usage.prompt_tokens={response.usage.prompt_tokens if response.usage else None}"
        )
    except openai.APIStatusError as exc:
        print(f"POST /router/v1/chat/completions -> {exc.status_code}")
        return 1

    try:
        response = create_message(
            model="anthropic/claude-sonnet-5",
            messages=[{"role": "user", "content": "Reply with the single word: pong"}],
            max_tokens=16,
        )
        print(
            "POST /router/v1/messages -> 200, "
            f"content_blocks={len(response.content)}, "
            f"usage.input_tokens={response.usage.input_tokens if response.usage else None}"
        )
    except anthropic.APIStatusError as exc:
        print(f"POST /router/v1/messages -> {exc.status_code}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
