"""Errors raised by the Perplexity Gateway integration."""

from __future__ import annotations


class PerplexityConfigError(RuntimeError):
    """Raised when the gateway client is misconfigured (e.g. missing API key)."""
