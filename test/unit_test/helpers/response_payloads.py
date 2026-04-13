"""Helper builders for Gemini-compatible response payloads used in unit tests."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

_DEFAULT_RESPONSE_ID = "gemini_resp_123"
_DEFAULT_CREATED_AT = "2026-04-13T00:00:00+00:00"


def build_gemini_response_payload(
    *,
    overrides: dict[str, Any] | None = None,
    candidate_overrides: dict[str, Any] | None = None,
    parts: list[dict[str, Any]] | None = None,
    usage_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a GenerateContentResponse-like payload."""
    payload: dict[str, Any] = {
        "response_id": _DEFAULT_RESPONSE_ID,
        "create_time": _DEFAULT_CREATED_AT,
        "model_version": "gemini-3-flash-preview-001",
        "candidates": [
            {
                "index": 0,
                "finish_reason": "STOP",
                "content": {
                    "role": "model",
                    "parts": parts
                    or [
                        {
                            "text": "Hello from Gemini.",
                        }
                    ],
                },
                "citation_metadata": None,
                "logprobs_result": None,
            }
        ],
        "usage_metadata": {
            "prompt_token_count": 12,
            "response_token_count": 8,
            "total_token_count": 20,
            "cached_content_token_count": 0,
            "thoughts_token_count": 0,
        },
    }

    if candidate_overrides:
        payload["candidates"][0].update(deepcopy(candidate_overrides))
    if usage_overrides:
        payload["usage_metadata"].update(deepcopy(usage_overrides))
    if overrides:
        payload.update(deepcopy(overrides))
    return payload


def build_gemini_stream_chunk_payloads() -> list[dict[str, Any]]:
    """Return cumulative stream chunks for a simple text response."""
    return [
        build_gemini_response_payload(
            overrides={"response_id": "stream_resp_1"},
            parts=[{"text": "Hello"}],
            candidate_overrides={"finish_reason": None},
        ),
        build_gemini_response_payload(
            overrides={"response_id": "stream_resp_1"},
            parts=[{"text": "Hello world"}],
            candidate_overrides={"finish_reason": "STOP"},
        ),
    ]
