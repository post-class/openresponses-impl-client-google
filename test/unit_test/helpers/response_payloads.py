"""Helper builders for Gemini-compatible response payloads used in unit tests."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

_DEFAULT_RESPONSE_ID = "gemini_resp_123"
_DEFAULT_CREATED_AT = "2026-04-13T00:00:00+00:00"


def build_gemini_text_part(
    text: str,
    *,
    thought: bool = False,
    thought_signature: str | bytes | None = None,
) -> dict[str, Any]:
    """Return a Gemini text/thought part."""
    part: dict[str, Any] = {
        "text": text,
    }
    if thought:
        part["thought"] = True
    if thought_signature is not None:
        part["thought_signature"] = thought_signature
    return part


def build_gemini_function_call_part(
    *,
    call_id: str,
    name: str,
    args: dict[str, Any],
    thought_signature: str | bytes | None = None,
) -> dict[str, Any]:
    """Return a Gemini function-call part."""
    part: dict[str, Any] = {
        "function_call": {
            "id": call_id,
            "name": name,
            "args": deepcopy(args),
        }
    }
    if thought_signature is not None:
        part["thought_signature"] = thought_signature
    return part


def build_gemini_citation(
    *,
    uri: str,
    title: str,
    start_index: int,
    end_index: int,
) -> dict[str, Any]:
    """Return a Gemini citation entry."""
    return {
        "uri": uri,
        "title": title,
        "start_index": start_index,
        "end_index": end_index,
    }


def build_gemini_logprobs_result(
    *,
    chosen_tokens: list[tuple[str, float]],
    top_tokens: list[list[tuple[str, float]]] | None = None,
) -> dict[str, Any]:
    """Return a Gemini logprobs_result payload."""
    top_tokens = top_tokens or [[] for _ in chosen_tokens]
    return {
        "chosen_candidates": [
            {
                "token": token,
                "log_probability": log_probability,
            }
            for token, log_probability in chosen_tokens
        ],
        "top_candidates": [
            {
                "candidates": [
                    {
                        "token": token,
                        "log_probability": log_probability,
                    }
                    for token, log_probability in top_group
                ]
            }
            for top_group in top_tokens
        ],
    }


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
            parts=[build_gemini_text_part("Hello")],
            candidate_overrides={"finish_reason": None},
        ),
        build_gemini_response_payload(
            overrides={"response_id": "stream_resp_1"},
            parts=[build_gemini_text_part("Hello world")],
            candidate_overrides={"finish_reason": "STOP"},
        ),
    ]


def build_gemini_reasoning_stream_chunk_payloads() -> list[dict[str, Any]]:
    """Return cumulative stream chunks for a reasoning response."""
    return [
        build_gemini_response_payload(
            overrides={"response_id": "stream_reasoning_1"},
            parts=[build_gemini_text_part("Thinking", thought=True)],
            candidate_overrides={"finish_reason": None},
        ),
        build_gemini_response_payload(
            overrides={"response_id": "stream_reasoning_1"},
            parts=[build_gemini_text_part("Thinking more", thought=True)],
            candidate_overrides={"finish_reason": "STOP"},
        ),
    ]


def build_gemini_function_call_stream_chunk_payloads() -> list[dict[str, Any]]:
    """Return cumulative stream chunks for a function call response."""
    part = build_gemini_function_call_part(
        call_id="call_stream_1",
        name="lookup_weather",
        args={"city": "Tokyo"},
        thought_signature="c2lnbmF0dXJlLTEyMw",
    )
    return [
        build_gemini_response_payload(
            overrides={"response_id": "stream_func_1"},
            parts=[part],
            candidate_overrides={"finish_reason": None},
        ),
        build_gemini_response_payload(
            overrides={"response_id": "stream_func_1"},
            parts=[part],
            candidate_overrides={"finish_reason": "STOP"},
        ),
    ]
