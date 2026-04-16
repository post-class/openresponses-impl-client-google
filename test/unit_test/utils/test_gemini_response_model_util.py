"""Unit tests for GeminiResponseModelUtil class."""

from __future__ import annotations

import os
import sys
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

# src 配下を import path に追加（pytest 実行時の互換確保）
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from openresponses_impl_core.models.openresponses_models import ErrorStreamingEvent, ResponseResource

from openresponses_impl_client_google.utils.gemini_response_model_util import (
    GeminiResponseModelUtil,
)
from test.unit_test.helpers.response_payloads import build_gemini_response_payload


class TestGeminiResponseModelUtil:
    """Tests for GeminiResponseModelUtil."""

    def test_parse_response_with_dict(self) -> None:
        payload = build_gemini_response_payload()
        request_payload = {"instructions": "Be concise"}

        result = GeminiResponseModelUtil.parse_response(
            payload=payload,
            request_payload=request_payload,
            model="gemini-3-flash-preview",
            default_response_id="fallback_resp",
        )

        assert isinstance(result, ResponseResource)
        assert result.id == "gemini_resp_123"
        assert result.model == "gemini-3-flash-preview"
        assert result.output[0].root.type == "message"

    def test_parse_response_with_pydantic_model(self) -> None:
        class CustomModel(BaseModel):
            model_config = ConfigDict(extra="allow")

            response_id: str
            create_time: str
            candidates: list[dict[str, Any]]
            usage_metadata: dict[str, Any]

        payload = CustomModel(**build_gemini_response_payload())

        result = GeminiResponseModelUtil.parse_response(
            payload=payload,
            request_payload={},
            model="gemini-3-flash-preview",
            default_response_id="fallback_resp",
        )

        assert isinstance(result, ResponseResource)
        assert result.id == "gemini_resp_123"

    def test_parse_response_with_function_call_updates_cache(self) -> None:
        payload = build_gemini_response_payload(
            parts=[
                {
                    "function_call": {
                        "id": "call_1",
                        "name": "lookup_weather",
                        "args": {"city": "Tokyo"},
                    },
                    "thought_signature": "c2lnbmF0dXJlLTEyMw",
                }
            ]
        )
        cache: dict[str, str] = {}
        thought_signature_cache: dict[str, str] = {}

        result = GeminiResponseModelUtil.parse_response(
            payload=payload,
            request_payload={"tools": [{"type": "function", "name": "lookup_weather"}]},
            model="gemini-3-flash-preview",
            default_response_id="fallback_resp",
            call_name_by_call_id=cache,
            thought_signature_by_call_id=thought_signature_cache,
        )

        assert result.output[0].root.type == "function_call"
        assert cache["call_1"] == "lookup_weather"
        assert thought_signature_cache["call_1"] == "c2lnbmF0dXJlLTEyMw"

    def test_parse_response_with_function_call_bytes_signature_updates_extensions_and_cache(self) -> None:
        payload = build_gemini_response_payload(
            parts=[
                {
                    "function_call": {
                        "id": "call_1",
                        "name": "lookup_weather",
                        "args": {"city": "Tokyo"},
                    },
                    "thought_signature": b"signature-123",
                }
            ]
        )
        cache: dict[str, str] = {}
        thought_signature_cache: dict[str, str] = {}

        result = GeminiResponseModelUtil.parse_response(
            payload=payload,
            request_payload={"tools": [{"type": "function", "name": "lookup_weather"}]},
            model="gemini-3-flash-preview",
            default_response_id="fallback_resp",
            call_name_by_call_id=cache,
            thought_signature_by_call_id=thought_signature_cache,
        )

        function_call = result.output[0].root
        assert function_call.type == "function_call"
        assert function_call.extensions == {
            "google": {
                "thought_signature": "c2lnbmF0dXJlLTEyMw",
            }
        }
        assert cache["call_1"] == "lookup_weather"
        assert thought_signature_cache["call_1"] == "c2lnbmF0dXJlLTEyMw"

    def test_parse_response_with_json_schema_text_config(self) -> None:
        payload = build_gemini_response_payload()

        result = GeminiResponseModelUtil.parse_response(
            payload=payload,
            request_payload={
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "fruit_response",
                        "schema": {
                            "type": "object",
                            "properties": {"name": {"type": "string"}},
                        },
                    }
                }
            },
            model="gemini-3-flash-preview",
            default_response_id="fallback_resp",
        )

        assert result.text.format.type == "json_schema"
        assert result.text.format.name == "fruit_response"
        assert result.text.format.description is None
        assert result.text.format.schema_ is None
        assert result.text.format.strict is False

    def test_parse_response_with_reasoning_config_defaults_summary_to_none(self) -> None:
        payload = build_gemini_response_payload()

        result = GeminiResponseModelUtil.parse_response(
            payload=payload,
            request_payload={"reasoning": {"effort": "low"}},
            model="gemini-3-flash-preview",
            default_response_id="fallback_resp",
        )

        assert result.reasoning is not None
        assert result.reasoning.effort is not None
        assert result.reasoning.effort.value == "low"
        assert result.reasoning.summary is None

    def test_parse_response_with_max_tokens_becomes_incomplete(self) -> None:
        payload = build_gemini_response_payload(candidate_overrides={"finish_reason": "MAX_TOKENS"})

        result = GeminiResponseModelUtil.parse_response(
            payload=payload,
            request_payload={},
            model="gemini-3-flash-preview",
            default_response_id="fallback_resp",
        )

        assert result.status == "incomplete"
        assert result.incomplete_details is not None
        assert result.incomplete_details.reason == "max_tokens"

    def test_parse_response_with_prompt_block_becomes_failed(self) -> None:
        payload = build_gemini_response_payload(
            overrides={
                "prompt_feedback": {
                    "block_reason": "SAFETY",
                    "block_reason_message": "Blocked by safety system",
                },
                "candidates": [],
            }
        )

        result = GeminiResponseModelUtil.parse_response(
            payload=payload,
            request_payload={},
            model="gemini-3-flash-preview",
            default_response_id="fallback_resp",
        )

        assert result.status == "failed"
        assert result.error is not None
        assert result.error.code == "safety"

    def test_normalize_payload_with_invalid_type(self) -> None:
        with pytest.raises(ValueError, match="payload must be a dict or model"):
            GeminiResponseModelUtil._normalize_payload(payload="invalid")

    def test_parse_response_with_invalid_data(self) -> None:
        result = GeminiResponseModelUtil.parse_response(
            payload={"invalid": "payload"},
            request_payload={},
            model="gemini-3-flash-preview",
            default_response_id="fallback_resp",
        )

        assert result.status == "failed"
        assert result.error is not None
        assert result.error.code == "no_candidates"

    def test_build_error_event(self) -> None:
        event = GeminiResponseModelUtil._build_error_event(
            payload={"sequence_number": 7},
            message="bad chunk",
        )

        assert isinstance(event, ErrorStreamingEvent)
        assert event.sequence_number == 7
        assert event.error.message == "bad chunk"
