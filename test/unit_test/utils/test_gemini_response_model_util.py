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
from test.unit_test.helpers.response_payloads import (
    build_gemini_citation,
    build_gemini_function_call_part,
    build_gemini_logprobs_result,
    build_gemini_text_part,
)


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _assert_valid_response(response: ResponseResource) -> None:
    round_tripped = ResponseResource.model_validate(
        response.model_dump(mode="json", exclude_none=False)
    )
    assert isinstance(round_tripped, ResponseResource)


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

    def test_resolve_text_config_defaults_format_for_verbosity_only(self) -> None:
        result = GeminiResponseModelUtil._resolve_text_config(
            request_dict={"text": {"verbosity": "high"}}
        )

        assert result == {
            "format": {"type": "text"},
            "verbosity": "high",
        }

    def test_parse_response_with_verbosity_only_text_config_defaults_to_text_format(self) -> None:
        payload = build_gemini_response_payload()

        result = GeminiResponseModelUtil.parse_response(
            payload=payload,
            request_payload={"text": {"verbosity": "high"}},
            model="gemini-3-flash-preview",
            default_response_id="fallback_resp",
        )

        assert result.text.format.type == "text"
        assert result.text.verbosity is not None
        assert result.text.verbosity.value == "high"

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


class TestGeminiResponseModelUtilRegression:
    """Regression coverage for response normalization."""

    @pytest.mark.parametrize(
        ("request_payload", "expected_format_type", "expected_verbosity"),
        [
            ({}, "text", None),
            ({"text": {"verbosity": "high"}}, "text", "high"),
            ({"text": {"format": {"type": "text"}}}, "text", None),
            ({"text": {"format": {"type": "json_object"}}}, "json_object", None),
        ],
    )
    def test_parse_response_normalizes_required_text_fields(
        self,
        request_payload: dict[str, Any],
        expected_format_type: str,
        expected_verbosity: str | None,
    ) -> None:
        result = GeminiResponseModelUtil.parse_response(
            payload=build_gemini_response_payload(),
            request_payload=request_payload,
            model="gemini-3-flash-preview",
            default_response_id="fallback_resp",
        )

        _assert_valid_response(result)
        assert result.text.format.type == expected_format_type
        assert _enum_value(result.text.verbosity) == expected_verbosity

    def test_parse_response_preserves_json_schema_text_config_fields(self) -> None:
        result = GeminiResponseModelUtil.parse_response(
            payload=build_gemini_response_payload(),
            request_payload={
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "fruit_response",
                        "description": "Fruit schema",
                        "schema": {
                            "type": "object",
                            "properties": {"name": {"type": "string"}},
                        },
                        "strict": True,
                    },
                    "verbosity": "high",
                }
            },
            model="gemini-3-flash-preview",
            default_response_id="fallback_resp",
        )

        assert isinstance(result, ResponseResource)
        assert result.text.format.type == "json_schema"
        assert result.text.format.name == "fruit_response"
        assert result.text.format.description == "Fruit schema"
        assert result.text.format.strict is True
        assert _enum_value(result.text.verbosity) == "high"

    def test_parse_response_fills_request_default_fields(self) -> None:
        result = GeminiResponseModelUtil.parse_response(
            payload=build_gemini_response_payload(),
            request_payload={},
            model="gemini-3-flash-preview",
            default_response_id="fallback_resp",
        )

        _assert_valid_response(result)
        assert _enum_value(result.tool_choice) == "none"
        assert _enum_value(result.truncation) == "disabled"
        assert result.parallel_tool_calls is False
        assert result.top_p == 1.0
        assert result.presence_penalty == 0.0
        assert result.frequency_penalty == 0.0
        assert result.top_logprobs == 0
        assert result.temperature == 1.0
        assert result.store is False
        assert result.background is False
        assert result.service_tier == "auto"
        assert isinstance(result.metadata, dict)
        assert result.metadata["gemini_model_version"] == "gemini-3-flash-preview-001"

    def test_parse_response_plain_text_candidate_becomes_completed(self) -> None:
        result = GeminiResponseModelUtil.parse_response(
            payload=build_gemini_response_payload(
                parts=[build_gemini_text_part("Hello from Gemini.")]
            ),
            request_payload={},
            model="gemini-3-flash-preview",
            default_response_id="fallback_resp",
        )

        _assert_valid_response(result)
        assert result.status == "completed"
        assert result.error is None
        assert result.incomplete_details is None

    def test_parse_response_with_no_candidates_becomes_failed(self) -> None:
        result = GeminiResponseModelUtil.parse_response(
            payload=build_gemini_response_payload(overrides={"candidates": []}),
            request_payload={},
            model="gemini-3-flash-preview",
            default_response_id="fallback_resp",
        )

        _assert_valid_response(result)
        assert result.status == "failed"
        assert result.error is not None
        assert result.error.code == "no_candidates"

    def test_parse_response_with_function_call_is_completed_and_normalized(self) -> None:
        result = GeminiResponseModelUtil.parse_response(
            payload=build_gemini_response_payload(
                parts=[
                    build_gemini_function_call_part(
                        call_id="call_1",
                        name="lookup_weather",
                        args={"city": "Tokyo", "unit": "celsius"},
                    )
                ]
            ),
            request_payload={},
            model="gemini-3-flash-preview",
            default_response_id="fallback_resp",
        )

        _assert_valid_response(result)
        assert result.status == "completed"
        function_call = result.output[0].root
        assert function_call.type == "function_call"
        assert function_call.arguments == "{\"city\":\"Tokyo\",\"unit\":\"celsius\"}"
        assert function_call.call_id == "call_1"
        assert function_call.name == "lookup_weather"
        assert _enum_value(function_call.status) == "completed"

    def test_parse_response_normalizes_assistant_text_output_item(self) -> None:
        result = GeminiResponseModelUtil.parse_response(
            payload=build_gemini_response_payload(parts=[build_gemini_text_part("Hello world")]),
            request_payload={},
            model="gemini-3-flash-preview",
            default_response_id="fallback_resp",
        )

        _assert_valid_response(result)
        message = result.output[0].root
        assert message.type == "message"
        assert message.content[0].type == "output_text"
        assert message.content[0].text == "Hello world"

    def test_parse_response_normalizes_reasoning_output_item(self) -> None:
        result = GeminiResponseModelUtil.parse_response(
            payload=build_gemini_response_payload(
                parts=[build_gemini_text_part("Thinking step", thought=True)]
            ),
            request_payload={},
            model="gemini-3-flash-preview",
            default_response_id="fallback_resp",
        )

        _assert_valid_response(result)
        reasoning = result.output[0].root
        assert reasoning.type == "reasoning"
        assert reasoning.content[0].type == "reasoning_text"
        assert reasoning.content[0].text == "Thinking step"
        assert reasoning.summary[0].type == "summary_text"
        assert reasoning.summary[0].text == "Thinking step"

    def test_parse_response_normalizes_citations_into_annotations(self) -> None:
        result = GeminiResponseModelUtil.parse_response(
            payload=build_gemini_response_payload(
                parts=[build_gemini_text_part("Hello world")],
                candidate_overrides={
                    "citation_metadata": {
                        "citations": [
                            build_gemini_citation(
                                uri="https://example.com",
                                title="Example",
                                start_index=0,
                                end_index=5,
                            )
                        ]
                    }
                },
            ),
            request_payload={},
            model="gemini-3-flash-preview",
            default_response_id="fallback_resp",
        )

        _assert_valid_response(result)
        annotation = result.output[0].root.content[0].annotations[0].root
        assert annotation.type == "url_citation"
        assert annotation.url == "https://example.com"
        assert annotation.title == "Example"
        assert annotation.start_index == 0
        assert annotation.end_index == 5

    def test_parse_response_normalizes_logprobs(self) -> None:
        result = GeminiResponseModelUtil.parse_response(
            payload=build_gemini_response_payload(
                parts=[build_gemini_text_part("Hello")],
                candidate_overrides={
                    "logprobs_result": build_gemini_logprobs_result(
                        chosen_tokens=[("Hello", -0.1)],
                        top_tokens=[[("Hello", -0.1), ("Hi", -0.4)]],
                    )
                },
            ),
            request_payload={},
            model="gemini-3-flash-preview",
            default_response_id="fallback_resp",
        )

        _assert_valid_response(result)
        logprob = result.output[0].root.content[0].logprobs[0]
        assert logprob.token == "Hello"
        assert logprob.logprob == pytest.approx(-0.1)
        assert logprob.top_logprobs[1].token == "Hi"
        assert logprob.top_logprobs[1].logprob == pytest.approx(-0.4)
