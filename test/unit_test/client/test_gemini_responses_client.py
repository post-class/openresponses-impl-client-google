"""Unit tests for GeminiResponsesClient class."""

from __future__ import annotations

import base64
import logging
import os
import sys
from copy import deepcopy
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import TypeAdapter

# src 配下を import path に追加（pytest 実行時の互換確保）
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from openresponses_impl_core.models.openresponses_models import CreateResponseBody, ResponseResource
from openresponses_impl_core.models.response_event_types import ResponseStreamingEvent

from openresponses_impl_client_google.client.gemini_responses_client import GeminiResponsesClient
from test.unit_test.helpers.response_payloads import (
    build_gemini_function_call_stream_chunk_payloads,
    build_gemini_response_payload,
    build_gemini_reasoning_stream_chunk_payloads,
    build_gemini_stream_chunk_payloads,
    build_gemini_text_part,
)

_STREAM_EVENT_ADAPTER = TypeAdapter(ResponseStreamingEvent)


def _build_mock_genai_client() -> MagicMock:
    client = MagicMock()
    client.aio = MagicMock()
    client.aio.models = MagicMock()
    return client


def _extract_system_instruction_text(config: object) -> str | None:
    system_instruction = getattr(config, "system_instruction", None)
    if system_instruction is None:
        return None

    parts = getattr(system_instruction, "parts", None) or []
    if not parts:
        return None

    first_part = parts[0]
    return getattr(first_part, "text", None)


def _assert_valid_response(response: ResponseResource) -> None:
    round_tripped = ResponseResource.model_validate(response.model_dump(mode="json"))
    assert isinstance(round_tripped, ResponseResource)


def _assert_valid_stream_event(event: Any) -> None:
    _STREAM_EVENT_ADAPTER.validate_python(event.model_dump(mode="json"))


def _event_types(events: list[Any]) -> list[str]:
    return [event.type for event in events]


class _ModelDumpStub:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = deepcopy(payload)

    def model_dump(self, mode: str = "json", exclude_none: bool = True) -> dict[str, Any]:
        del mode, exclude_none
        return deepcopy(self._payload)


class _FakeAsyncStream:
    def __init__(
        self,
        *,
        payloads: list[dict[str, Any]] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._payloads = list(payloads or [])
        self._error = error
        self.closed = False

    def __aiter__(self) -> _FakeAsyncStream:
        return self

    async def __anext__(self) -> object:
        if self._payloads:
            payload = self._payloads.pop(0)
            return MagicMock(
                model_dump=lambda mode="json", exclude_none=True, payload=payload: payload
            )
        if self._error is not None:
            error = self._error
            self._error = None
            raise error
        raise StopAsyncIteration

    async def aclose(self) -> None:
        self.closed = True


class TestGeminiResponsesClientInit:
    """Tests for GeminiResponsesClient initialization."""

    @patch("openresponses_impl_client_google.client.gemini_responses_client.genai.Client")
    def test_init_success_with_api_key(self, mock_client_cls: MagicMock) -> None:
        mock_client_cls.return_value = _build_mock_genai_client()

        client = GeminiResponsesClient(
            model="gemini-3-flash-preview",
            google_api_key="test-key",
        )

        assert client._model == "gemini-3-flash-preview"
        assert client._api_key == "test-key"
        mock_client_cls.assert_called_once_with(api_key="test-key")

    @patch("openresponses_impl_client_google.client.gemini_responses_client.genai.Client")
    def test_init_success_without_api_key(self, mock_client_cls: MagicMock) -> None:
        mock_client_cls.return_value = _build_mock_genai_client()

        client = GeminiResponsesClient(model="gemini-3-flash-preview")

        assert client._api_key is None
        mock_client_cls.assert_called_once_with()

    @patch("openresponses_impl_client_google.client.gemini_responses_client.genai.Client")
    def test_init_missing_model(self, mock_client_cls: MagicMock) -> None:
        mock_client_cls.return_value = _build_mock_genai_client()

        with pytest.raises(ValueError, match="model is required"):
            GeminiResponsesClient(model="")


class TestGeminiResponsesClientBuildRequest:
    """Tests for request construction helpers."""

    @patch("openresponses_impl_client_google.client.gemini_responses_client.genai.Client")
    def test_build_kwargs_merges_instructions_and_json_schema(self, mock_client_cls: MagicMock) -> None:
        mock_client_cls.return_value = _build_mock_genai_client()
        client = GeminiResponsesClient(model="gemini-3-flash-preview")
        payload = CreateResponseBody.model_validate(
            {
                "model": "ignored-model",
                "instructions": "Primary instructions",
                "input": [
                    {
                        "type": "message",
                        "role": "system",
                        "content": [{"type": "input_text", "text": "System message"}],
                    },
                    {
                        "type": "message",
                        "role": "developer",
                        "content": [{"type": "input_text", "text": "Developer message"}],
                    },
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "Hello"}],
                    },
                ],
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "answer",
                        "schema": {"type": "object", "properties": {"ok": {"type": "boolean"}}},
                    }
                },
            }
        )

        kwargs = client._build_generate_content_kwargs(payload=payload, extra_params=None)

        assert kwargs["model"] == "gemini-3-flash-preview"
        assert len(kwargs["contents"]) == 1
        config = kwargs["config"]
        assert config.system_instruction is not None
        assert config.response_mime_type == "application/json"
        assert config.response_json_schema == {
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
        }
        system_instruction_text = _extract_system_instruction_text(config)
        assert system_instruction_text is not None
        assert "Primary instructions" in system_instruction_text
        assert "System message" in system_instruction_text
        assert "Developer message" in system_instruction_text

    @patch("openresponses_impl_client_google.client.gemini_responses_client.genai.Client")
    def test_build_kwargs_reuses_cached_sticky_instruction_for_follow_up_without_old_instructions(
        self, mock_client_cls: MagicMock
    ) -> None:
        mock_client_cls.return_value = _build_mock_genai_client()
        client = GeminiResponsesClient(model="gemini-3-flash-preview")

        initial_payload = CreateResponseBody.model_validate(
            {
                "instructions": "Primary instructions",
                "input": [
                    {
                        "type": "message",
                        "role": "developer",
                        "content": [{"type": "input_text", "text": "Developer message"}],
                    },
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "Hello"}],
                    },
                ],
            }
        )
        first_kwargs = client._build_generate_content_kwargs(payload=initial_payload, extra_params=None)

        client._call_name_by_call_id["call_1"] = "lookup_weather"
        follow_up_payload = CreateResponseBody.model_validate(
            {
                "input": [
                    {
                        "type": "function_call_output",
                        "call_id": "call_1",
                        "output": "sunny",
                    }
                ]
            }
        )
        second_kwargs = client._build_generate_content_kwargs(payload=follow_up_payload, extra_params=None)

        first_system_instruction = _extract_system_instruction_text(first_kwargs["config"])
        second_system_instruction = _extract_system_instruction_text(second_kwargs["config"])
        assert first_system_instruction is not None
        assert second_system_instruction is not None
        assert "Primary instructions" not in second_system_instruction
        assert "Developer message" in second_system_instruction

    @patch("openresponses_impl_client_google.client.gemini_responses_client.genai.Client")
    def test_build_kwargs_rebuilds_system_instruction_with_new_request_instruction_and_cached_sticky_context(
        self, mock_client_cls: MagicMock
    ) -> None:
        mock_client_cls.return_value = _build_mock_genai_client()
        client = GeminiResponsesClient(model="gemini-3-flash-preview")

        initial_payload = CreateResponseBody.model_validate(
            {
                "instructions": "Initial instructions",
                "input": [
                    {
                        "type": "message",
                        "role": "developer",
                        "content": [{"type": "input_text", "text": "Initial developer"}],
                    },
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "Hello"}],
                    },
                ],
            }
        )
        client._build_generate_content_kwargs(payload=initial_payload, extra_params=None)

        replacement_payload = CreateResponseBody.model_validate(
            {
                "instructions": "Replacement instructions",
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "Next turn"}],
                    },
                ],
            }
        )
        replacement_kwargs = client._build_generate_content_kwargs(
            payload=replacement_payload,
            extra_params=None,
        )

        replacement_system_instruction = _extract_system_instruction_text(replacement_kwargs["config"])
        assert replacement_system_instruction is not None
        assert "Replacement instructions" in replacement_system_instruction
        assert "Initial developer" in replacement_system_instruction
        assert "Initial instructions" not in replacement_system_instruction
        assert "Replacement developer" not in replacement_system_instruction

    @patch("openresponses_impl_client_google.client.gemini_responses_client.genai.Client")
    def test_build_kwargs_replaces_cached_sticky_instruction_when_new_system_or_developer_is_provided(
        self, mock_client_cls: MagicMock
    ) -> None:
        mock_client_cls.return_value = _build_mock_genai_client()
        client = GeminiResponsesClient(model="gemini-3-flash-preview")

        initial_payload = CreateResponseBody.model_validate(
            {
                "instructions": "Initial instructions",
                "input": [
                    {
                        "type": "message",
                        "role": "developer",
                        "content": [{"type": "input_text", "text": "Initial developer"}],
                    },
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "Hello"}],
                    },
                ],
            }
        )
        client._build_generate_content_kwargs(payload=initial_payload, extra_params=None)

        replacement_payload = CreateResponseBody.model_validate(
            {
                "input": [
                    {
                        "type": "message",
                        "role": "system",
                        "content": [{"type": "input_text", "text": "Replacement system"}],
                    },
                    {
                        "type": "message",
                        "role": "developer",
                        "content": [{"type": "input_text", "text": "Replacement developer"}],
                    },
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "Next turn"}],
                    },
                ],
            }
        )
        replacement_kwargs = client._build_generate_content_kwargs(
            payload=replacement_payload,
            extra_params=None,
        )

        replacement_system_instruction = _extract_system_instruction_text(replacement_kwargs["config"])
        assert replacement_system_instruction is not None
        assert "Initial instructions" not in replacement_system_instruction
        assert "Initial developer" not in replacement_system_instruction
        assert "Replacement system" in replacement_system_instruction
        assert "Replacement developer" in replacement_system_instruction

    @patch("openresponses_impl_client_google.client.gemini_responses_client.genai.Client")
    def test_build_kwargs_with_function_tool_and_choice(self, mock_client_cls: MagicMock) -> None:
        mock_client_cls.return_value = _build_mock_genai_client()
        client = GeminiResponsesClient(model="gemini-3-flash-preview")
        payload = CreateResponseBody.model_validate(
            {
                "input": "Hello",
                "tools": [
                    {
                        "type": "function",
                        "name": "lookup_weather",
                        "description": "Lookup weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    }
                ],
                "tool_choice": {
                    "type": "function",
                    "name": "lookup_weather",
                },
            }
        )

        kwargs = client._build_generate_content_kwargs(payload=payload, extra_params=None)

        config = kwargs["config"]
        assert config.tools is not None
        assert config.automatic_function_calling is not None
        assert config.automatic_function_calling.disable is True
        assert config.tool_config is not None
        assert config.tool_config.function_calling_config is not None
        assert config.tool_config.function_calling_config.allowed_function_names == [
            "lookup_weather"
        ]
        assert config.tool_config.include_server_side_tool_invocations is None

    @patch("openresponses_impl_client_google.client.gemini_responses_client.genai.Client")
    def test_build_kwargs_with_google_search_tool(self, mock_client_cls: MagicMock) -> None:
        mock_client_cls.return_value = _build_mock_genai_client()
        client = GeminiResponsesClient(model="gemini-3-flash-preview")
        payload = CreateResponseBody.model_validate(
            {
                "input": "Hello",
                "tools": [
                    {
                        "type": "google_search",
                        "description": "Search the web",
                    }
                ],
            }
        )

        kwargs = client._build_generate_content_kwargs(payload=payload, extra_params=None)

        config = kwargs["config"]
        assert config is not None
        assert config.tools is not None
        assert len(config.tools) == 1
        tool = config.tools[0]
        assert tool.google_search is not None
        assert config.tool_config is not None
        assert config.tool_config.include_server_side_tool_invocations is True
        assert config.tool_config.function_calling_config is None

    @patch("openresponses_impl_client_google.client.gemini_responses_client.genai.Client")
    def test_build_kwargs_with_function_and_builtin_tools(self, mock_client_cls: MagicMock) -> None:
        mock_client_cls.return_value = _build_mock_genai_client()
        client = GeminiResponsesClient(model="gemini-3-flash-preview")
        payload = CreateResponseBody.model_validate(
            {
                "input": "Hello",
                "tools": [
                    {
                        "type": "function",
                        "name": "lookup_weather",
                        "description": "Lookup weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    },
                    {
                        "type": "google_search",
                        "description": "Search the web",
                    },
                ],
                "tool_choice": {
                    "type": "function",
                    "name": "lookup_weather",
                },
            }
        )

        kwargs = client._build_generate_content_kwargs(payload=payload, extra_params=None)

        config = kwargs["config"]
        assert config is not None
        assert config.tools is not None
        assert len(config.tools) == 2
        assert config.tool_config is not None
        assert config.tool_config.function_calling_config is not None
        assert config.tool_config.function_calling_config.allowed_function_names == [
            "lookup_weather"
        ]
        assert config.tool_config.include_server_side_tool_invocations is True

    @patch("openresponses_impl_client_google.client.gemini_responses_client.genai.Client")
    def test_build_kwargs_warns_for_previous_response_id(self, mock_client_cls: MagicMock) -> None:
        mock_client_cls.return_value = _build_mock_genai_client()
        client = GeminiResponsesClient(model="gemini-3-flash-preview")
        payload = CreateResponseBody.model_validate(
            {
                "input": "Hello",
                "previous_response_id": "resp_old",
            }
        )

        with patch(
            "openresponses_impl_client_google.client.gemini_responses_client.logger.warning"
        ) as mock_warning:
            kwargs = client._build_generate_content_kwargs(payload=payload, extra_params=None)

        assert kwargs["model"] == "gemini-3-flash-preview"
        assert mock_warning.called

    @patch("openresponses_impl_client_google.client.gemini_responses_client.genai.Client")
    def test_build_kwargs_with_input_file_file_data_uses_inline_bytes(
        self, mock_client_cls: MagicMock
    ) -> None:
        mock_client_cls.return_value = _build_mock_genai_client()
        client = GeminiResponsesClient(model="gemini-3-flash-preview")
        payload = CreateResponseBody.model_validate(
            {
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {
                                "type": "input_file",
                                "filename": "sample.pdf",
                                "file_data": base64.b64encode(b"PDF-DATA").decode("ascii"),
                            }
                        ],
                    }
                ]
            }
        )

        kwargs = client._build_generate_content_kwargs(payload=payload, extra_params=None)

        content = kwargs["contents"][0]
        part = content.parts[0]
        assert part.inline_data is not None
        assert part.inline_data.data == b"PDF-DATA"
        assert part.inline_data.mime_type == "application/pdf"
        assert part.file_data is None

    @patch("openresponses_impl_client_google.client.gemini_responses_client.genai.Client")
    def test_build_kwargs_with_input_file_file_data_unknown_filename_falls_back_to_octet_stream(
        self, mock_client_cls: MagicMock
    ) -> None:
        mock_client_cls.return_value = _build_mock_genai_client()
        client = GeminiResponsesClient(model="gemini-3-flash-preview")
        payload = CreateResponseBody.model_validate(
            {
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {
                                "type": "input_file",
                                "filename": "sample.unknownext",
                                "file_data": base64.b64encode(b"BINARY-DATA").decode("ascii"),
                            }
                        ],
                    }
                ]
            }
        )

        with patch(
            "openresponses_impl_client_google.client.gemini_responses_client.logger.warning"
        ) as mock_warning:
            kwargs = client._build_generate_content_kwargs(payload=payload, extra_params=None)

        content = kwargs["contents"][0]
        part = content.parts[0]
        assert part.inline_data is not None
        assert part.inline_data.data == b"BINARY-DATA"
        assert part.inline_data.mime_type == "application/octet-stream"
        mock_warning.assert_called_once()

    @patch("openresponses_impl_client_google.client.gemini_responses_client.genai.Client")
    def test_build_kwargs_with_input_file_file_url_keeps_uri_shape(
        self, mock_client_cls: MagicMock
    ) -> None:
        mock_client_cls.return_value = _build_mock_genai_client()
        client = GeminiResponsesClient(model="gemini-3-flash-preview")
        payload = CreateResponseBody.model_validate(
            {
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {
                                "type": "input_file",
                                "filename": "sample.pdf",
                                "file_url": "https://example.com/sample.pdf",
                            }
                        ],
                    }
                ]
            }
        )

        kwargs = client._build_generate_content_kwargs(payload=payload, extra_params=None)

        content = kwargs["contents"][0]
        part = content.parts[0]
        assert part.file_data is not None
        assert part.file_data.file_uri == "https://example.com/sample.pdf"
        assert part.file_data.mime_type == "application/pdf"
        assert part.inline_data is None

    @patch("openresponses_impl_client_google.client.gemini_responses_client.genai.Client")
    def test_build_kwargs_with_input_file_prefers_file_url_over_file_data(
        self, mock_client_cls: MagicMock
    ) -> None:
        mock_client_cls.return_value = _build_mock_genai_client()
        client = GeminiResponsesClient(model="gemini-3-flash-preview")
        payload = CreateResponseBody.model_validate(
            {
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {
                                "type": "input_file",
                                "filename": "sample.pdf",
                                "file_url": "https://example.com/sample.pdf",
                                "file_data": "%%%invalid%%%",
                            }
                        ],
                    }
                ]
            }
        )

        kwargs = client._build_generate_content_kwargs(payload=payload, extra_params=None)

        content = kwargs["contents"][0]
        part = content.parts[0]
        assert part.file_data is not None
        assert part.file_data.file_uri == "https://example.com/sample.pdf"
        assert part.inline_data is None

    @patch("openresponses_impl_client_google.client.gemini_responses_client.genai.Client")
    def test_build_kwargs_with_invalid_input_file_file_data_raises_value_error(
        self, mock_client_cls: MagicMock
    ) -> None:
        mock_client_cls.return_value = _build_mock_genai_client()
        client = GeminiResponsesClient(model="gemini-3-flash-preview")
        payload = CreateResponseBody.model_validate(
            {
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {
                                "type": "input_file",
                                "filename": "sample.pdf",
                                "file_data": "%%%invalid%%%",
                            }
                        ],
                    }
                ]
            }
        )

        with pytest.raises(ValueError, match="Invalid input_file\\.file_data payload"):
            client._build_generate_content_kwargs(payload=payload, extra_params=None)

    @patch("openresponses_impl_client_google.client.gemini_responses_client.genai.Client")
    def test_convert_function_call_output_requires_cache_hit(self, mock_client_cls: MagicMock) -> None:
        mock_client_cls.return_value = _build_mock_genai_client()
        client = GeminiResponsesClient(model="gemini-3-flash-preview")
        payload = CreateResponseBody.model_validate(
            {
                "input": [
                    {
                        "type": "function_call_output",
                        "call_id": "call_missing",
                        "output": "result",
                    }
                ]
            }
        )

        with pytest.raises(ValueError, match="Unknown call_id"):
            client._build_generate_content_kwargs(payload=payload, extra_params=None)

    @patch("openresponses_impl_client_google.client.gemini_responses_client.genai.Client")
    def test_convert_function_call_output_uses_cache_hit(self, mock_client_cls: MagicMock) -> None:
        mock_client_cls.return_value = _build_mock_genai_client()
        client = GeminiResponsesClient(model="gemini-3-flash-preview")
        client._call_name_by_call_id["call_1"] = "lookup_weather"
        payload = CreateResponseBody.model_validate(
            {
                "input": [
                    {
                        "type": "function_call_output",
                        "call_id": "call_1",
                        "output": "sunny",
                    }
                ]
            }
        )

        kwargs = client._build_generate_content_kwargs(payload=payload, extra_params=None)

        content = kwargs["contents"][0]
        assert content.parts[0].function_response is not None
        assert content.parts[0].function_response.name == "lookup_weather"

    @patch("openresponses_impl_client_google.client.gemini_responses_client.genai.Client")
    def test_convert_function_call_uses_cached_thought_signature(
        self, mock_client_cls: MagicMock
    ) -> None:
        mock_client_cls.return_value = _build_mock_genai_client()
        client = GeminiResponsesClient(model="gemini-3-flash-preview")
        client._thought_signature_by_call_id["call_1"] = "c2lnbmF0dXJlLTEyMw"
        payload = CreateResponseBody.model_validate(
            {
                "input": [
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "lookup_weather",
                        "arguments": "{\"city\":\"Tokyo\"}",
                    }
                ]
            }
        )

        kwargs = client._build_generate_content_kwargs(payload=payload, extra_params=None)

        content = kwargs["contents"][0]
        assert content.parts[0].function_call is not None
        assert content.parts[0].thought_signature == b"signature-123"

    @patch("openresponses_impl_client_google.client.gemini_responses_client.genai.Client")
    def test_build_kwargs_groups_parallel_function_calls_into_single_model_turn(
        self, mock_client_cls: MagicMock
    ) -> None:
        mock_client_cls.return_value = _build_mock_genai_client()
        client = GeminiResponsesClient(model="gemini-3-flash-preview")
        payload = CreateResponseBody.model_validate(
            {
                "input": [
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "tool_one",
                        "arguments": "{\"value\":1}",
                    },
                    {
                        "type": "function_call",
                        "call_id": "call_2",
                        "name": "tool_two",
                        "arguments": "{\"value\":2}",
                    },
                ]
            }
        )

        kwargs = client._build_generate_content_kwargs(payload=payload, extra_params=None)

        assert len(kwargs["contents"]) == 1
        content = kwargs["contents"][0]
        assert len(content.parts) == 2
        assert content.parts[0].function_call.name == "tool_one"
        assert content.parts[1].function_call.name == "tool_two"

    @patch("openresponses_impl_client_google.client.gemini_responses_client.genai.Client")
    def test_build_kwargs_groups_parallel_function_call_outputs_into_single_user_turn(
        self, mock_client_cls: MagicMock
    ) -> None:
        mock_client_cls.return_value = _build_mock_genai_client()
        client = GeminiResponsesClient(model="gemini-3-flash-preview")
        client._call_name_by_call_id["call_1"] = "tool_one"
        client._call_name_by_call_id["call_2"] = "tool_two"
        payload = CreateResponseBody.model_validate(
            {
                "input": [
                    {
                        "type": "function_call_output",
                        "call_id": "call_1",
                        "output": "first",
                    },
                    {
                        "type": "function_call_output",
                        "call_id": "call_2",
                        "output": "second",
                    },
                ]
            }
        )

        kwargs = client._build_generate_content_kwargs(payload=payload, extra_params=None)

        assert len(kwargs["contents"]) == 1
        content = kwargs["contents"][0]
        assert len(content.parts) == 2
        assert content.parts[0].function_response.name == "tool_one"
        assert content.parts[1].function_response.name == "tool_two"


class TestGeminiResponsesClientCreateResponse:
    """Tests for response creation methods."""

    @pytest.mark.asyncio
    @patch("openresponses_impl_client_google.client.gemini_responses_client.genai.Client")
    async def test_create_response_non_stream(self, mock_client_cls: MagicMock) -> None:
        mock_client = _build_mock_genai_client()
        mock_client.aio.models.generate_content = AsyncMock(
            return_value=MagicMock(
                model_dump=lambda mode="json", exclude_none=True: build_gemini_response_payload()
            )
        )
        mock_client_cls.return_value = mock_client

        client = GeminiResponsesClient(model="gemini-3-flash-preview")
        payload = CreateResponseBody.model_validate({"input": "Hello", "stream": False})

        result = await client.create_response(payload=payload)

        assert isinstance(result, ResponseResource)
        assert result.status == "completed"
        mock_client.aio.models.generate_content.assert_called_once()

    @pytest.mark.asyncio
    @patch("openresponses_impl_client_google.client.gemini_responses_client.genai.Client")
    async def test_create_response_non_stream_emits_debug_request_and_response_logs(
        self,
        mock_client_cls: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        mock_client = _build_mock_genai_client()
        mock_client.aio.models.generate_content = AsyncMock(
            return_value=MagicMock(
                model_dump=lambda mode="json", exclude_none=True: build_gemini_response_payload()
            )
        )
        mock_client_cls.return_value = mock_client

        client = GeminiResponsesClient(
            model="gemini-3-flash-preview",
            google_api_key="test-secret-key",
        )
        payload = CreateResponseBody.model_validate({"input": "Hello", "stream": False})

        with caplog.at_level(
            logging.DEBUG,
            logger="openresponses_impl_client_google.client.gemini_responses_client",
        ):
            await client.create_response(payload=payload)

        assert "Gemini request payload:" in caplog.text
        assert "Gemini response payload:" in caplog.text
        assert '"text": "Hello"' in caplog.text
        assert '"response_id": "gemini_resp_123"' in caplog.text
        assert "test-secret-key" not in caplog.text

    @pytest.mark.asyncio
    @patch("openresponses_impl_client_google.client.gemini_responses_client.genai.Client")
    async def test_create_response_non_stream_does_not_emit_debug_logs_at_info_level(
        self,
        mock_client_cls: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        mock_client = _build_mock_genai_client()
        mock_client.aio.models.generate_content = AsyncMock(
            return_value=MagicMock(
                model_dump=lambda mode="json", exclude_none=True: build_gemini_response_payload()
            )
        )
        mock_client_cls.return_value = mock_client

        client = GeminiResponsesClient(model="gemini-3-flash-preview")
        payload = CreateResponseBody.model_validate({"input": "Hello", "stream": False})

        with caplog.at_level(
            logging.INFO,
            logger="openresponses_impl_client_google.client.gemini_responses_client",
        ):
            await client.create_response(payload=payload)

        assert "Gemini request payload:" not in caplog.text
        assert "Gemini response payload:" not in caplog.text

    @pytest.mark.asyncio
    @patch("openresponses_impl_client_google.client.gemini_responses_client.genai.Client")
    async def test_create_response_non_stream_function_call_updates_cache(
        self, mock_client_cls: MagicMock
    ) -> None:
        function_call_payload = build_gemini_response_payload(
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
        mock_client = _build_mock_genai_client()
        mock_client.aio.models.generate_content = AsyncMock(
            return_value=MagicMock(
                model_dump=lambda mode="json", exclude_none=True: function_call_payload
            )
        )
        mock_client_cls.return_value = mock_client

        client = GeminiResponsesClient(model="gemini-3-flash-preview")
        payload = CreateResponseBody.model_validate({"input": "Hello", "stream": False})

        result = await client.create_response(payload=payload)

        assert result.output[0].root.type == "function_call"
        assert client._call_name_by_call_id["call_1"] == "lookup_weather"
        assert client._thought_signature_by_call_id["call_1"] == "c2lnbmF0dXJlLTEyMw"

    @pytest.mark.asyncio
    @patch("openresponses_impl_client_google.client.gemini_responses_client.genai.Client")
    async def test_create_response_non_stream_function_call_updates_cache_for_bytes_signature(
        self, mock_client_cls: MagicMock
    ) -> None:
        function_call_payload = build_gemini_response_payload(
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
        mock_client = _build_mock_genai_client()
        mock_client.aio.models.generate_content = AsyncMock(
            return_value=MagicMock(
                model_dump=lambda mode="json", exclude_none=True: function_call_payload
            )
        )
        mock_client_cls.return_value = mock_client

        client = GeminiResponsesClient(model="gemini-3-flash-preview")
        payload = CreateResponseBody.model_validate({"input": "Hello", "stream": False})

        result = await client.create_response(payload=payload)

        function_call = result.output[0].root
        assert function_call.type == "function_call"
        assert function_call.extensions == {
            "google": {
                "thought_signature": "c2lnbmF0dXJlLTEyMw",
            }
        }
        assert client._call_name_by_call_id["call_1"] == "lookup_weather"
        assert client._thought_signature_by_call_id["call_1"] == "c2lnbmF0dXJlLTEyMw"

    @pytest.mark.asyncio
    @patch("openresponses_impl_client_google.client.gemini_responses_client.genai.Client")
    async def test_create_response_non_stream_follow_up_uses_cached_native_turn_history(
        self, mock_client_cls: MagicMock
    ) -> None:
        first_response_payload = build_gemini_response_payload(
            parts=[
                {
                    "function_call": {
                        "id": "call_1",
                        "name": "lookup_weather",
                        "args": {"city": "Tokyo"},
                    },
                    "thought_signature": "c2lnbmF0dXJlLTEyMw",
                },
                {
                    "function_call": {
                        "id": "call_2",
                        "name": "lookup_time",
                        "args": {"city": "Tokyo"},
                    },
                },
            ]
        )
        second_response_payload = build_gemini_response_payload(parts=[{"text": "done"}])

        mock_client = _build_mock_genai_client()
        mock_client.aio.models.generate_content = AsyncMock(
            side_effect=[
                MagicMock(
                    model_dump=lambda mode="json", exclude_none=True: first_response_payload
                ),
                MagicMock(
                    model_dump=lambda mode="json", exclude_none=True: second_response_payload
                ),
            ]
        )
        mock_client_cls.return_value = mock_client

        client = GeminiResponsesClient(model="gemini-3-flash-preview")
        first_payload = CreateResponseBody.model_validate(
            {
                "instructions": "Initial instructions",
                "input": [
                    {
                        "type": "message",
                        "role": "developer",
                        "content": [{"type": "input_text", "text": "Developer message"}],
                    },
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "Hello"}],
                    },
                ],
                "stream": False,
            }
        )
        await client.create_response(payload=first_payload)

        second_payload = CreateResponseBody.model_validate(
            {
                "input": [
                    {
                        "type": "function_call_output",
                        "call_id": "call_1",
                        "output": "sunny",
                    },
                    {
                        "type": "function_call_output",
                        "call_id": "call_2",
                        "output": "18:00",
                    },
                ],
                "stream": False,
            }
        )
        await client.create_response(payload=second_payload)

        second_call_kwargs = mock_client.aio.models.generate_content.await_args_list[1].kwargs
        contents = second_call_kwargs["contents"]
        assert len(contents) == 3
        assert contents[0].parts[0].text == "Hello"
        assert len(contents[1].parts) == 2
        assert contents[1].parts[0].function_call.name == "lookup_weather"
        assert contents[1].parts[0].thought_signature == b"signature-123"
        assert contents[1].parts[1].function_call.name == "lookup_time"
        assert len(contents[2].parts) == 2
        assert contents[2].parts[0].function_response.name == "lookup_weather"
        assert contents[2].parts[1].function_response.name == "lookup_time"
        config = second_call_kwargs["config"]
        system_instruction_text = _extract_system_instruction_text(config)
        assert system_instruction_text == "Developer message"

    @pytest.mark.asyncio
    @patch("openresponses_impl_client_google.client.gemini_responses_client.genai.Client")
    async def test_create_response_non_stream_debug_logs_serialize_bytes_payload(
        self,
        mock_client_cls: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        function_call_payload = build_gemini_response_payload(
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
        mock_client = _build_mock_genai_client()
        mock_client.aio.models.generate_content = AsyncMock(
            return_value=MagicMock(
                model_dump=lambda mode="json", exclude_none=True: function_call_payload
            )
        )
        mock_client_cls.return_value = mock_client

        client = GeminiResponsesClient(model="gemini-3-flash-preview")
        payload = CreateResponseBody.model_validate({"input": "Hello", "stream": False})

        with caplog.at_level(
            logging.DEBUG,
            logger="openresponses_impl_client_google.client.gemini_responses_client",
        ):
            await client.create_response(payload=payload)

        assert '"__type__": "bytes"' in caplog.text
        assert '"length": 13' in caplog.text
        assert '"base64": "c2lnbmF0dXJlLTEyMw=="' in caplog.text

    @pytest.mark.asyncio
    @patch("openresponses_impl_client_google.client.gemini_responses_client.genai.Client")
    async def test_create_response_stream(self, mock_client_cls: MagicMock) -> None:
        async def _stream() -> object:
            for chunk_payload in build_gemini_stream_chunk_payloads():
                yield MagicMock(
                    model_dump=lambda mode="json", exclude_none=True, payload=chunk_payload: payload
                )

        mock_client = _build_mock_genai_client()
        mock_client.aio.models.generate_content_stream = AsyncMock(return_value=_stream())
        mock_client_cls.return_value = mock_client

        client = GeminiResponsesClient(model="gemini-3-flash-preview")
        payload = CreateResponseBody.model_validate({"input": "Hello", "stream": True})

        result = await client.create_response(payload=payload)
        events = [event async for event in result]

        assert events[0].type == "response.created"
        assert any(event.type == "response.output_text.delta" for event in events)
        assert events[-1].type == "response.completed"

    @pytest.mark.asyncio
    @patch("openresponses_impl_client_google.client.gemini_responses_client.genai.Client")
    async def test_create_response_stream_emits_chunk_and_aggregate_debug_logs(
        self,
        mock_client_cls: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        async def _stream() -> object:
            for chunk_payload in build_gemini_stream_chunk_payloads():
                yield MagicMock(
                    model_dump=lambda mode="json", exclude_none=True, payload=chunk_payload: payload
                )

        mock_client = _build_mock_genai_client()
        mock_client.aio.models.generate_content_stream = AsyncMock(return_value=_stream())
        mock_client_cls.return_value = mock_client

        client = GeminiResponsesClient(model="gemini-3-flash-preview")
        payload = CreateResponseBody.model_validate({"input": "Hello", "stream": True})

        with caplog.at_level(
            logging.DEBUG,
            logger="openresponses_impl_client_google.client.gemini_responses_client",
        ):
            result = await client.create_response(payload=payload)
            events = [event async for event in result]

        assert "Gemini request payload:" in caplog.text
        assert caplog.text.count("Gemini stream chunk payload:") == 2
        assert "Gemini stream aggregated payload:" in caplog.text
        assert '"text": "Hello world"' in caplog.text
        assert events[0].type == "response.created"
        assert events[-1].type == "response.completed"


class TestGeminiResponsesClientBuildRequestRegression:
    """Regression tests for request-to-Gemini conversion."""

    @pytest.mark.parametrize(
        ("payload_data", "expected_mime_type", "expected_json_schema"),
        [
            ({"input": "Hello"}, "text/plain", None),
            (
                {"input": "Hello", "text": {"format": {"type": "text"}}},
                "text/plain",
                None,
            ),
            (
                {
                    "input": "Hello",
                    "text": {
                        "format": {
                            "type": "json_schema",
                            "name": "answer",
                            "schema": {"type": "object", "properties": {"ok": {"type": "boolean"}}},
                        }
                    },
                },
                "application/json",
                {"type": "object", "properties": {"ok": {"type": "boolean"}}},
            ),
        ],
    )
    @patch("openresponses_impl_client_google.client.gemini_responses_client.genai.Client")
    def test_normalize_text_config_for_valid_create_response_body_cases(
        self,
        mock_client_cls: MagicMock,
        payload_data: dict[str, Any],
        expected_mime_type: str,
        expected_json_schema: dict[str, Any] | None,
    ) -> None:
        mock_client_cls.return_value = _build_mock_genai_client()
        client = GeminiResponsesClient(model="gemini-3-flash-preview")
        payload = CreateResponseBody.model_validate(payload_data)

        text_config = client._normalize_text_config(payload=payload)

        assert text_config["response_mime_type"] == expected_mime_type
        assert text_config["response_json_schema"] == expected_json_schema

    @patch("openresponses_impl_client_google.client.gemini_responses_client.genai.Client")
    def test_normalize_text_config_maps_json_object_to_application_json(
        self, mock_client_cls: MagicMock
    ) -> None:
        mock_client_cls.return_value = _build_mock_genai_client()
        client = GeminiResponsesClient(model="gemini-3-flash-preview")
        payload = CreateResponseBody.model_validate({"input": "Hello"})
        object.__setattr__(
            payload,
            "text",
            _ModelDumpStub({"format": {"type": "json_object"}}),
        )

        text_config = client._normalize_text_config(payload=payload)

        assert text_config["response_mime_type"] == "application/json"
        assert text_config["response_json_schema"] is None

    @pytest.mark.parametrize(
        ("effort", "expected_level", "expected_budget", "expect_warning"),
        [
            ("none", None, 0, False),
            ("low", "LOW", None, False),
            ("medium", "MEDIUM", None, False),
            ("high", "HIGH", None, False),
            ("xhigh", "HIGH", None, True),
        ],
    )
    @patch("openresponses_impl_client_google.client.gemini_responses_client.genai.Client")
    def test_build_thinking_config_maps_reasoning_effort(
        self,
        mock_client_cls: MagicMock,
        effort: str,
        expected_level: str | None,
        expected_budget: int | None,
        expect_warning: bool,
    ) -> None:
        mock_client_cls.return_value = _build_mock_genai_client()
        client = GeminiResponsesClient(model="gemini-3-flash-preview")
        payload = CreateResponseBody.model_validate(
            {
                "input": "Hello",
                "reasoning": {"effort": effort},
            }
        )

        with patch(
            "openresponses_impl_client_google.client.gemini_responses_client.logger.warning"
        ) as mock_warning:
            thinking_config = client._build_thinking_config(payload=payload)

        assert thinking_config is not None
        assert getattr(thinking_config, "thinking_level", None) == expected_level
        assert getattr(thinking_config, "thinking_budget", None) == expected_budget
        if expect_warning:
            mock_warning.assert_called_once_with(
                "Gemini client maps reasoning.effort=xhigh to thinking_level=HIGH"
            )
        else:
            mock_warning.assert_not_called()

    @patch("openresponses_impl_client_google.client.gemini_responses_client.genai.Client")
    def test_build_thinking_config_warns_for_reasoning_summary_without_crashing(
        self, mock_client_cls: MagicMock
    ) -> None:
        mock_client_cls.return_value = _build_mock_genai_client()
        client = GeminiResponsesClient(model="gemini-3-flash-preview")
        payload = CreateResponseBody.model_validate(
            {
                "input": "Hello",
                "reasoning": {"effort": "low", "summary": "auto"},
            }
        )

        with patch(
            "openresponses_impl_client_google.client.gemini_responses_client.logger.warning"
        ) as mock_warning:
            thinking_config = client._build_thinking_config(payload=payload)

        assert thinking_config is not None
        assert thinking_config.thinking_level == "LOW"
        mock_warning.assert_called_once_with(
            "Gemini client ignores unsupported reasoning.summary setting."
        )

    @pytest.mark.parametrize(
        ("tool_choice", "expected_mode", "expected_allowed_function_names"),
        [
            ("none", "NONE", None),
            ("auto", "AUTO", None),
            ("required", "ANY", ["lookup_weather", "lookup_time"]),
        ],
    )
    @patch("openresponses_impl_client_google.client.gemini_responses_client.genai.Client")
    def test_build_tool_config_maps_supported_string_tool_choice_values(
        self,
        mock_client_cls: MagicMock,
        tool_choice: str,
        expected_mode: str,
        expected_allowed_function_names: list[str] | None,
    ) -> None:
        mock_client_cls.return_value = _build_mock_genai_client()
        client = GeminiResponsesClient(model="gemini-3-flash-preview")
        payload = CreateResponseBody.model_validate(
            {
                "input": "Hello",
                "tools": [
                    {"type": "function", "name": "lookup_weather"},
                    {"type": "function", "name": "lookup_time"},
                ],
                "tool_choice": tool_choice,
            }
        )

        _, tool_config = client._build_tools_and_tool_config(payload=payload)

        assert tool_config is not None
        assert tool_config.function_calling_config is not None
        assert tool_config.function_calling_config.mode == expected_mode
        assert tool_config.function_calling_config.allowed_function_names == (
            expected_allowed_function_names
        )

    @patch("openresponses_impl_client_google.client.gemini_responses_client.genai.Client")
    def test_build_tool_config_maps_specific_function_tool_choice(
        self, mock_client_cls: MagicMock
    ) -> None:
        mock_client_cls.return_value = _build_mock_genai_client()
        client = GeminiResponsesClient(model="gemini-3-flash-preview")
        payload = CreateResponseBody.model_validate(
            {
                "input": "Hello",
                "tools": [
                    {"type": "function", "name": "lookup_weather"},
                    {"type": "function", "name": "lookup_time"},
                ],
            }
        )
        object.__setattr__(
            payload,
            "tool_choice",
            _ModelDumpStub({"type": "function", "name": "lookup_time"}),
        )

        _, tool_config = client._build_tools_and_tool_config(payload=payload)

        assert tool_config is not None
        assert tool_config.function_calling_config is not None
        assert tool_config.function_calling_config.mode == "ANY"
        assert tool_config.function_calling_config.allowed_function_names == ["lookup_time"]

    @patch("openresponses_impl_client_google.client.gemini_responses_client.genai.Client")
    def test_build_tool_config_maps_allowed_tools_tool_choice(
        self, mock_client_cls: MagicMock
    ) -> None:
        mock_client_cls.return_value = _build_mock_genai_client()
        client = GeminiResponsesClient(model="gemini-3-flash-preview")
        payload = CreateResponseBody.model_validate(
            {
                "input": "Hello",
                "tools": [
                    {"type": "function", "name": "lookup_weather"},
                    {"type": "function", "name": "lookup_time"},
                ],
            }
        )
        object.__setattr__(
            payload,
            "tool_choice",
            _ModelDumpStub(
                {
                    "type": "allowed_tools",
                    "tools": [
                        {"type": "function", "name": "lookup_weather"},
                        {"type": "function", "name": "lookup_time"},
                    ],
                }
            ),
        )

        _, tool_config = client._build_tools_and_tool_config(payload=payload)

        assert tool_config is not None
        assert tool_config.function_calling_config is not None
        assert tool_config.function_calling_config.mode == "ANY"
        assert tool_config.function_calling_config.allowed_function_names == [
            "lookup_weather",
            "lookup_time",
        ]

    @patch("openresponses_impl_client_google.client.gemini_responses_client.genai.Client")
    def test_build_tool_config_warns_for_unsupported_shape_without_crashing(
        self, mock_client_cls: MagicMock
    ) -> None:
        mock_client_cls.return_value = _build_mock_genai_client()
        client = GeminiResponsesClient(model="gemini-3-flash-preview")
        payload = CreateResponseBody.model_validate(
            {
                "input": "Hello",
                "tools": [
                    {"type": "function", "name": "lookup_weather"},
                ],
            }
        )
        object.__setattr__(
            payload,
            "tool_choice",
            _ModelDumpStub({"type": "not_supported", "name": "ignored"}),
        )

        with patch(
            "openresponses_impl_client_google.client.gemini_responses_client.logger.warning"
        ) as mock_warning:
            _, tool_config = client._build_tools_and_tool_config(payload=payload)

        assert tool_config is None
        mock_warning.assert_called_once_with(
            "Gemini client ignores unsupported tool_choice payload: %s",
            {"type": "not_supported", "name": "ignored"},
        )

    @pytest.mark.parametrize(
        ("field_name", "field_value"),
        [
            ("previous_response_id", "resp_old"),
            ("store", True),
            ("background", True),
            ("parallel_tool_calls", True),
            ("max_tool_calls", 2),
            ("truncation", "auto"),
            ("include", ["reasoning.encrypted_content"]),
            ("safety_identifier", "safe_id"),
            ("prompt_cache_key", "cache_key"),
        ],
    )
    @patch("openresponses_impl_client_google.client.gemini_responses_client.genai.Client")
    def test_build_generate_content_kwargs_warns_for_unsupported_openresponses_fields(
        self,
        mock_client_cls: MagicMock,
        field_name: str,
        field_value: Any,
    ) -> None:
        mock_client_cls.return_value = _build_mock_genai_client()
        client = GeminiResponsesClient(model="gemini-3-flash-preview")
        payload_data = {"input": "Hello", field_name: field_value}
        payload = CreateResponseBody.model_validate(payload_data)

        with patch(
            "openresponses_impl_client_google.client.gemini_responses_client.logger.warning"
        ) as mock_warning:
            kwargs = client._build_generate_content_kwargs(payload=payload, extra_params=None)

        assert kwargs["model"] == "gemini-3-flash-preview"
        mock_warning.assert_any_call(
            "Gemini client ignores unsupported OpenResponses field: %s",
            field_name,
        )

    @patch("openresponses_impl_client_google.client.gemini_responses_client.genai.Client")
    def test_build_kwargs_converts_user_and_assistant_messages_to_native_roles(
        self, mock_client_cls: MagicMock
    ) -> None:
        mock_client_cls.return_value = _build_mock_genai_client()
        client = GeminiResponsesClient(model="gemini-3-flash-preview")
        payload = CreateResponseBody.model_validate(
            {
                "input": [
                    {
                        "type": "message",
                        "role": "system",
                        "content": [{"type": "input_text", "text": "System message"}],
                    },
                    {
                        "type": "message",
                        "role": "developer",
                        "content": [{"type": "input_text", "text": "Developer message"}],
                    },
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "User message"}],
                    },
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Assistant message"}],
                    },
                ]
            }
        )

        kwargs = client._build_generate_content_kwargs(payload=payload, extra_params=None)

        assert len(kwargs["contents"]) == 2
        assert kwargs["contents"][0].role == "user"
        assert kwargs["contents"][0].parts[0].text == "User message"
        assert kwargs["contents"][1].role == "model"
        assert kwargs["contents"][1].parts[0].text == "Assistant message"
        system_instruction_text = _extract_system_instruction_text(kwargs["config"])
        assert system_instruction_text == "System message\n\nDeveloper message"

    @patch("openresponses_impl_client_google.client.gemini_responses_client.genai.Client")
    def test_build_kwargs_with_data_uri_input_file_file_data_raises_value_error(
        self, mock_client_cls: MagicMock
    ) -> None:
        mock_client_cls.return_value = _build_mock_genai_client()
        client = GeminiResponsesClient(model="gemini-3-flash-preview")
        payload = CreateResponseBody.model_validate(
            {
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {
                                "type": "input_file",
                                "filename": "sample.pdf",
                                "file_data": "data:application/pdf;base64,SGVsbG8=",
                            }
                        ],
                    }
                ]
            }
        )

        with pytest.raises(ValueError, match="must be raw base64 data, not a data URI"):
            client._build_generate_content_kwargs(payload=payload, extra_params=None)

    @patch("openresponses_impl_client_google.client.gemini_responses_client.genai.Client")
    def test_build_kwargs_warns_for_unsupported_message_content_type_but_keeps_other_parts(
        self, mock_client_cls: MagicMock
    ) -> None:
        mock_client_cls.return_value = _build_mock_genai_client()
        client = GeminiResponsesClient(model="gemini-3-flash-preview")

        with patch(
            "openresponses_impl_client_google.client.gemini_responses_client.logger.warning"
        ) as mock_warning:
            parts = client._convert_message_content_to_parts(
                content=[
                    {"type": "input_text", "text": "Hello"},
                    {"type": "audio_blob", "url": "https://example.com/audio.wav"},
                ]
            )

        assert len(parts) == 1
        assert parts[0].text == "Hello"
        mock_warning.assert_any_call(
            "Gemini client ignores unsupported message content type: %s",
            "audio_blob",
        )


class TestGeminiResponsesClientResponseRegression:
    """Regression tests for non-stream and stream response conversion."""

    @pytest.mark.asyncio
    @patch("openresponses_impl_client_google.client.gemini_responses_client.genai.Client")
    async def test_create_response_non_stream_with_verbosity_only_text_config_is_valid(
        self, mock_client_cls: MagicMock
    ) -> None:
        mock_client = _build_mock_genai_client()
        mock_client.aio.models.generate_content = AsyncMock(
            return_value=MagicMock(
                model_dump=lambda mode="json", exclude_none=True: build_gemini_response_payload()
            )
        )
        mock_client_cls.return_value = mock_client

        client = GeminiResponsesClient(model="gemini-3-flash-preview")
        payload = CreateResponseBody.model_validate(
            {
                "input": "Hello",
                "stream": False,
                "text": {"verbosity": "high"},
            }
        )

        result = await client.create_response(payload=payload)

        assert isinstance(result, ResponseResource)
        _assert_valid_response(result)
        assert result.text.format.type == "text"
        assert result.text.verbosity is not None
        assert result.text.verbosity.value == "high"

    @pytest.mark.asyncio
    @patch("openresponses_impl_client_google.client.gemini_responses_client.genai.Client")
    async def test_create_response_stream_text_events_follow_expected_sequence_and_validate(
        self, mock_client_cls: MagicMock
    ) -> None:
        mock_client = _build_mock_genai_client()
        stream = _FakeAsyncStream(payloads=build_gemini_stream_chunk_payloads())
        mock_client.aio.models.generate_content_stream = AsyncMock(return_value=stream)
        mock_client_cls.return_value = mock_client

        client = GeminiResponsesClient(model="gemini-3-flash-preview")
        payload = CreateResponseBody.model_validate({"input": "Hello", "stream": True})

        result = await client.create_response(payload=payload)
        events = [event async for event in result]

        for event in events:
            _assert_valid_stream_event(event)

        assert _event_types(events) == [
            "response.created",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.completed",
        ]
        assert stream.closed is True

    @pytest.mark.asyncio
    @patch("openresponses_impl_client_google.client.gemini_responses_client.genai.Client")
    async def test_create_response_stream_reasoning_events_follow_expected_sequence_and_validate(
        self, mock_client_cls: MagicMock
    ) -> None:
        mock_client = _build_mock_genai_client()
        stream = _FakeAsyncStream(payloads=build_gemini_reasoning_stream_chunk_payloads())
        mock_client.aio.models.generate_content_stream = AsyncMock(return_value=stream)
        mock_client_cls.return_value = mock_client

        client = GeminiResponsesClient(model="gemini-3-flash-preview")
        payload = CreateResponseBody.model_validate({"input": "Hello", "stream": True})

        result = await client.create_response(payload=payload)
        events = [event async for event in result]

        for event in events:
            _assert_valid_stream_event(event)

        assert _event_types(events) == [
            "response.created",
            "response.output_item.added",
            "response.reasoning.delta",
            "response.reasoning.delta",
            "response.reasoning.done",
            "response.output_item.done",
            "response.completed",
        ]
        assert stream.closed is True

    @pytest.mark.asyncio
    @patch("openresponses_impl_client_google.client.gemini_responses_client.genai.Client")
    async def test_create_response_stream_function_call_events_follow_expected_sequence_and_validate(
        self, mock_client_cls: MagicMock
    ) -> None:
        mock_client = _build_mock_genai_client()
        stream = _FakeAsyncStream(payloads=build_gemini_function_call_stream_chunk_payloads())
        mock_client.aio.models.generate_content_stream = AsyncMock(return_value=stream)
        mock_client_cls.return_value = mock_client

        client = GeminiResponsesClient(model="gemini-3-flash-preview")
        payload = CreateResponseBody.model_validate({"input": "Hello", "stream": True})

        result = await client.create_response(payload=payload)
        events = [event async for event in result]

        for event in events:
            _assert_valid_stream_event(event)

        assert _event_types(events) == [
            "response.created",
            "response.output_item.added",
            "response.function_call_arguments.done",
            "response.output_item.done",
            "response.completed",
        ]
        assert stream.closed is True

    @pytest.mark.asyncio
    @patch("openresponses_impl_client_google.client.gemini_responses_client.genai.Client")
    async def test_create_response_stream_empty_stream_uses_valid_fallback_response(
        self, mock_client_cls: MagicMock
    ) -> None:
        mock_client = _build_mock_genai_client()
        stream = _FakeAsyncStream(payloads=[])
        mock_client.aio.models.generate_content_stream = AsyncMock(return_value=stream)
        mock_client_cls.return_value = mock_client

        client = GeminiResponsesClient(model="gemini-3-flash-preview")
        payload = CreateResponseBody.model_validate({"input": "Hello", "stream": True})

        result = await client.create_response(payload=payload)
        events = [event async for event in result]

        for event in events:
            _assert_valid_stream_event(event)

        assert _event_types(events) == [
            "response.created",
            "response.failed",
        ]
        assert events[0].response.status == "failed"
        assert events[-1].response.error is not None
        assert events[-1].response.error.code == "no_candidates"
        assert stream.closed is True

    @pytest.mark.asyncio
    @patch("openresponses_impl_client_google.client.gemini_responses_client.genai.Client")
    async def test_create_response_stream_with_verbosity_only_text_config_keeps_text_format(
        self, mock_client_cls: MagicMock
    ) -> None:
        mock_client = _build_mock_genai_client()
        stream = _FakeAsyncStream(payloads=build_gemini_stream_chunk_payloads())
        mock_client.aio.models.generate_content_stream = AsyncMock(return_value=stream)
        mock_client_cls.return_value = mock_client

        client = GeminiResponsesClient(model="gemini-3-flash-preview")
        payload = CreateResponseBody.model_validate(
            {
                "input": "Hello",
                "stream": True,
                "text": {"verbosity": "high"},
            }
        )

        result = await client.create_response(payload=payload)
        events = [event async for event in result]

        for event in events:
            _assert_valid_stream_event(event)

        assert events[0].response.text.format.type == "text"
        assert events[0].response.text.verbosity is not None
        assert events[0].response.text.verbosity.value == "high"
        assert events[-1].type == "response.completed"
        assert stream.closed is True

    @pytest.mark.asyncio
    @patch("openresponses_impl_client_google.client.gemini_responses_client.genai.Client")
    async def test_create_response_stream_returns_error_event_when_stream_iteration_fails(
        self, mock_client_cls: MagicMock
    ) -> None:
        mock_client = _build_mock_genai_client()
        stream = _FakeAsyncStream(
            payloads=[
                build_gemini_response_payload(
                    overrides={"response_id": "stream_resp_1"},
                    parts=[build_gemini_text_part("Hello")],
                    candidate_overrides={"finish_reason": None},
                )
            ],
            error=RuntimeError("stream exploded"),
        )
        mock_client.aio.models.generate_content_stream = AsyncMock(return_value=stream)
        mock_client_cls.return_value = mock_client

        client = GeminiResponsesClient(model="gemini-3-flash-preview")
        payload = CreateResponseBody.model_validate({"input": "Hello", "stream": True})

        result = await client.create_response(payload=payload)
        events = [event async for event in result]

        for event in events:
            _assert_valid_stream_event(event)

        assert _event_types(events)[-1] == "error"
        assert events[-1].error.message == "stream exploded"
        assert stream.closed is True
