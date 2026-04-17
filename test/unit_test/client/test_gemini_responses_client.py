"""Unit tests for GeminiResponsesClient class."""

from __future__ import annotations

import logging
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# src 配下を import path に追加（pytest 実行時の互換確保）
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from openresponses_impl_core.models.openresponses_models import CreateResponseBody, ResponseResource

from openresponses_impl_client_google.client.gemini_responses_client import GeminiResponsesClient
from test.unit_test.helpers.response_payloads import (
    build_gemini_response_payload,
    build_gemini_stream_chunk_payloads,
)


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
