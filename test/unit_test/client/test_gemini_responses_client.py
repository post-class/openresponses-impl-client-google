"""Unit tests for GeminiResponsesClient class."""

from __future__ import annotations

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
