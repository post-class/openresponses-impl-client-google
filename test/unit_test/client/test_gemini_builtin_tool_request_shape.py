"""Unit tests for OpenAI-style builtin tool request shape support."""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# src 配下を import path に追加（pytest 実行時の互換確保）
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
_REPO_ROOT = os.path.dirname(_ROOT)
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

_CORE_ROOT = os.path.abspath(os.path.join(_REPO_ROOT, "..", "openresponses-impl-core"))
_CORE_SRC = os.path.join(_CORE_ROOT, "src")
if _CORE_SRC not in sys.path:
    sys.path.insert(0, _CORE_SRC)

from openresponses_impl_core.models.openresponses_models import CreateResponseBody

from openresponses_impl_client_google.client.gemini_responses_client import GeminiResponsesClient


def _build_mock_genai_client() -> MagicMock:
    client = MagicMock()
    client.aio = MagicMock()
    client.aio.models = MagicMock()
    return client


class TestGeminiBuiltinToolRequestShape:
    """Tests for generic builtin tool request support."""

    def test_core_generic_tool_preserves_extra_fields(self) -> None:
        payload = CreateResponseBody.model_validate(
            {
                "input": "Hello",
                "tools": [
                    {
                        "type": "web_search",
                        "external_web_access": False,
                    }
                ],
            }
        )

        assert payload.tools is not None
        assert payload.tools[0].root.model_dump(mode="json", exclude_none=True) == {
            "type": "web_search",
            "external_web_access": False,
        }

    @patch("openresponses_impl_client_google.client.gemini_responses_client.genai.Client")
    def test_build_kwargs_with_google_maps_builtin_tool_flat_config(
        self, mock_client_cls: MagicMock
    ) -> None:
        mock_client_cls.return_value = _build_mock_genai_client()
        client = GeminiResponsesClient(model="gemini-3-flash-preview")
        payload = CreateResponseBody.model_validate(
            {
                "input": "Hello",
                "tools": [
                    {
                        "type": "google_maps",
                        "enable_widget": True,
                        "description": "Use Google Maps",
                    }
                ],
            }
        )

        kwargs = client._build_generate_content_kwargs(payload=payload, extra_params=None)

        config = kwargs["config"]
        assert config is not None
        assert config.tools is not None
        assert len(config.tools) == 1
        assert config.tools[0].model_dump(exclude_none=True) == {
            "google_maps": {
                "enable_widget": True,
            }
        }

    @patch("openresponses_impl_client_google.client.gemini_responses_client.genai.Client")
    def test_build_kwargs_with_object_style_builtin_tools(
        self, mock_client_cls: MagicMock
    ) -> None:
        mock_client_cls.return_value = _build_mock_genai_client()
        client = GeminiResponsesClient(model="gemini-3-flash-preview")
        payload = CreateResponseBody.model_validate(
            {
                "input": "Hello",
                "tools": [
                    {"type": "code_execution"},
                    {"type": "url_context"},
                    {
                        "type": "file_search",
                        "file_search_store_names": ["fileSearchStores/STORE_ID"],
                        "top_k": 5,
                    },
                ],
            }
        )

        kwargs = client._build_generate_content_kwargs(payload=payload, extra_params=None)

        config = kwargs["config"]
        assert config is not None
        assert config.tools is not None
        assert [tool.model_dump(exclude_none=True) for tool in config.tools] == [
            {"code_execution": {}},
            {"url_context": {}},
            {
                "file_search": {
                    "file_search_store_names": ["fileSearchStores/STORE_ID"],
                    "top_k": 5,
                }
            },
        ]

    @patch("openresponses_impl_client_google.client.gemini_responses_client.genai.Client")
    def test_build_kwargs_warns_for_unknown_generic_tool_type(
        self, mock_client_cls: MagicMock
    ) -> None:
        mock_client_cls.return_value = _build_mock_genai_client()
        client = GeminiResponsesClient(model="gemini-3-flash-preview")
        payload = CreateResponseBody.model_validate(
            {
                "input": "Hello",
                "tools": [
                    {
                        "type": "not_a_real_tool",
                        "some_setting": "value",
                    }
                ],
            }
        )

        with patch(
            "openresponses_impl_client_google.client.gemini_responses_client.logger.warning"
        ) as mock_warning:
            kwargs = client._build_generate_content_kwargs(payload=payload, extra_params=None)

        assert kwargs["config"] is not None
        assert kwargs["config"].tools is None
        mock_warning.assert_any_call(
            "Gemini client ignores unsupported generic tool type: %s",
            "not_a_real_tool",
        )

    @patch("openresponses_impl_client_google.client.gemini_responses_client.genai.Client")
    def test_build_kwargs_invalid_builtin_tool_config_raises_value_error(
        self, mock_client_cls: MagicMock
    ) -> None:
        mock_client_cls.return_value = _build_mock_genai_client()
        client = GeminiResponsesClient(model="gemini-3-flash-preview")
        payload = CreateResponseBody.model_validate(
            {
                "input": "Hello",
                "tools": [
                    {
                        "type": "google_maps",
                        "enable_widget": "invalid-bool",
                    }
                ],
            }
        )

        with pytest.raises(ValueError, match="Invalid Gemini builtin tool config"):
            client._build_generate_content_kwargs(payload=payload, extra_params=None)
