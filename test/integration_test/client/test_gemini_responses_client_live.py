"""Live integration tests for GeminiResponsesClient using the real Gemini API."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
from google import genai

# src 配下を import path に追加（pytest 実行時の互換確保）
_REPO_ROOT = str(Path(__file__).resolve().parents[3])
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from openresponses_impl_core.models.openresponses_models import CreateResponseBody, FunctionCall

from openresponses_impl_client_google.client.gemini_responses_client import GeminiResponsesClient

_LIVE_MODEL = "gemini-3-flash-preview"
_REPO_ROOT_PATH = Path(_REPO_ROOT)


def _load_env_file() -> None:
    env_path = _REPO_ROOT_PATH / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _require_api_key() -> str:
    _load_env_file()
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise AssertionError(
            "GOOGLE_API_KEY or GEMINI_API_KEY is required for live integration tests. "
            "Set it in the shell environment or in the repo-root .env file."
        )
    return api_key


_API_KEY = _require_api_key()


def _validate_live_api_access() -> None:
    try:
        client = genai.Client(api_key=_API_KEY)
        client.models.get(model=_LIVE_MODEL)
    except Exception as exc:
        raise AssertionError(
            "Live Gemini integration tests require a valid GOOGLE_API_KEY or GEMINI_API_KEY. "
            "The configured key was found but Gemini API access validation failed."
        ) from exc


_validate_live_api_access()


def _create_client() -> GeminiResponsesClient:
    return GeminiResponsesClient(
        model=_LIVE_MODEL,
        google_api_key=_API_KEY,
    )


def _extract_assistant_text(response_text: str) -> str:
    return response_text.strip()


def _extract_first_assistant_message_text(response: object) -> str:
    if not hasattr(response, "output"):
        return ""
    for item_field in response.output:
        item = item_field.root
        if getattr(item, "type", None) != "message":
            continue
        if getattr(item, "role", None) != "assistant":
            continue
        for content_part in getattr(item, "content", []):
            text = getattr(content_part, "text", None)
            if isinstance(text, str) and text.strip():
                return text
    return ""


def _extract_function_calls(response: object) -> list[FunctionCall]:
    function_calls: list[FunctionCall] = []
    if not hasattr(response, "output"):
        return function_calls

    for item_field in response.output:
        item = item_field.root
        if isinstance(item, FunctionCall):
            function_calls.append(item)
    return function_calls


class TestGeminiResponsesClientLive:
    """Live integration tests against the real Gemini API."""

    @pytest.mark.asyncio
    async def test_live_non_stream_text_response(self) -> None:
        client = _create_client()
        payload = CreateResponseBody.model_validate(
            {
                "input": "Reply with one short sentence about Tokyo.",
                "stream": False,
            }
        )

        response = await client.create_response(payload=payload)

        assert response.status in {"completed", "incomplete"}
        assert len(response.output) >= 1
        assert _extract_first_assistant_message_text(response)

    @pytest.mark.asyncio
    async def test_live_stream_text_response(self) -> None:
        client = _create_client()
        payload = CreateResponseBody.model_validate(
            {
                "input": "Write a very short greeting in English.",
                "stream": True,
            }
        )

        event_stream = await client.create_response(payload=payload)
        events = [event async for event in event_stream]

        assert events
        assert events[0].type == "response.created"
        assert any(event.type in {"response.output_text.delta", "response.output_text.done"} for event in events)
        assert events[-1].type in {"response.completed", "response.incomplete"}

    @pytest.mark.asyncio
    async def test_live_function_call_and_follow_up(self) -> None:
        client = _create_client()
        function_tool = {
            "type": "function",
            "name": "lookup_weather",
            "description": "Return a fixed weather payload for the requested city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                },
                "required": ["city"],
            },
        }
        user_prompt = (
            "Use the lookup_weather tool to get the weather for Tokyo. "
            "Do not answer from memory. After receiving the tool result, "
            "mention the city and the token MAGIC_WEATHER_TOKEN in your final answer."
        )

        first_payload = CreateResponseBody.model_validate(
            {
                "instructions": "You must call the tool before answering.",
                "input": user_prompt,
                "tools": [function_tool],
                "tool_choice": "required",
                "stream": False,
            }
        )

        first_response = await client.create_response(payload=first_payload)
        function_calls = _extract_function_calls(first_response)

        assert function_calls, "Expected Gemini to return at least one function_call item."
        function_call = function_calls[0]

        tool_output_payload = {
            "city": "Tokyo",
            "weather": "Sunny",
            "temperature_c": 25,
            "token": "MAGIC_WEATHER_TOKEN",
        }

        follow_up_payload = CreateResponseBody.model_validate(
            {
                "instructions": (
                    "After tool results, answer normally and include the exact token "
                    "from the tool output."
                ),
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": user_prompt}],
                    },
                    {
                        "type": "function_call",
                        "call_id": function_call.call_id,
                        "name": function_call.name,
                        "arguments": function_call.arguments,
                    },
                    {
                        "type": "function_call_output",
                        "call_id": function_call.call_id,
                        "output": json.dumps(tool_output_payload, ensure_ascii=True),
                    },
                ],
                "stream": False,
            }
        )

        follow_up_response = await client.create_response(payload=follow_up_payload)
        assistant_text = _extract_assistant_text(_extract_first_assistant_message_text(follow_up_response))

        assert follow_up_response.status in {"completed", "incomplete"}
        assert assistant_text
        assert "MAGIC_WEATHER_TOKEN" in assistant_text
        assert "Tokyo" in assistant_text

    @pytest.mark.asyncio
    async def test_live_json_schema_response(self) -> None:
        client = _create_client()
        payload = CreateResponseBody.model_validate(
            {
                "instructions": "Return valid JSON only.",
                "input": "Return a short JSON object describing one fruit.",
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "fruit_response",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "color": {"type": "string"},
                            },
                            "required": ["name", "color"],
                        },
                    }
                },
                "stream": False,
            }
        )

        response = await client.create_response(payload=payload)
        assistant_text = _extract_assistant_text(_extract_first_assistant_message_text(response))

        assert response.status in {"completed", "incomplete"}
        assert assistant_text
        parsed = json.loads(assistant_text)
        assert isinstance(parsed, dict)
        assert "name" in parsed
        assert "color" in parsed

    @pytest.mark.asyncio
    async def test_live_reasoning_smoke(self) -> None:
        client = _create_client()
        payload = CreateResponseBody.model_validate(
            {
                "input": "What is 17 plus 25? Answer briefly.",
                "reasoning": {"effort": "low"},
                "stream": False,
            }
        )

        response = await client.create_response(payload=payload)

        assert response.status in {"completed", "incomplete"}
        assert len(response.output) >= 1
        assert _extract_first_assistant_message_text(response)
