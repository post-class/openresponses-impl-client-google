from __future__ import annotations

import base64
import binascii
import inspect
import json
import logging
import mimetypes
import time
from collections.abc import AsyncIterator
from typing import Any, get_args, get_origin, override
from urllib.parse import unquote_to_bytes

from google import genai
from google.genai import types
from openresponses_impl_core.client.base_responses_client import BaseResponsesClient
from openresponses_impl_core.models.openresponses_models import (
    CreateResponseBody,
    ErrorPayload,
    ErrorStreamingEvent,
    FunctionCall,
    Message,
    ReasoningBody,
    ResponseCompletedStreamingEvent,
    ResponseContentPartAddedStreamingEvent,
    ResponseContentPartDoneStreamingEvent,
    ResponseCreatedStreamingEvent,
    ResponseFailedStreamingEvent,
    ResponseFunctionCallArgumentsDoneStreamingEvent,
    ResponseIncompleteStreamingEvent,
    ResponseOutputItemAddedStreamingEvent,
    ResponseOutputItemDoneStreamingEvent,
    ResponseOutputTextDeltaStreamingEvent,
    ResponseOutputTextDoneStreamingEvent,
    ResponseReasoningDeltaStreamingEvent,
    ResponseReasoningDoneStreamingEvent,
    ResponseResource,
)
from openresponses_impl_core.models.response_event_types import ResponseStreamingEvent

from openresponses_impl_client_google.utils.copy_util import CopyUtil
from openresponses_impl_client_google.utils.gemini_response_model_util import (
    GeminiResponseModelUtil,
)

logger = logging.getLogger(__name__)


class GeminiResponsesClient(BaseResponsesClient):
    """Responses client for Gemini API."""

    def __init__(
        self,
        *,
        model: str,
        google_api_key: str | None = None,
    ) -> None:
        """Responses client for Google Gemini."""
        if not model:
            raise ValueError("model is required.")

        self._model = model
        self._api_key = google_api_key
        self._client = self._create_client()
        self._call_name_by_call_id: dict[str, str] = {}
        self._thought_signature_by_call_id: dict[str, str] = {}
        self._response_counter = 0

    @override
    async def create_response(
        self,
        payload: CreateResponseBody,
        **kwargs: Any,
    ) -> ResponseResource | AsyncIterator[ResponseStreamingEvent]:
        """Create a response based on the stream field in the payload."""
        if payload.stream:
            return await self._create_response_stream(payload=payload, extra_params=kwargs)
        return await self._create_response_non_stream(payload=payload, extra_params=kwargs)

    async def _create_response_non_stream(
        self,
        *,
        payload: CreateResponseBody,
        extra_params: dict[str, Any] | None = None,
    ) -> ResponseResource:
        request_payload = payload.model_copy(deep=True)
        request_kwargs = self._build_generate_content_kwargs(
            payload=request_payload,
            extra_params=extra_params,
        )
        response = await self._client.aio.models.generate_content(**request_kwargs)
        return GeminiResponseModelUtil.parse_response(
            payload=response,
            request_payload=request_payload,
            model=self._model,
            default_response_id=self._next_response_id(),
            call_name_by_call_id=self._call_name_by_call_id,
            thought_signature_by_call_id=self._thought_signature_by_call_id,
            completed_at=int(time.time()),
        )

    async def _create_response_stream(
        self,
        *,
        payload: CreateResponseBody,
        extra_params: dict[str, Any] | None = None,
    ) -> AsyncIterator[ResponseStreamingEvent]:
        request_payload = payload.model_copy(deep=True)
        request_kwargs = self._build_generate_content_kwargs(
            payload=request_payload,
            extra_params=extra_params,
        )
        return self._iter_stream_events(
            request_payload=request_payload,
            request_kwargs=request_kwargs,
        )

    def _build_generate_content_kwargs(
        self,
        *,
        payload: CreateResponseBody,
        extra_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._warn_unsupported_request_fields(payload=payload)

        contents, system_instruction = self._build_contents_and_system_instruction(payload=payload)
        config = self._build_generate_content_config(
            payload=payload,
            system_instruction=system_instruction,
            extra_params=extra_params,
        )

        request_kwargs: dict[str, Any] = {
            "model": self._model,
            "contents": contents,
        }
        if config is not None:
            request_kwargs["config"] = config
        return request_kwargs

    def _build_contents_and_system_instruction(
        self,
        *,
        payload: CreateResponseBody,
    ) -> tuple[str | list[types.Content], str | None]:
        system_fragments: list[str] = []
        if payload.instructions:
            system_fragments.append(payload.instructions)

        if isinstance(payload.input, str):
            return payload.input, self._join_system_fragments(system_fragments)

        contents: list[types.Content] = []
        for item_param in payload.input or []:
            item = getattr(item_param, "root", item_param)
            converted = self._convert_item_to_content(
                item=item,
                system_fragments=system_fragments,
            )
            if converted is not None:
                contents.append(converted)

        if not contents:
            logger.warning("Gemini request did not contain any input content. Sending an empty string.")
            return "", self._join_system_fragments(system_fragments)

        return contents, self._join_system_fragments(system_fragments)

    def _convert_item_to_content(
        self,
        *,
        item: Any,
        system_fragments: list[str],
    ) -> types.Content | None:
        item_type = getattr(item, "type", None)
        item_dict = self._normalize_model_or_dict(value=item)

        if item_type == "item_reference":
            logger.warning("Gemini client ignores unsupported input item type: item_reference")
            return None

        if item_type == "reasoning":
            logger.warning("Gemini client ignores unsupported input item type: reasoning")
            return None

        if item_type == "message":
            role = item_dict.get("role")
            if role in {"system", "developer"}:
                extracted = self._extract_message_text_for_instruction(content=item_dict.get("content"))
                if extracted:
                    system_fragments.append(extracted)
                return None

            parts = self._convert_message_content_to_parts(content=item_dict.get("content"))
            if not parts:
                return None
            if role == "assistant":
                return types.ModelContent(parts=parts)
            return types.UserContent(parts=parts)

        if item_type == "function_call":
            function_name = item_dict.get("name") or "unknown_function"
            call_id = item_dict.get("call_id") or f"call_input_{len(self._call_name_by_call_id) + 1}"
            arguments = self._parse_function_arguments(item_dict.get("arguments"))
            self._call_name_by_call_id[call_id] = function_name
            function_call_part_kwargs: dict[str, Any] = {
                "function_call": types.FunctionCall(
                    id=call_id,
                    name=function_name,
                    args=arguments,
                )
            }
            thought_signature = self._extract_google_thought_signature(item_dict.get("extensions"))
            if not thought_signature:
                thought_signature = self._thought_signature_by_call_id.get(call_id)
            if thought_signature:
                function_call_part_kwargs["thought_signature"] = self._decode_thought_signature(
                    value=thought_signature
                )
            function_call_part = types.Part(
                **function_call_part_kwargs,
            )
            return types.ModelContent(parts=[function_call_part])

        if item_type == "function_call_output":
            call_id = item_dict.get("call_id")
            function_name = self._call_name_by_call_id.get(call_id or "")
            if not function_name:
                raise ValueError(
                    f"Unknown call_id for function_call_output: {call_id!r}. "
                    "Gemini follow-up requires the original function name."
                )

            function_response = types.FunctionResponse(
                id=call_id,
                name=function_name,
                response={
                    "output": self._convert_function_call_output(
                        output=item_dict.get("output")
                    )
                },
            )
            return types.UserContent(parts=[types.Part(function_response=function_response)])

        logger.warning("Gemini client ignores unsupported input item type: %s", item_type)
        return None

    def _convert_message_content_to_parts(self, *, content: Any) -> list[types.Part]:
        if isinstance(content, str):
            return [types.Part.from_text(text=content)]

        parts: list[types.Part] = []
        for raw_part in content or []:
            part_dict = self._normalize_model_or_dict(value=raw_part)
            part_type = part_dict.get("type")

            if part_type in {"input_text", "output_text", "text", "summary_text", "reasoning_text"}:
                text = part_dict.get("text")
                if text is not None:
                    parts.append(types.Part.from_text(text=text))
                continue

            if part_type == "refusal":
                refusal = part_dict.get("refusal")
                if refusal is not None:
                    parts.append(types.Part.from_text(text=refusal))
                continue

            if part_type == "input_image":
                image_url = part_dict.get("image_url")
                if image_url:
                    parts.append(self._build_media_part(uri=image_url))
                continue

            if part_type == "input_file":
                file_url = part_dict.get("file_url")
                if file_url:
                    parts.append(
                        self._build_media_part(
                            uri=file_url,
                            filename=part_dict.get("filename"),
                        )
                    )
                continue

            if part_type == "input_video":
                video_url = part_dict.get("video_url")
                if video_url:
                    parts.append(self._build_media_part(uri=video_url))
                continue

            logger.warning("Gemini client ignores unsupported message content type: %s", part_type)

        return parts

    def _build_media_part(
        self,
        *,
        uri: str,
        filename: str | None = None,
    ) -> types.Part:
        if uri.startswith("data:"):
            mime_type, data = self._decode_data_uri(uri=uri)
            return types.Part.from_bytes(data=data, mime_type=mime_type)

        mime_type = self._guess_mime_type(uri=uri, filename=filename)
        return types.Part.from_uri(file_uri=uri, mime_type=mime_type)

    def _build_generate_content_config(
        self,
        *,
        payload: CreateResponseBody,
        system_instruction: str | None,
        extra_params: dict[str, Any] | None = None,
    ) -> types.GenerateContentConfig | None:
        config_kwargs: dict[str, Any] = {}

        if system_instruction:
            config_kwargs["system_instruction"] = types.Content(
                role="user",
                parts=[types.Part.from_text(text=system_instruction)],
            )

        tools, tool_choice_config = self._build_tools_and_tool_config(payload=payload)
        if tools:
            config_kwargs["tools"] = tools
            config_kwargs["automatic_function_calling"] = types.AutomaticFunctionCallingConfig(
                disable=True
            )
        if tool_choice_config is not None:
            config_kwargs["tool_config"] = tool_choice_config

        text_config = self._normalize_text_config(payload=payload)
        if text_config["response_mime_type"] is not None:
            config_kwargs["response_mime_type"] = text_config["response_mime_type"]
        if text_config["response_json_schema"] is not None:
            config_kwargs["response_json_schema"] = text_config["response_json_schema"]

        if payload.temperature is not None:
            config_kwargs["temperature"] = payload.temperature
        if payload.top_p is not None:
            config_kwargs["top_p"] = payload.top_p
        if payload.presence_penalty is not None:
            config_kwargs["presence_penalty"] = payload.presence_penalty
        if payload.frequency_penalty is not None:
            config_kwargs["frequency_penalty"] = payload.frequency_penalty
        if payload.max_output_tokens is not None:
            config_kwargs["max_output_tokens"] = payload.max_output_tokens
        if payload.service_tier is not None:
            config_kwargs["service_tier"] = payload.service_tier.value

        thinking_config = self._build_thinking_config(payload=payload)
        if thinking_config is not None:
            config_kwargs["thinking_config"] = thinking_config

        if extra_params:
            for key, value in extra_params.items():
                if key in types.GenerateContentConfig.model_fields:
                    config_kwargs[key] = value
                    continue
                logger.warning("Gemini client ignores unsupported provider-specific config key: %s", key)

        if not config_kwargs:
            return None
        return types.GenerateContentConfig(**config_kwargs)

    def _build_tools_and_tool_config(
        self,
        *,
        payload: CreateResponseBody,
    ) -> tuple[list[types.Tool] | None, types.ToolConfig | None]:
        if not payload.tools:
            return None, None

        function_declarations: list[types.FunctionDeclaration] = []
        function_names: list[str] = []
        builtin_tools: list[types.Tool] = []

        for tool_param in payload.tools:
            tool_root = getattr(tool_param, "root", tool_param)
            tool_dict = self._normalize_model_or_dict(value=tool_root)
            tool_type = tool_dict.get("type")
            if tool_type == "function":
                function_name = tool_dict.get("name")
                if not function_name:
                    continue
                function_names.append(function_name)
                function_declarations.append(
                    types.FunctionDeclaration(
                        name=function_name,
                        description=tool_dict.get("description"),
                        parameters_json_schema=CopyUtil.deep_copy(
                            tool_dict.get("parameters") or {"type": "object", "properties": {}}
                        ),
                    )
                )
                continue

            builtin_tool = self._build_builtin_tool(tool_dict=tool_dict)
            if builtin_tool is not None:
                builtin_tools.append(builtin_tool)
                continue

            logger.warning(
                "Gemini client ignores unsupported generic tool type: %s",
                tool_type,
            )

        if not function_declarations and not builtin_tools:
            return None, None

        tool_config = self._build_tool_config(
            payload=payload,
            function_names=function_names,
            include_server_side_tool_invocations=bool(builtin_tools),
        )

        tools: list[types.Tool] = []
        if function_declarations:
            tools.append(types.Tool(function_declarations=function_declarations))
        tools.extend(builtin_tools)
        return tools, tool_config

    def _build_builtin_tool(self, *, tool_dict: dict[str, Any]) -> types.Tool | None:
        tool_type = tool_dict.get("type")
        if not isinstance(tool_type, str):
            return None

        supported_builtin_tool_types = self._get_supported_builtin_tool_types()
        if tool_type not in supported_builtin_tool_types:
            return None

        builtin_tool_payload = CopyUtil.deep_copy(
            {
                key: value
                for key, value in tool_dict.items()
                if key not in {"type", "description"} and value is not None
            }
        )
        try:
            return types.Tool(**{tool_type: builtin_tool_payload})
        except Exception as exc:
            error_message = f"Invalid Gemini builtin tool config for {tool_type!r}."
            raise ValueError(error_message) from exc

    def _get_supported_builtin_tool_types(self) -> set[str]:
        supported_builtin_tool_types: set[str] = set()
        for field_name, field_info in types.Tool.model_fields.items():
            if field_name == "function_declarations":
                continue
            model_type = self._resolve_builtin_tool_model_type(annotation=field_info.annotation)
            if model_type is None:
                continue
            supported_builtin_tool_types.add(field_name)
        return supported_builtin_tool_types

    def _resolve_builtin_tool_model_type(self, *, annotation: Any) -> type[Any] | None:
        if hasattr(annotation, "model_fields"):
            return annotation

        origin = get_origin(annotation)
        if origin in {list, tuple, set, frozenset}:
            return None

        for candidate in get_args(annotation):
            if candidate is type(None):
                continue
            resolved = self._resolve_builtin_tool_model_type(annotation=candidate)
            if resolved is not None:
                return resolved
        return None

    def _build_tool_config(
        self,
        *,
        payload: CreateResponseBody,
        function_names: list[str],
        include_server_side_tool_invocations: bool,
    ) -> types.ToolConfig | None:
        function_calling_config: types.FunctionCallingConfig | None = None

        if payload.tool_choice:
            tool_choice = payload.tool_choice.model_dump(mode="json")

            if isinstance(tool_choice, str):
                normalized_choice = tool_choice.lower()
                if normalized_choice == "none":
                    function_calling_config = types.FunctionCallingConfig(mode="NONE")
                elif normalized_choice == "auto":
                    function_calling_config = types.FunctionCallingConfig(mode="AUTO")
                elif normalized_choice == "required":
                    function_calling_config = types.FunctionCallingConfig(
                        mode="ANY",
                        allowed_function_names=function_names,
                    )
                else:
                    logger.warning("Gemini client ignores unsupported string tool_choice: %s", tool_choice)
            elif isinstance(tool_choice, dict):
                if not tool_choice:
                    function_calling_config = types.FunctionCallingConfig(
                        mode="ANY",
                        allowed_function_names=function_names,
                    )
                elif tool_choice.get("type") == "function":
                    function_name = tool_choice.get("name")
                    if function_name:
                        function_calling_config = types.FunctionCallingConfig(
                            mode="ANY",
                            allowed_function_names=[function_name],
                        )
                elif tool_choice.get("type") == "allowed_tools":
                    allowed_function_names = [
                        entry.get("name")
                        for entry in tool_choice.get("tools", [])
                        if isinstance(entry, dict) and entry.get("type") == "function" and entry.get("name")
                    ]
                    if allowed_function_names:
                        function_calling_config = types.FunctionCallingConfig(
                            mode="ANY",
                            allowed_function_names=allowed_function_names,
                        )
                    else:
                        logger.warning(
                            "Gemini client ignores allowed_tools without function entries."
                        )
                else:
                    logger.warning("Gemini client ignores unsupported tool_choice payload: %s", tool_choice)

        if function_calling_config is None and not include_server_side_tool_invocations:
            return None

        return types.ToolConfig(
            function_calling_config=function_calling_config,
            include_server_side_tool_invocations=include_server_side_tool_invocations or None,
        )

    def _normalize_text_config(self, *, payload: CreateResponseBody) -> dict[str, Any]:
        response_mime_type = "text/plain"
        response_json_schema: dict[str, Any] | None = None
        text_payload = payload.text.model_dump(mode="json", exclude_none=True) if payload.text else None
        if not text_payload:
            return {
                "response_mime_type": response_mime_type,
                "response_json_schema": None,
            }

        format_payload = text_payload.get("format") or {}
        if format_payload.get("type") == "json_schema":
            response_mime_type = "application/json"
            response_json_schema = CopyUtil.deep_copy(
                format_payload.get("schema") or format_payload.get("schema_")
            )

        return {
            "response_mime_type": response_mime_type,
            "response_json_schema": response_json_schema,
        }

    def _build_thinking_config(self, *, payload: CreateResponseBody) -> types.ThinkingConfig | None:
        if payload.reasoning is None:
            return None

        reasoning_payload = payload.reasoning.model_dump(mode="json", exclude_none=True)
        thinking_config_kwargs: dict[str, Any] = {}
        effort = reasoning_payload.get("effort")
        if effort == "none":
            thinking_config_kwargs["thinking_budget"] = 0
        elif effort == "low":
            thinking_config_kwargs["thinking_level"] = "LOW"
        elif effort == "medium":
            thinking_config_kwargs["thinking_level"] = "MEDIUM"
        elif effort == "high":
            thinking_config_kwargs["thinking_level"] = "HIGH"
        elif effort == "xhigh":
            logger.warning("Gemini client maps reasoning.effort=xhigh to thinking_level=HIGH")
            thinking_config_kwargs["thinking_level"] = "HIGH"

        if reasoning_payload.get("summary") is not None:
            logger.warning("Gemini client ignores unsupported reasoning.summary setting.")

        if not thinking_config_kwargs:
            return None
        return types.ThinkingConfig(**thinking_config_kwargs)

    async def _iter_stream_events(
        self,
        *,
        request_payload: CreateResponseBody,
        request_kwargs: dict[str, Any],
    ) -> AsyncIterator[ResponseStreamingEvent]:
        sequence_number = 1
        fallback_response_id = self._next_response_id()
        aggregate_payload: dict[str, Any] = {}
        previous_response: ResponseResource | None = None
        stream = await self._client.aio.models.generate_content_stream(**request_kwargs)

        try:
            async for chunk in stream:
                normalized_chunk = GeminiResponseModelUtil._normalize_payload(payload=chunk)
                aggregate_payload = self._merge_stream_payloads(
                    existing=aggregate_payload,
                    incoming=normalized_chunk,
                )
                current_response = GeminiResponseModelUtil.parse_response(
                    payload=aggregate_payload,
                    request_payload=request_payload,
                    model=self._model,
                    default_response_id=fallback_response_id,
                    call_name_by_call_id=self._call_name_by_call_id,
                    thought_signature_by_call_id=self._thought_signature_by_call_id,
                    status_override="in_progress",
                    completed_at=None,
                )

                if previous_response is None:
                    yield ResponseCreatedStreamingEvent(
                        type="response.created",
                        sequence_number=sequence_number,
                        response=current_response,
                    )
                    sequence_number += 1

                emitted, sequence_number = self._build_incremental_events(
                    previous_response=previous_response,
                    current_response=current_response,
                    sequence_number=sequence_number,
                )
                for event in emitted:
                    yield event

                previous_response = current_response

            if previous_response is None:
                final_response = GeminiResponseModelUtil.parse_response(
                    payload=aggregate_payload or {},
                    request_payload=request_payload,
                    model=self._model,
                    default_response_id=fallback_response_id,
                    call_name_by_call_id=self._call_name_by_call_id,
                    thought_signature_by_call_id=self._thought_signature_by_call_id,
                    completed_at=int(time.time()),
                )
                yield ResponseCreatedStreamingEvent(
                    type="response.created",
                    sequence_number=sequence_number,
                    response=final_response,
                )
                sequence_number += 1
            else:
                final_response = GeminiResponseModelUtil.parse_response(
                    payload=aggregate_payload,
                    request_payload=request_payload,
                    model=self._model,
                    default_response_id=fallback_response_id,
                    call_name_by_call_id=self._call_name_by_call_id,
                    thought_signature_by_call_id=self._thought_signature_by_call_id,
                    completed_at=int(time.time()),
                )

            terminal_output_events, sequence_number = self._build_terminal_output_events(
                response=final_response,
                sequence_number=sequence_number,
            )
            for event in terminal_output_events:
                yield event

            yield self._build_terminal_response_event(
                response=final_response,
                sequence_number=sequence_number,
            )
        except Exception as exc:
            yield ErrorStreamingEvent(
                type="error",
                sequence_number=sequence_number,
                error=ErrorPayload(
                    type="stream_error",
                    code=None,
                    message=str(exc),
                    param=None,
                    headers=None,
                ),
            )
        finally:
            await self._close_stream(stream=stream)

    def _build_incremental_events(
        self,
        *,
        previous_response: ResponseResource | None,
        current_response: ResponseResource,
        sequence_number: int,
    ) -> tuple[list[ResponseStreamingEvent], int]:
        events: list[ResponseStreamingEvent] = []
        previous_items_by_id = self._index_output_items(response=previous_response)

        for output_index, item_field in enumerate(current_response.output):
            current_item = item_field.root
            previous_item = previous_items_by_id.get(current_item.id)

            if previous_item is None:
                events.append(
                    ResponseOutputItemAddedStreamingEvent(
                        type="response.output_item.added",
                        sequence_number=sequence_number,
                        output_index=output_index,
                        item=item_field.model_dump(mode="json"),
                    )
                )
                sequence_number += 1

                if isinstance(current_item, Message):
                    content_part = current_item.content[0]
                    events.append(
                        ResponseContentPartAddedStreamingEvent(
                            type="response.content_part.added",
                            sequence_number=sequence_number,
                            item_id=current_item.id,
                            output_index=output_index,
                            content_index=0,
                            part=content_part.model_dump(mode="json"),
                        )
                    )
                    sequence_number += 1

                    delta = self._extract_message_text(item=current_item)
                    if delta:
                        events.append(
                            ResponseOutputTextDeltaStreamingEvent(
                                type="response.output_text.delta",
                                sequence_number=sequence_number,
                                item_id=current_item.id,
                                output_index=output_index,
                                content_index=0,
                                delta=delta,
                                logprobs=None,
                                obfuscation=None,
                            )
                        )
                        sequence_number += 1
                    continue

                if isinstance(current_item, ReasoningBody):
                    delta = self._extract_reasoning_text(item=current_item)
                    if delta:
                        events.append(
                            ResponseReasoningDeltaStreamingEvent(
                                type="response.reasoning.delta",
                                sequence_number=sequence_number,
                                item_id=current_item.id,
                                output_index=output_index,
                                content_index=0,
                                delta=delta,
                                obfuscation=None,
                            )
                        )
                        sequence_number += 1
                    continue

                continue

            if isinstance(current_item, Message) and isinstance(previous_item, Message):
                previous_text = self._extract_message_text(item=previous_item)
                current_text = self._extract_message_text(item=current_item)
                delta = self._calculate_delta(previous_text=previous_text, current_text=current_text)
                if delta:
                    events.append(
                        ResponseOutputTextDeltaStreamingEvent(
                            type="response.output_text.delta",
                            sequence_number=sequence_number,
                            item_id=current_item.id,
                            output_index=output_index,
                            content_index=0,
                            delta=delta,
                            logprobs=None,
                            obfuscation=None,
                        )
                    )
                    sequence_number += 1
                continue

            if isinstance(current_item, ReasoningBody) and isinstance(previous_item, ReasoningBody):
                previous_text = self._extract_reasoning_text(item=previous_item)
                current_text = self._extract_reasoning_text(item=current_item)
                delta = self._calculate_delta(previous_text=previous_text, current_text=current_text)
                if delta:
                    events.append(
                        ResponseReasoningDeltaStreamingEvent(
                            type="response.reasoning.delta",
                            sequence_number=sequence_number,
                            item_id=current_item.id,
                            output_index=output_index,
                            content_index=0,
                            delta=delta,
                            obfuscation=None,
                        )
                    )
                    sequence_number += 1

        return events, sequence_number

    def _build_terminal_output_events(
        self,
        *,
        response: ResponseResource,
        sequence_number: int,
    ) -> tuple[list[ResponseStreamingEvent], int]:
        events: list[ResponseStreamingEvent] = []

        for output_index, item_field in enumerate(response.output):
            item = item_field.root
            if isinstance(item, Message):
                content_part = item.content[0]
                text = self._extract_message_text(item=item)
                events.append(
                    ResponseOutputTextDoneStreamingEvent(
                        type="response.output_text.done",
                        sequence_number=sequence_number,
                        item_id=item.id,
                        output_index=output_index,
                        content_index=0,
                        text=text,
                        logprobs=None,
                    )
                )
                sequence_number += 1
                events.append(
                    ResponseContentPartDoneStreamingEvent(
                        type="response.content_part.done",
                        sequence_number=sequence_number,
                        item_id=item.id,
                        output_index=output_index,
                        content_index=0,
                        part=content_part.model_dump(mode="json"),
                    )
                )
                sequence_number += 1
                events.append(
                    ResponseOutputItemDoneStreamingEvent(
                        type="response.output_item.done",
                        sequence_number=sequence_number,
                        output_index=output_index,
                        item=item_field.model_dump(mode="json"),
                    )
                )
                sequence_number += 1
                continue

            if isinstance(item, ReasoningBody):
                text = self._extract_reasoning_text(item=item)
                events.append(
                    ResponseReasoningDoneStreamingEvent(
                        type="response.reasoning.done",
                        sequence_number=sequence_number,
                        item_id=item.id,
                        output_index=output_index,
                        content_index=0,
                        text=text,
                    )
                )
                sequence_number += 1
                events.append(
                    ResponseOutputItemDoneStreamingEvent(
                        type="response.output_item.done",
                        sequence_number=sequence_number,
                        output_index=output_index,
                        item=item_field.model_dump(mode="json"),
                    )
                )
                sequence_number += 1
                continue

            if isinstance(item, FunctionCall):
                events.append(
                    ResponseFunctionCallArgumentsDoneStreamingEvent(
                        type="response.function_call_arguments.done",
                        sequence_number=sequence_number,
                        item_id=item.id,
                        output_index=output_index,
                        arguments=item.arguments,
                    )
                )
                sequence_number += 1
                events.append(
                    ResponseOutputItemDoneStreamingEvent(
                        type="response.output_item.done",
                        sequence_number=sequence_number,
                        output_index=output_index,
                        item=item_field.model_dump(mode="json"),
                    )
                )
                sequence_number += 1

        return events, sequence_number

    def _build_terminal_response_event(
        self,
        *,
        response: ResponseResource,
        sequence_number: int,
    ) -> ResponseStreamingEvent:
        if response.status == "failed":
            return ResponseFailedStreamingEvent(
                type="response.failed",
                sequence_number=sequence_number,
                response=response,
            )
        if response.status == "incomplete":
            return ResponseIncompleteStreamingEvent(
                type="response.incomplete",
                sequence_number=sequence_number,
                response=response,
            )
        return ResponseCompletedStreamingEvent(
            type="response.completed",
            sequence_number=sequence_number,
            response=response,
        )

    def _merge_stream_payloads(
        self,
        *,
        existing: dict[str, Any],
        incoming: dict[str, Any],
    ) -> dict[str, Any]:
        if not existing:
            return CopyUtil.deep_copy(incoming)

        merged = CopyUtil.deep_copy(existing)
        for key, value in incoming.items():
            if key == "candidates":
                continue
            if value is not None:
                merged[key] = CopyUtil.deep_copy(value)

        incoming_candidates = incoming.get("candidates") or []
        if incoming_candidates:
            merged_candidates = list(merged.get("candidates") or [])
            for index, incoming_candidate in enumerate(incoming_candidates):
                if index >= len(merged_candidates):
                    merged_candidates.append(CopyUtil.deep_copy(incoming_candidate))
                    continue
                merged_candidates[index] = self._merge_dict_tree(
                    base=merged_candidates[index],
                    incoming=incoming_candidate,
                )
            merged["candidates"] = merged_candidates

        return merged

    def _merge_dict_tree(self, *, base: Any, incoming: Any) -> Any:
        if isinstance(base, dict) and isinstance(incoming, dict):
            merged = CopyUtil.deep_copy(base)
            for key, value in incoming.items():
                if value is None:
                    continue
                if key in merged:
                    merged[key] = self._merge_dict_tree(base=merged[key], incoming=value)
                else:
                    merged[key] = CopyUtil.deep_copy(value)
            return merged

        if isinstance(base, list) and isinstance(incoming, list):
            merged_list = list(CopyUtil.deep_copy(base))
            for index, value in enumerate(incoming):
                if index >= len(merged_list):
                    merged_list.append(CopyUtil.deep_copy(value))
                    continue
                merged_list[index] = self._merge_dict_tree(base=merged_list[index], incoming=value)
            return merged_list

        if isinstance(base, str) and isinstance(incoming, str):
            if incoming.startswith(base):
                return incoming
            if base.startswith(incoming):
                return base
        return CopyUtil.deep_copy(incoming)

    def _index_output_items(
        self,
        *,
        response: ResponseResource | None,
    ) -> dict[str, Message | ReasoningBody | FunctionCall | Any]:
        if response is None:
            return {}
        return {
            item_field.root.id: item_field.root
            for item_field in response.output
            if getattr(item_field.root, "id", None)
        }

    def _extract_message_text(self, *, item: Message) -> str:
        if not item.content:
            return ""
        content_part = item.content[0]
        if hasattr(content_part, "text"):
            return str(content_part.text)
        if hasattr(content_part, "refusal"):
            return str(content_part.refusal)
        return ""

    def _extract_reasoning_text(self, *, item: ReasoningBody) -> str:
        if item.content:
            content_part = item.content[0]
            if hasattr(content_part, "text"):
                return str(content_part.text)
        if item.summary:
            summary_part = item.summary[0]
            if hasattr(summary_part, "text"):
                return str(summary_part.text)
        return ""

    def _calculate_delta(self, *, previous_text: str, current_text: str) -> str:
        if not previous_text:
            return current_text
        if current_text.startswith(previous_text):
            return current_text[len(previous_text) :]
        return current_text

    async def _close_stream(self, *, stream: Any) -> None:
        aclose_method = getattr(stream, "aclose", None)
        if callable(aclose_method):
            result = aclose_method()
            if inspect.isawaitable(result):
                await result
            return

        close_method = getattr(stream, "close", None)
        if callable(close_method):
            result = close_method()
            if inspect.isawaitable(result):
                await result

    def _warn_unsupported_request_fields(self, *, payload: CreateResponseBody) -> None:
        unsupported_fields = [
            "previous_response_id",
            "store",
            "background",
            "parallel_tool_calls",
            "max_tool_calls",
            "truncation",
            "include",
            "safety_identifier",
            "prompt_cache_key",
        ]
        for field_name in unsupported_fields:
            if getattr(payload, field_name) is not None:
                logger.warning("Gemini client ignores unsupported OpenResponses field: %s", field_name)

    def _extract_message_text_for_instruction(self, *, content: Any) -> str:
        if isinstance(content, str):
            return content

        fragments: list[str] = []
        for raw_part in content or []:
            part_dict = self._normalize_model_or_dict(value=raw_part)
            if part_dict.get("type") in {
                "input_text",
                "output_text",
                "text",
                "summary_text",
                "reasoning_text",
            }:
                if part_dict.get("text"):
                    fragments.append(str(part_dict["text"]))
                continue
            if part_dict.get("type") == "refusal" and part_dict.get("refusal"):
                fragments.append(str(part_dict["refusal"]))
                continue
            logger.warning(
                "Gemini client ignores non-text system/developer content type: %s",
                part_dict.get("type"),
            )
        return "\n".join(fragment for fragment in fragments if fragment)

    def _convert_function_call_output(self, *, output: Any) -> Any:
        if isinstance(output, str):
            return output
        if hasattr(output, "model_dump"):
            return output.model_dump(mode="json", exclude_none=True)
        if isinstance(output, list):
            return [self._normalize_model_or_dict(value=item) for item in output]
        if isinstance(output, dict):
            return output
        return str(output)

    def _parse_function_arguments(self, value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                logger.warning("Gemini client received non-JSON function arguments; wrapping raw text.")
                return {"raw_arguments": value}
            if isinstance(parsed, dict):
                return parsed
            return {"value": parsed}
        return {"value": value}

    def _decode_data_uri(self, *, uri: str) -> tuple[str, bytes]:
        header, encoded = uri.split(",", 1)
        mime_type = "application/octet-stream"
        if ";" in header:
            mime_type_candidate = header[5:].split(";", 1)[0]
            if mime_type_candidate:
                mime_type = mime_type_candidate
        elif header.startswith("data:") and len(header) > 5:
            mime_type = header[5:]

        try:
            if ";base64" in header:
                return mime_type, base64.b64decode(encoded)
            return mime_type, unquote_to_bytes(encoded)
        except (binascii.Error, ValueError) as exc:
            raise ValueError(f"Invalid data URI payload: {exc}") from exc

    def _decode_thought_signature(self, *, value: str) -> bytes:
        padded_value = value + ("=" * (-len(value) % 4))
        try:
            return base64.urlsafe_b64decode(padded_value)
        except (binascii.Error, ValueError) as exc:
            raise ValueError(f"Invalid thought_signature payload: {exc}") from exc

    def _extract_google_thought_signature(self, extensions: Any) -> str | None:
        if not isinstance(extensions, dict):
            return None
        google_extensions = extensions.get("google")
        if not isinstance(google_extensions, dict):
            return None
        return GeminiResponseModelUtil._normalize_thought_signature(
            google_extensions.get("thought_signature")
        )

    def _guess_mime_type(self, *, uri: str, filename: str | None = None) -> str:
        mime_type, _ = mimetypes.guess_type(filename or uri)
        if mime_type:
            return mime_type
        logger.warning(
            "Gemini client could not infer MIME type for %s. Falling back to application/octet-stream.",
            filename or uri,
        )
        return "application/octet-stream"

    def _join_system_fragments(self, fragments: list[str]) -> str | None:
        normalized = [fragment.strip() for fragment in fragments if fragment and fragment.strip()]
        if not normalized:
            return None
        return "\n\n".join(normalized)

    def _normalize_model_or_dict(self, *, value: Any) -> dict[str, Any]:
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json", exclude_none=True)
        if isinstance(value, dict):
            return value
        raise ValueError(f"Unsupported value type: {type(value).__name__}")

    def _next_response_id(self) -> str:
        self._response_counter += 1
        return f"resp_gemini_{self._response_counter}"

    def _create_client(self) -> genai.Client:
        if self._api_key:
            return genai.Client(api_key=self._api_key)
        return genai.Client()
