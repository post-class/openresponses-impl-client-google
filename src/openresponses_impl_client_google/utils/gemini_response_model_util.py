from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any

from openresponses_impl_core.models.openresponses_models import (
    ErrorPayload,
    ErrorStreamingEvent,
    ResponseResource,
)


class GeminiResponseModelUtil:
    """Utility helpers for Gemini response normalization."""

    @staticmethod
    def parse_response(
        *,
        payload: Any,
        request_payload: Any,
        model: str,
        default_response_id: str,
        call_name_by_call_id: dict[str, str] | None = None,
        completed_at: int | None = None,
        status_override: str | None = None,
        created_at_override: int | None = None,
    ) -> ResponseResource:
        """Convert a Gemini response payload into ResponseResource."""
        normalized = GeminiResponseModelUtil._normalize_payload(payload=payload)
        request_dict = GeminiResponseModelUtil._normalize_request_payload(payload=request_payload)
        response_payload = GeminiResponseModelUtil._build_response_payload(
            payload=normalized,
            request_dict=request_dict,
            model=model,
            default_response_id=default_response_id,
            call_name_by_call_id=call_name_by_call_id,
            completed_at=completed_at,
            status_override=status_override,
            created_at_override=created_at_override,
        )
        return ResponseResource.model_validate(response_payload)

    @staticmethod
    def _build_response_payload(
        *,
        payload: dict[str, Any],
        request_dict: dict[str, Any],
        model: str,
        default_response_id: str,
        call_name_by_call_id: dict[str, str] | None = None,
        completed_at: int | None = None,
        status_override: str | None = None,
        created_at_override: int | None = None,
    ) -> dict[str, Any]:
        response_id = payload.get("response_id") or default_response_id
        created_at = (
            created_at_override
            or GeminiResponseModelUtil._coerce_timestamp(payload.get("create_time"))
            or int(time.time())
        )

        output_items = GeminiResponseModelUtil._build_output_items(
            payload=payload,
            response_id=response_id,
            call_name_by_call_id=call_name_by_call_id,
        )
        status, incomplete_details, error = GeminiResponseModelUtil._derive_status(
            payload=payload,
            output_items=output_items,
        )

        if status_override is not None:
            status = status_override
            if status_override not in {"failed", "incomplete"}:
                incomplete_details = None
                error = None

        response_completed_at = completed_at
        if response_completed_at is None and status == "completed":
            response_completed_at = created_at

        metadata = GeminiResponseModelUtil._build_metadata(
            request_dict=request_dict,
            payload=payload,
        )

        return {
            "id": response_id,
            "object": "response",
            "created_at": created_at,
            "completed_at": response_completed_at if status == "completed" else None,
            "status": status,
            "incomplete_details": incomplete_details,
            "model": model,
            "previous_response_id": request_dict.get("previous_response_id"),
            "instructions": request_dict.get("instructions"),
            "output": output_items,
            "error": error,
            "tools": GeminiResponseModelUtil._normalize_tools(request_dict=request_dict),
            "tool_choice": GeminiResponseModelUtil._resolve_tool_choice(request_dict=request_dict),
            "truncation": request_dict.get("truncation", "disabled"),
            "parallel_tool_calls": bool(request_dict.get("parallel_tool_calls", False)),
            "text": GeminiResponseModelUtil._resolve_text_config(request_dict=request_dict),
            "top_p": request_dict.get("top_p", 1.0),
            "presence_penalty": request_dict.get("presence_penalty", 0.0),
            "frequency_penalty": request_dict.get("frequency_penalty", 0.0),
            "top_logprobs": request_dict.get("top_logprobs", 0),
            "temperature": request_dict.get("temperature", 1.0),
            "reasoning": request_dict.get("reasoning"),
            "usage": GeminiResponseModelUtil._build_usage(payload.get("usage_metadata")),
            "max_output_tokens": request_dict.get("max_output_tokens"),
            "max_tool_calls": request_dict.get("max_tool_calls"),
            "store": bool(request_dict.get("store", False)),
            "background": bool(request_dict.get("background", False)),
            "service_tier": request_dict.get("service_tier", "auto"),
            "metadata": metadata,
            "safety_identifier": request_dict.get("safety_identifier"),
            "prompt_cache_key": request_dict.get("prompt_cache_key"),
        }

    @staticmethod
    def _build_output_items(
        *,
        payload: dict[str, Any],
        response_id: str,
        call_name_by_call_id: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        candidate = GeminiResponseModelUtil._get_first_candidate(payload=payload)
        parts = GeminiResponseModelUtil._get_candidate_parts(candidate=candidate)
        citations = GeminiResponseModelUtil._build_annotations(candidate=candidate)
        logprobs = GeminiResponseModelUtil._build_logprobs(candidate=candidate)

        output_items: list[dict[str, Any]] = []
        attached_annotations = False
        attached_logprobs = False

        for part_index, part in enumerate(parts):
            if isinstance(part.get("text"), str):
                text = part["text"]
                if part.get("thought"):
                    if text:
                        output_items.append(
                            {
                                "type": "reasoning",
                                "id": f"rsn_{response_id}_{part_index}",
                                "content": [{"type": "reasoning_text", "text": text}],
                                "summary": [{"type": "summary_text", "text": text}],
                                "encrypted_content": None,
                            }
                        )
                    continue

                if not text:
                    continue

                annotations = citations if not attached_annotations else []
                item_logprobs = logprobs if not attached_logprobs else None
                output_items.append(
                    {
                        "type": "message",
                        "id": f"msg_{response_id}_{part_index}",
                        "status": "completed",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": text,
                                "annotations": annotations,
                                "logprobs": item_logprobs,
                            }
                        ],
                    }
                )
                attached_annotations = True
                attached_logprobs = True
                continue

            function_call = part.get("function_call")
            if isinstance(function_call, dict):
                function_name = function_call.get("name") or "unknown_function"
                call_id = function_call.get("id") or f"call_{response_id}_{part_index}"
                arguments = GeminiResponseModelUtil._json_dumps(function_call.get("args", {}))
                output_items.append(
                    {
                        "type": "function_call",
                        "id": f"fc_{response_id}_{part_index}",
                        "call_id": call_id,
                        "name": function_name,
                        "arguments": arguments,
                        "status": "completed",
                    }
                )
                if call_name_by_call_id is not None:
                    call_name_by_call_id[call_id] = function_name

        return output_items

    @staticmethod
    def _derive_status(
        *,
        payload: dict[str, Any],
        output_items: list[dict[str, Any]],
    ) -> tuple[str, dict[str, Any] | None, dict[str, Any] | None]:
        prompt_feedback = payload.get("prompt_feedback") or {}
        if prompt_feedback.get("block_reason"):
            block_reason = str(prompt_feedback.get("block_reason"))
            message = prompt_feedback.get("block_reason_message") or block_reason
            return (
                "failed",
                None,
                {
                    "code": block_reason.lower(),
                    "message": message,
                },
            )

        candidate = GeminiResponseModelUtil._get_first_candidate(payload=payload)
        if candidate is None:
            return (
                "failed",
                None,
                {
                    "code": "no_candidates",
                    "message": "Gemini response did not contain any candidates.",
                },
            )

        finish_reason_raw = candidate.get("finish_reason")
        finish_reason = str(finish_reason_raw or "").upper()
        has_function_call = any(item.get("type") == "function_call" for item in output_items)

        if finish_reason in {"", "STOP", "FINISH_REASON_UNSPECIFIED"}:
            return "completed", None, None
        if has_function_call:
            return "completed", None, None
        if finish_reason == "MAX_TOKENS":
            return "incomplete", {"reason": "max_tokens"}, None
        return (
            "incomplete",
            {"reason": finish_reason.lower() if finish_reason else "unknown"},
            None,
        )

    @staticmethod
    def _build_usage(usage_payload: Any) -> dict[str, Any] | None:
        if not isinstance(usage_payload, dict):
            return None

        input_tokens = GeminiResponseModelUtil._safe_int(usage_payload.get("prompt_token_count"))
        output_tokens = GeminiResponseModelUtil._safe_int(
            usage_payload.get("response_token_count")
            or usage_payload.get("candidates_token_count")
        )
        total_tokens = GeminiResponseModelUtil._safe_int(usage_payload.get("total_token_count"))

        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens or (input_tokens + output_tokens),
            "input_tokens_details": {
                "cached_tokens": GeminiResponseModelUtil._safe_int(
                    usage_payload.get("cached_content_token_count")
                )
            },
            "output_tokens_details": {
                "reasoning_tokens": GeminiResponseModelUtil._safe_int(
                    usage_payload.get("thoughts_token_count")
                )
            },
        }

    @staticmethod
    def _build_metadata(
        *,
        request_dict: dict[str, Any],
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        metadata = dict(request_dict.get("metadata") or {})
        if payload.get("model_version") is not None:
            metadata["gemini_model_version"] = payload.get("model_version")
        if payload.get("model_status") is not None:
            metadata["gemini_model_status"] = payload.get("model_status")
        if payload.get("prompt_feedback") is not None:
            metadata["gemini_prompt_feedback"] = payload.get("prompt_feedback")
        if payload.get("automatic_function_calling_history") is not None:
            metadata["gemini_automatic_function_calling_history"] = payload.get(
                "automatic_function_calling_history"
            )
        return metadata

    @staticmethod
    def _build_annotations(*, candidate: dict[str, Any] | None) -> list[dict[str, Any]]:
        if candidate is None:
            return []
        citation_metadata = candidate.get("citation_metadata") or {}
        citations = citation_metadata.get("citations") or []
        annotations: list[dict[str, Any]] = []

        for citation in citations:
            if not isinstance(citation, dict):
                continue
            url = citation.get("uri")
            title = citation.get("title")
            start_index = citation.get("start_index")
            end_index = citation.get("end_index")
            if not url or title is None or start_index is None or end_index is None:
                continue
            annotations.append(
                {
                    "type": "url_citation",
                    "url": url,
                    "title": title,
                    "start_index": start_index,
                    "end_index": end_index,
                }
            )

        return annotations

    @staticmethod
    def _build_logprobs(*, candidate: dict[str, Any] | None) -> list[dict[str, Any]] | None:
        if candidate is None:
            return None
        logprobs_result = candidate.get("logprobs_result") or {}
        chosen_candidates = logprobs_result.get("chosen_candidates") or []
        top_candidates = logprobs_result.get("top_candidates") or []
        if not chosen_candidates:
            return None

        result: list[dict[str, Any]] = []
        for index, chosen in enumerate(chosen_candidates):
            if not isinstance(chosen, dict):
                continue
            token = chosen.get("token") or ""
            top_candidates_entry = top_candidates[index] if index < len(top_candidates) else {}
            top_candidate_items = top_candidates_entry.get("candidates") or []
            result.append(
                {
                    "token": token,
                    "logprob": float(chosen.get("log_probability") or 0.0),
                    "bytes": list(token.encode("utf-8")),
                    "top_logprobs": [
                        {
                            "token": top_candidate.get("token") or "",
                            "logprob": float(top_candidate.get("log_probability") or 0.0),
                            "bytes": list(str(top_candidate.get("token") or "").encode("utf-8")),
                        }
                        for top_candidate in top_candidate_items
                        if isinstance(top_candidate, dict)
                    ],
                }
            )

        return result

    @staticmethod
    def _resolve_tool_choice(*, request_dict: dict[str, Any]) -> Any:
        if request_dict.get("tool_choice") is not None:
            return request_dict["tool_choice"]
        if request_dict.get("tools"):
            return "auto"
        return "none"

    @staticmethod
    def _normalize_tools(*, request_dict: dict[str, Any]) -> list[dict[str, Any]]:
        tools = request_dict.get("tools") or []
        normalized: list[dict[str, Any]] = []
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            if tool.get("type") == "function":
                normalized.append(
                    {
                        "type": "function",
                        "name": tool.get("name"),
                        "description": tool.get("description"),
                        "parameters": tool.get("parameters"),
                        "strict": tool.get("strict"),
                    }
                )
                continue
            normalized.append(tool)
        return normalized

    @staticmethod
    def _resolve_text_config(*, request_dict: dict[str, Any]) -> dict[str, Any]:
        if request_dict.get("text") is not None:
            return request_dict["text"]
        return {"format": {"type": "text"}, "verbosity": None}

    @staticmethod
    def _get_first_candidate(*, payload: dict[str, Any]) -> dict[str, Any] | None:
        candidates = payload.get("candidates") or []
        if not candidates:
            return None
        candidate = candidates[0]
        return candidate if isinstance(candidate, dict) else None

    @staticmethod
    def _get_candidate_parts(*, candidate: dict[str, Any] | None) -> list[dict[str, Any]]:
        if candidate is None:
            return []
        content = candidate.get("content") or {}
        parts = content.get("parts") or []
        return [part for part in parts if isinstance(part, dict)]

    @staticmethod
    def _normalize_request_payload(*, payload: Any) -> dict[str, Any]:
        if hasattr(payload, "model_dump"):
            return payload.model_dump(mode="json", exclude_none=True)
        if isinstance(payload, dict):
            return payload
        raise ValueError("request_payload must be a dict or model")

    @staticmethod
    def _normalize_payload(*, payload: Any, allow_non_dict: bool = False) -> Any:
        if hasattr(payload, "model_dump"):
            payload = payload.model_dump(mode="json", exclude_none=True)
        if isinstance(payload, dict):
            return payload
        if allow_non_dict:
            return payload
        raise ValueError("payload must be a dict or model")

    @staticmethod
    def _coerce_timestamp(value: Any) -> int | None:
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            try:
                normalized = value.replace("Z", "+00:00")
                return int(datetime.fromisoformat(normalized).timestamp())
            except ValueError:
                return None
        return None

    @staticmethod
    def _json_dumps(value: Any) -> str:
        return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)

    @staticmethod
    def _safe_int(value: Any) -> int:
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _build_error_event(*, payload: dict[str, Any], message: str) -> ErrorStreamingEvent:
        sequence_number = payload.get("sequence_number")
        if not isinstance(sequence_number, int):
            sequence_number = 0
        return ErrorStreamingEvent(
            type="error",
            sequence_number=sequence_number,
            error=ErrorPayload(
                type="invalid_stream_event",
                code=None,
                message=message,
                param=None,
                headers=None,
            ),
        )
