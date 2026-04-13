from __future__ import annotations

from typing import Any

from openresponses_impl_core.models.openresponses_models import (
    ErrorPayload,
    ErrorStreamingEvent,
    ResponseResource,
)
from openresponses_impl_core.models.response_event_types import ResponseStreamingEvent
from pydantic import TypeAdapter, ValidationError


class GoogleResponseModelUtil:
    """Utility class for converting Google API responses to OpenResponses models.

    This class provides functionality to convert responses and streaming events
    returned from the Google API into OpenResponses standard model format.
    """

    @staticmethod
    def parse_response(*, payload: Any) -> ResponseResource:
        """Convert response payload to ResponseResource model.

        Args:
            payload: Payload to be converted. Accepts ResponseResource instance,
                    Pydantic model, or dictionary.

        Returns:
            ResponseResource: Converted response resource.

        Raises:
            ValueError: When payload is neither a dictionary nor a Pydantic model.
            ValidationError: When payload validation fails.

        """
        if isinstance(payload, ResponseResource):
            return payload
        normalized = GoogleResponseModelUtil._normalize_payload(payload=payload)
        return ResponseResource.model_validate(normalized)

    @staticmethod
    def parse_stream_event(*, payload: Any) -> ResponseStreamingEvent:
        """Convert streaming event payload to ResponseStreamingEvent model.

        Returns an ErrorStreamingEvent containing error information if conversion fails.

        Args:
            payload: Streaming event payload to be converted.

        Returns:
            ResponseStreamingEvent: Converted streaming event.
                                   ErrorStreamingEvent if conversion fails.

        """
        normalized = GoogleResponseModelUtil._normalize_payload(
            payload=payload, allow_non_dict=True
        )
        if not isinstance(normalized, dict):
            message = f"unsupported streaming event payload: {type(payload).__name__}"
            return GoogleResponseModelUtil._build_error_event(payload={}, message=message)

        adapter = TypeAdapter(ResponseStreamingEvent)
        try:
            return adapter.validate_python(normalized)
        except ValidationError as exc:
            message = f"invalid streaming event: {exc}"
            return GoogleResponseModelUtil._build_error_event(payload=normalized, message=message)

    @staticmethod
    def _normalize_payload(*, payload: Any, allow_non_dict: bool = False) -> Any:
        """Normalize the payload.

        Converts Pydantic models to dictionaries using model_dump.

        Args:
            payload: Payload to be normalized.
            allow_non_dict: Whether to allow non-dictionary types. Default is False.

        Returns:
            Any: Normalized payload. Usually a dictionary.

        Raises:
            ValueError: When allow_non_dict is False and payload is neither a dictionary nor a Pydantic model.

        """
        if hasattr(payload, "model_dump"):
            payload = payload.model_dump(mode="json")
        if isinstance(payload, dict):
            return payload
        if allow_non_dict:
            return payload
        raise ValueError("payload must be a dict or model")

    @staticmethod
    def _build_error_event(*, payload: dict[str, Any], message: str) -> ErrorStreamingEvent:
        """Build an error streaming event.

        Args:
            payload: Payload that caused the error. Used to retrieve sequence_number.
            message: Error message.

        Returns:
            ErrorStreamingEvent: Constructed error streaming event.

        """
        sequence_number = payload.get("sequence_number")
        if not isinstance(sequence_number, int):
            sequence_number = 0
        error = ErrorPayload(
            type="invalid_stream_event",
            code=None,
            message=message,
            param=None,
            headers=None,
        )
        return ErrorStreamingEvent(
            type="error",
            sequence_number=sequence_number,
            error=error,
        )

# Made with Bob
