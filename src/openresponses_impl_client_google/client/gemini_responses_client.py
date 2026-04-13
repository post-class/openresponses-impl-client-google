from __future__ import annotations

import inspect
from collections.abc import AsyncIterator
from typing import Any, Literal, override

from google import AsyncGoogle
from openresponses_impl_core.client.base_responses_client import BaseResponsesClient
from openresponses_impl_core.models.openresponses_models import (
    CreateResponseBody,
    ResponseResource,
)
from openresponses_impl_core.models.response_event_types import ResponseStreamingEvent

from openresponses_impl_client_google.utils.copy_util import CopyUtil
from openresponses_impl_client_google.utils.google_response_model_util import (
    GoogleResponseModelUtil,
)


class GeminiResponsesClient(BaseResponsesClient):
    """Responses client for Google/Azure Google"""

    def __init__(
        self,
        *,
        model: str,
        google_api_key: str | None = None,
    ) -> None:
        """Responses client for Google

        """
        self._model = model
        self._api_key = google_api_key
        self._client = self._create_client()
