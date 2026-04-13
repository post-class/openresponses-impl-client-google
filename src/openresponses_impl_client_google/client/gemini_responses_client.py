from __future__ import annotations


from openresponses_impl_core.client.base_responses_client import BaseResponsesClient



class GeminiResponsesClient(BaseResponsesClient):
    """Responses client for gemini api"""

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

