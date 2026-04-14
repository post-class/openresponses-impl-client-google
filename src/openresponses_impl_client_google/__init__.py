"""OpenResponses implementation for Google Gemini API.

This package provides a client library that implements the OpenResponses
specification for Google's Gemini API.
"""

__version__ = "0.1.0"

from openresponses_impl_client_google.client.gemini_responses_client import (
    GeminiResponsesClient,
)

__all__ = [
    "GeminiResponsesClient",
    "__version__",
]
