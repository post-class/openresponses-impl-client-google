"""Live test for Gemini built-in google_search tool integration."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# src 配下を import path に追加（pytest 実行時の互換確保）
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from openresponses_impl_core.models.openresponses_models import CreateResponseBody

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
            "GOOGLE_API_KEY or GEMINI_API_KEY is required for builtin tool tests. "
            "Set it in the shell environment or in the repo-root .env file."
        )
    return api_key


_API_KEY = _require_api_key()


def _create_client() -> GeminiResponsesClient:
    return GeminiResponsesClient(
        model=_LIVE_MODEL,
        google_api_key=_API_KEY,
    )


def _extract_first_url_citation(response: object) -> object | None:
    if not hasattr(response, "output"):
        return None

    for item_field in response.output:
        item = item_field.root
        if getattr(item, "type", None) != "message":
            continue
        for content_part in getattr(item, "content", []):
            annotations = getattr(content_part, "annotations", None)
            if not annotations:
                continue
            for annotation in annotations:
                annotation_type = getattr(annotation, "type", None)
                annotation_type_value = getattr(annotation_type, "value", annotation_type)
                if annotation_type_value == "url_citation":
                    return annotation
    return None


class TestBuiltinGoogleSearch:
    """Live test for google_search built-in tool configured via tools field."""

    @pytest.mark.asyncio
    async def test_google_search_with_citations(self) -> None:
        client = _create_client()
        payload = CreateResponseBody.model_validate(
            {
                "input": "直近のNVIDIAの主要発表を、重要度順に3つ。出典リンク付きで検索して。",
                "tools": [
                    {
                        "type": "google_search",
                        "description": "Use Google Search to ground answers with citations.",
                    }
                ],
                "stream": False,
            }
        )

        response = await client.create_response(payload=payload)

        assert response.status in {"completed", "incomplete"}
        citation = _extract_first_url_citation(response)
        assert citation is not None
