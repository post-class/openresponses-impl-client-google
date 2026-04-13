"""Live test for uploading a movie file and describing it with Gemini."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest
from google import genai

# src 配下を import path に追加（pytest 実行時の互換確保）
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC = str(_REPO_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from openresponses_impl_core.models.openresponses_models import CreateResponseBody

from openresponses_impl_client_google.client.gemini_responses_client import GeminiResponsesClient

_LIVE_MODEL = "gemini-3-flash-preview"
_MOVIE_FILE_PATH = _REPO_ROOT / "test" / "test_data" / "cute_cat.mp4"


def _load_env_file() -> None:
    env_path = _REPO_ROOT / ".env"
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
            "GOOGLE_API_KEY or GEMINI_API_KEY is required for this live upload test. "
            "Set it in the shell environment or in the repo-root .env file."
        )
    return api_key


def _require_movie_file() -> Path:
    if not _MOVIE_FILE_PATH.exists():
        raise AssertionError(f"Movie test file not found: {_MOVIE_FILE_PATH}")
    return _MOVIE_FILE_PATH


def _extract_first_assistant_message_text(response: object) -> str:
    if not hasattr(response, "output"):
        return ""

    for item_field in response.output:
        item = item_field.root
        if getattr(item, "type", None) != "message":
            continue

        role = getattr(item, "role", None)
        role_value = getattr(role, "value", role)
        if role_value != "assistant":
            continue

        for content_part in getattr(item, "content", []):
            text = getattr(content_part, "text", None)
            if isinstance(text, str) and text.strip():
                return text.strip()

    return ""


def _wait_until_file_active(
    *,
    client: genai.Client,
    uploaded_file: object,
    poll_sec: int = 2,
    timeout_sec: int = 300,
) -> object:
    deadline = time.time() + timeout_sec
    file_name = getattr(uploaded_file, "name", None)
    if not file_name:
        raise AssertionError("Uploaded file does not have a file name.")

    latest_file = uploaded_file
    while True:
        latest_file = client.files.get(name=file_name)
        state = getattr(latest_file, "state", None)
        state_value = getattr(state, "value", state)

        if state_value == "ACTIVE":
            return latest_file
        if state_value == "FAILED":
            raise AssertionError(f"Uploaded file processing failed: name={file_name}")
        if state_value != "PROCESSING":
            raise AssertionError(
                f"Unexpected uploaded file state: state={state_value!r}, name={file_name}"
            )
        if time.time() >= deadline:
            raise TimeoutError(
                f"Timed out waiting for uploaded file to become ACTIVE: name={file_name}"
            )

        time.sleep(poll_sec)


class TestMovieUpload:
    """Live tests for Gemini movie upload flow."""

    @pytest.mark.asyncio
    async def test_upload_movie_via_files_api_and_describe_it(self) -> None:
        api_key = _require_api_key()
        movie_file_path = _require_movie_file()
        uploaded_file_name: str | None = None
        main_error: BaseException | None = None

        try:
            with genai.Client(api_key=api_key) as files_client:
                uploaded_file = files_client.files.upload(file=movie_file_path)
                uploaded_file_name = getattr(uploaded_file, "name", None)
                active_file = _wait_until_file_active(
                    client=files_client,
                    uploaded_file=uploaded_file,
                )

                file_uri = getattr(active_file, "uri", None)
                if not file_uri:
                    raise AssertionError("Uploaded file does not have a uri.")

                responses_client = GeminiResponsesClient(
                    model=_LIVE_MODEL,
                    google_api_key=api_key,
                )
                payload = CreateResponseBody.model_validate(
                    {
                        "input": [
                            {
                                "type": "message",
                                "role": "user",
                                "content": [
                                    {
                                        "type": "input_file",
                                        "file_url": file_uri,
                                        "filename": movie_file_path.name,
                                    },
                                    {
                                        "type": "input_text",
                                        "text": "この動画を説明して",
                                    },
                                ],
                            }
                        ],
                        "stream": False,
                    }
                )

                response = await responses_client.create_response(payload=payload)
                description = _extract_first_assistant_message_text(response)

                print(description)

                assert response.status in {"completed", "incomplete"}
                assert description
        except BaseException as exc:
            main_error = exc
            raise
        finally:
            if uploaded_file_name:
                cleanup_error: Exception | None = None
                try:
                    with genai.Client(api_key=api_key) as files_client:
                        files_client.files.delete(name=uploaded_file_name)
                except Exception as exc:  # pragma: no cover - cleanup only
                    cleanup_error = exc

                if cleanup_error is not None:
                    if main_error is None:
                        raise cleanup_error
                    print(f"Cleanup failed for uploaded file {uploaded_file_name}: {cleanup_error}")
