"""Test Gemini API key validation with a simple hello message.

This test verifies that the API key is valid by sending a simple "hello" message
to the Gemini API and checking that a response is received.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from google.genai import Client

# src 配下を import path に追加（pytest 実行時の互換確保）
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

_REPO_ROOT = Path(_ROOT)
_MODEL_ID = "gemini-3-flash-preview"


def _load_env_file() -> None:
    """Load environment variables from .env file in the repository root.

    This function reads the .env file and sets environment variables
    that are not already set in the shell environment.
    """
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
    """Get API key from environment variables.

    Priority order (as per official recommendation):
    1. GOOGLE_API_KEY
    2. GEMINI_API_KEY

    Returns:
        str: The API key

    Raises:
        AssertionError: If no API key is found

    """
    _load_env_file()
    # GOOGLE_API_KEY が優先（公式推奨）
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise AssertionError(
            "API key is required for this test. "
            "Set GOOGLE_API_KEY or GEMINI_API_KEY in the shell environment "
            "or in the repo-root .env file."
        )
    return api_key


class TestGeminiHello:
    """Test Gemini API key validation with a simple hello message."""

    def test_gemini_hello_message(self) -> None:
        """Send 'hello' message to Gemini and verify response.

        This test:
        1. Loads API key from environment or .env file
        2. Creates a Gemini client with proper resource management
        3. Sends a simple "hello" message
        4. Verifies that a valid text response is received

        This confirms that:
        - The API key is valid
        - The Gemini API is accessible
        - Basic text generation works
        """
        api_key = _require_api_key()

        # ベストプラクティス：with でクライアントを確実にクローズ
        with Client(api_key=api_key) as client:
            response = client.models.generate_content(
                model=_MODEL_ID,
                contents="hello",
            )

        print(response)

        # アサーション：レスポンスが正常に返されることを確認
        assert hasattr(response, "text"), "Response should have text attribute"
        assert response.text, "Response text should not be empty"
        assert isinstance(response.text, str), "Response text should be a string"

        # 追加確認：レスポンスが意味のある長さを持つことを確認
        assert len(response.text.strip()) > 0, "Response text should contain actual content"
