#!/usr/bin/env zsh
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
MIN_UV_VERSION="0.9.17"

version_ge() {
  local current="$1"
  local required="$2"
  local current_major=0
  local current_minor=0
  local current_patch=0
  local required_major=0
  local required_minor=0
  local required_patch=0

  IFS=. read -r current_major current_minor current_patch <<< "$current"
  IFS=. read -r required_major required_minor required_patch <<< "$required"

  current_patch="${current_patch:-0}"
  required_patch="${required_patch:-0}"

  if (( current_major != required_major )); then
    (( current_major > required_major ))
    return
  fi

  if (( current_minor != required_minor )); then
    (( current_minor > required_minor ))
    return
  fi

  (( current_patch >= required_patch ))
}

require_uv() {
  if ! command -v uv >/dev/null 2>&1; then
    echo "エラー: uv が見つかりません。uv ${MIN_UV_VERSION} 以上をインストールしてください。" >&2
    exit 1
  fi

  local current_version
  current_version=$(uv --version 2>/dev/null | awk 'NR==1 {print $2}')

  if [[ -z "$current_version" ]]; then
    echo "エラー: uv のバージョンを判定できませんでした。uv ${MIN_UV_VERSION} 以上をインストールしてください。" >&2
    exit 1
  fi

  if ! version_ge "$current_version" "$MIN_UV_VERSION"; then
    echo "エラー: uv ${MIN_UV_VERSION} 以上が必要です。現在のバージョン: ${current_version}" >&2
    echo "理由: pyproject.toml の exclude-newer = \"7 days\" は古い uv では解釈できません。" >&2
    echo "対処: uv を更新してから再実行してください。" >&2
    exit 1
  fi
}

require_uv

export UV_CACHE_DIR="${SCRIPT_DIR}/.uv_cache"
mkdir -p "${UV_CACHE_DIR}"

# main
uv add google-genai openresponses-impl-core

# dev dependencies
uv add --dev build pytest pytest-asyncio pytest-cov ruff twine

# 追加後の lockfile を監査し、脆弱性や adverse status があれば失敗させる
uv audit --locked --preview-features audit

echo "すべてのライブラリの追加と監査が完了しました"
