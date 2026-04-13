# 1. Build the package
UV_CACHE_DIR="$PWD/.uv_cache" uv run python -m build

# 2. Test on TestPyPI (recommended)
UV_CACHE_DIR="$PWD/.uv_cache" uv run python -m twine upload --repository testpypi dist/*

# 3. Upload to production PyPI
UV_CACHE_DIR="$PWD/.uv_cache" uv run python -m twine upload dist/*