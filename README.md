# openresponses-impl-client-google

Python client library for Google Gemini implementing the OpenResponses interface.

## Overview

This package exposes a `BaseResponsesClient`-compatible Gemini client:

- Non-streaming: returns `ResponseResource`
- Streaming: returns `AsyncIterator[ResponseStreamingEvent]`
- Request model: `CreateResponseBody`

It uses the Google GenAI SDK (`google-genai`) underneath, but normalizes requests and responses to OpenResponses models from `openresponses-impl-core`.

## Installation

```bash
uv add openresponses-impl-client-google
```

Dependencies:

- Python `>=3.12`
- `google-genai>=1.72.0`
- `openresponses-impl-core>=0.1.0`

## Basic Usage

```python
from openresponses_impl_core.models.openresponses_models import CreateResponseBody
from openresponses_impl_client_google.client.gemini_responses_client import GeminiResponsesClient


client = GeminiResponsesClient(
    model="gemini-3-flash-preview",
    google_api_key="YOUR_API_KEY",  # optional if GOOGLE_API_KEY / GEMINI_API_KEY is set
)

payload = CreateResponseBody(
    input="Hello",
    stream=False,
)

response = await client.create_response(payload=payload)
print(response.output)
```

Streaming:

```python
payload = CreateResponseBody(
    input="Explain recursion briefly.",
    stream=True,
)

event_stream = await client.create_response(payload=payload)
async for event in event_stream:
    print(event.type)
```

## Special Handling

### System and developer instructions are merged

Gemini does not consume OpenResponses `instructions`, `system` messages, and `developer` messages in the same shape as OpenAI Responses API.

This client merges:

- `payload.instructions`
- `input` items with `role="system"`
- `input` items with `role="developer"`

into a single Gemini `system_instruction` string.

### Tool follow-up requires the same client instance

Gemini tool follow-up uses `function_response`, which requires the original function name.  
OpenResponses `function_call_output` only carries `call_id`, so this client keeps an in-memory mapping:

- `call_id -> function name`

Implications:

- Tool follow-up must happen on the same `GeminiResponsesClient` instance.
- Stateless replay from `previous_response_id` is not implemented.
- If a `function_call_output` arrives for an unknown `call_id`, the client raises `ValueError`.

### Media input is normalized best-effort

OpenResponses `input_image`, `input_file`, and `input_video` are automatically converted to Gemini API format as follows:

#### 1. Data URI format (URIs starting with `data:`)
```python
# Example: data:image/png;base64,iVBORw0KGgoAAAANSUhEUg...
Message(
    role="user",
    content=[
        InputImage(image_url="data:image/png;base64,iVBORw0KGgoAAAANSUhEUg...")
    ]
)
```
- Base64-encoded data is decoded and sent as byte array
- Uses Gemini API's `types.Part.from_bytes(data=..., mime_type=...)`
- **Use case**: Embedding small images/videos directly in requests

#### 2. URI format (GCS, YouTube, HTTPS, etc.)
```python
# Example: gs://bucket/video.mp4, https://example.com/image.jpg
Message(
    role="user",
    content=[
        InputVideo(video_url="gs://my-bucket/video.mp4")
    ]
)
```
- URIs are sent as-is as references
- Uses Gemini API's `types.Part.from_uri(file_uri=..., mime_type=...)`
- **Use case**: Referencing files on GCS, YouTube videos, external URLs

#### 3. Automatic MIME type inference
- MIME type is automatically inferred from URI or filename extension (e.g., `.mp4` → `video/mp4`)
- Falls back to `application/octet-stream` if inference fails
- Warning log is emitted on fallback

#### 4. Files uploaded via Files API
```python
# Pre-upload using Google GenAI SDK
video_file = genai_client.files.upload(file="path/to/video.mp4")
# Pass directly to input
payload = CreateResponseBody(input=[video_file, ...])
```
- Uploaded `File` objects can be included directly in `input`
- Gemini API handles them appropriately internally

#### Recommended method for video input (Files API)

For long or large video files, we recommend pre-uploading via Files API before using this library:

```python
from google import genai
from openresponses_impl_core.models.openresponses_models import CreateResponseBody, Message
from openresponses_impl_client_google.client.gemini_responses_client import GeminiResponsesClient
import time

# 1. Upload video using Google GenAI SDK
genai_client = genai.Client()
video_file = genai_client.files.upload(file="path/to/video.mp4")

# 2. Wait for processing completion (videos may require processing time)
while True:
    video_file = genai_client.files.get(name=video_file.name)
    if video_file.state != "PROCESSING":
        break
    time.sleep(2)

# 3. Analyze using OpenResponses client
responses_client = GeminiResponsesClient(
    model="gemini-3-flash-preview",
    google_api_key="YOUR_API_KEY",
)

payload = CreateResponseBody(
    input=[
        video_file,  # Pass uploaded File object directly
        Message(role="user", content="Summarize this video and list key points in bullet format.")
    ],
    stream=False,
)

response = await responses_client.create_response(payload=payload)
print(response.output)

# 4. Cleanup (optional)
genai_client.files.delete(name=video_file.name)
```

**Notes:**
- File objects uploaded via Files API can be included directly in the `input` array
- Wait for video processing to complete (`PROCESSING` → `ACTIVE`) before use
- For small videos, you can also use `data:` URIs or GCS URIs

### Reasoning is mapped approximately

OpenResponses reasoning settings do not map 1:1 to Gemini.

Current behavior:

- `reasoning.effort="none"` -> `thinking_budget=0`
- `reasoning.effort="low" | "medium" | "high"` -> Gemini `thinking_level`
- `reasoning.effort="xhigh"` -> mapped to `HIGH` with warning
- `reasoning.summary` -> ignored with warning

### Response fields are partially synthesized

Gemini does not return an object identical to `ResponseResource`, so some fields are reconstructed from:

- request payload
- Gemini response metadata
- client-generated fallback IDs and timestamps

Examples:

- `id` falls back to a synthetic response ID if Gemini does not return `response_id`
- `tools`, `tool_choice`, `text`, `service_tier`, and similar fields are echoed from the effective request
- Gemini-specific metadata is stored under `metadata["gemini_*"]`

## OpenResponses Compatibility Notes

This client is intentionally best-effort. It keeps the OpenResponses public interface, but not every field can be represented natively by Gemini.

### Supported well

- plain text input
- user / assistant message history
- function tools
- Gemini built-in tools expressed as OpenAI-style flat tool objects
- non-streaming responses
- basic streaming text responses
- JSON-schema-style structured output via Gemini `response_json_schema`

### Supported with translation

- `instructions`, `system`, `developer` -> merged `system_instruction`
- `function_call` -> Gemini `function_call`
- `function_call_output` -> Gemini `function_response`
- reasoning text / thought parts -> OpenResponses `reasoning`
- usage fields -> mapped from Gemini `usage_metadata`

### Warning-and-ignore fields

These fields are preserved in the normalized `ResponseResource` when possible, but are not sent to Gemini as functional request controls:

- `previous_response_id`
- `store`
- `background`
- `parallel_tool_calls`
- `max_tool_calls`
- `truncation`
- `include`
- `safety_identifier`
- `prompt_cache_key`

### Unsupported or partially supported behavior

- `previous_response_id`
  - Gemini request execution ignores it.
  - The field is kept in normalized responses for interface compatibility only.

- generic tools
  - Gemini built-in tools are converted dynamically when `tool.type` matches a supported `google.genai.types.Tool` field with object-style configuration.
  - Built-in tool config follows the OpenAI-style flat request shape, for example:
    - `{"type": "google_maps", "enable_widget": true}`
    - `{"type": "file_search", "file_search_store_names": ["fileSearchStores/STORE_ID"], "top_k": 5}`
    - `{"type": "code_execution"}`
  - `description` is preserved in the echoed request, but is not sent to Gemini as executable tool config.
  - Unknown generic tool types are ignored with warning.
  - Known Gemini built-in tool types with invalid config fail fast with `ValueError`.
  - Built-in outputs are still normalized best-effort; dedicated loops such as computer-use action roundtrips are not yet mapped to OpenResponses-specific output items.

- item types without Gemini equivalents
  - `item_reference` is ignored with warning.
  - OpenResponses input `reasoning` items are ignored with warning.

- exact tool-choice fidelity
  - The client translates tool choice to Gemini function-calling config best-effort.
  - OpenResponses/Core model serialization may already be lossy for some `tool_choice` shapes before the Gemini client sees them.

## Streaming Semantics

Streaming is normalized to a minimal OpenResponses event set.

Currently emitted event families:

- `response.created`
- `response.output_item.added`
- `response.content_part.added`
- `response.output_text.delta`
- `response.output_text.done`
- `response.content_part.done`
- `response.output_item.done`
- `response.reasoning.delta`
- `response.reasoning.done`
- `response.function_call_arguments.done`
- terminal event:
  - `response.completed`
  - `response.incomplete`
  - `response.failed`
- `error`

Notes:

- Gemini stream chunks are merged cumulatively before event translation.
- Delta calculation is prefix-based best-effort.
- If Gemini emits unexpected chunk shapes, the client may emit an `error` event.

## Status Mapping

Gemini finish state is mapped to OpenResponses status as follows:

- prompt blocked / no candidate -> `failed`
- `MAX_TOKENS` -> `incomplete` with `reason="max_tokens"`
- `STOP` -> `completed`
- response containing function calls -> `completed`
- other Gemini finish reasons -> `incomplete` with the Gemini reason string

## Authentication

You can pass `google_api_key=` directly, or rely on the Google SDK environment variable resolution.

Common environment variables:

- `GOOGLE_API_KEY`
- `GEMINI_API_KEY`

## Logging Behavior

This client uses warnings for non-fatal incompatibilities.  
You should expect warning logs when:

- unsupported OpenResponses fields are provided
- unsupported generic tool types are provided
- unsupported item/content types are provided
- MIME type inference falls back to `application/octet-stream`
- `reasoning.summary` is requested
- `reasoning.effort="xhigh"` is downgraded

## Testing

Run tests with:

```bash
UV_CACHE_DIR="$PWD/.uv_cache" uv run pytest -q
```

This now includes live Gemini API integration tests under `test/integration_test/`.

Live test behavior:

- the test module reads `GOOGLE_API_KEY` or `GEMINI_API_KEY` from the repo-root `.env` file if the variables are not already exported
- if the key is missing or invalid, pytest fails immediately during test collection
- normal `pytest -q` requires network access and may incur Gemini API cost
- the live suite covers non-stream, stream, function-call follow-up, JSON schema output, and reasoning smoke paths

## Summary

Use this package when you want Gemini behind the OpenResponses interface, but keep in mind:

- it is interface-compatible, not wire-compatible
- several OpenResponses controls are emulated or ignored
- tool follow-up depends on client-local in-memory state
- streaming is normalized to a practical subset rather than a perfect Gemini-to-Responses projection
