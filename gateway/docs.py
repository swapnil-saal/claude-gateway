"""OpenAPI text for the interactive docs at /docs (Swagger UI) and /redoc.

The routes read raw JSON bodies, so request shapes are documented here as examples rather
than generated from models. The Anthropic and OpenAI references define the full formats.
"""

DESCRIPTION = """
A local server that answers **Anthropic Messages API** and **OpenAI Chat Completions API**
requests with your own Claude Code login.

**Authentication:** every route except `/health` needs the gateway token, sent either as
`x-api-key: <token>` (Anthropic style) or `Authorization: Bearer <token>` (OpenAI style).
Get it with `claude-gateway token`. Click **Authorize** above to use it here.

**Formats:** request and response bodies follow the official references, so SDKs work unchanged.

- Messages API: https://platform.claude.com/docs/en/api/messages
- Chat Completions API: https://platform.openai.com/docs/api-reference/chat

**Errors:** Anthropic routes return `{"type": "error", "error": {"type", "message"}}`; the chat
completions route returns `{"error": {"message", "type", "param", "code"}}`. Status codes are
400, 401, 404, 429 (plan limit), 500 and 529 (gateway busy).
"""

TAGS = [
    {"name": "Anthropic", "description": "Messages API. Use the `anthropic` SDK with `base_url` set to this server."},
    {"name": "OpenAI", "description": "Chat Completions API. Use the `openai` SDK with `base_url` set to this "
                                      "server plus `/v1`; the route also works without the prefix."},
    {"name": "Agent", "description": "A whole Claude Code agent run with built-in tools and MCP servers."},
    {"name": "Gateway", "description": "Status and housekeeping."},
]


def _body(example: dict, description: str = "") -> dict:
    return {"requestBody": {"required": True, "description": description,
                            "content": {"application/json": {"example": example}}}}


def _json(description: str, example=None) -> dict:
    return {"description": description, "content": {"application/json": {"example": example}} if example else {}}


MESSAGES = dict(
    tags=["Anthropic"], summary="Create a message",
    description="""
The Messages API. `model`, `max_tokens` and `messages` are required. Supports `system`, multi-turn
history, `tools` with `tool_use`/`tool_result` and `tool_choice`, images and documents, `thinking`,
`stop_sequences`, and `stream: true` for server-sent events (`message_start` … `message_stop`).

Send `x-session-id: <any id>` to keep one Claude Code process for a conversation; keep sending the
full `messages` list and only the newest message is forwarded. Sessions end after
`CLAUDE_GATEWAY_SESSION_TTL` idle seconds or with `DELETE /v1/sessions/{id}`.

Model ids: `claude-sonnet-5-5`, `claude-haiku-4-5`, `claude-opus-5-5`, `claude-fable-5-1`, or the
aliases `sonnet`, `haiku`, `opus`, `fable`.
""",
    openapi_extra=_body({"model": "claude-haiku-4-5", "max_tokens": 1024,
                         "system": "You are terse.",
                         "messages": [{"role": "user", "content": "Hello"}],
                         "stream": False}),
    responses={200: _json("A Message object, or `text/event-stream` when `stream` is true.", {
        "id": "msg_01…", "type": "message", "role": "assistant", "model": "claude-haiku-4-5-20251001",
        "content": [{"type": "text", "text": "Hi! How can I help you today?"}],
        "stop_reason": "end_turn", "stop_sequence": None,
        "usage": {"input_tokens": 12, "output_tokens": 10}})},
)

CHAT = dict(
    tags=["OpenAI"], summary="Create a chat completion",
    description="""
The Chat Completions API, translated onto the Messages path. Supports `system`/`developer` messages,
`stream: true` (chunks ending with `data: [DONE]`, `stream_options.include_usage`), `tools` with
`tool_choice` and `tool` messages, `image_url` (data or http URLs), `stop`, `max_completion_tokens`,
`response_format` `json_object`/`json_schema`, `reasoning_effort` (mapped to thinking, returned as
`reasoning_content`), the legacy `functions` API and `x-session-id` sessions.

Not supported: `n > 1`, `logprobs`, audio. Also served at `/chat/completions`.
""",
    openapi_extra=_body({"model": "claude-haiku-4-5",
                         "messages": [{"role": "system", "content": "You are terse."},
                                      {"role": "user", "content": "Hello"}],
                         "stream": False}),
    responses={200: _json("A chat.completion object, or `text/event-stream` when `stream` is true.", {
        "id": "chatcmpl-01…", "object": "chat.completion", "created": 1791200000,
        "model": "claude-haiku-4-5-20251001",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi! How can I help you today?"},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 10, "total_tokens": 22}})},
)

COUNT_TOKENS = dict(
    tags=["Anthropic"], summary="Estimate input tokens",
    description="Claude Code has no token counter, so this is an estimate (about 4 characters per token).",
    openapi_extra=_body({"model": "claude-haiku-4-5", "messages": [{"role": "user", "content": "Hello"}]}),
    responses={200: _json("Estimated count.", {"input_tokens": 3})},
)

END_SESSION = dict(
    tags=["Anthropic"], summary="End a session",
    description="Stops the Claude Code process kept for an `x-session-id`. Idle sessions end on their own.",
    responses={200: _json("Whether a session existed.", {"id": "my-chat", "deleted": True})},
)

MODELS = dict(
    tags=["Gateway"], summary="List models",
    description="Model objects in both Anthropic and OpenAI shape, so both SDKs' `models.list()` work. "
                "Any id Claude Code accepts also works in requests. Also served at `/models`.",
)

MODEL = dict(tags=["Gateway"], summary="Get a model", description="Also served at `/models/{model_id}`.")

USAGE = dict(
    tags=["Gateway"], summary="Plan limits and process pools",
    description="Rate-limit windows reported by Claude Code (5-hour and weekly), request slots in use, "
                "warm process pools and live sessions. `claude-gateway status` prints the same.",
)

HEALTH = dict(tags=["Gateway"], summary="Liveness", description="No authentication. `warm_spares` is the "
              "number of Claude Code processes ready to answer immediately.")

AGENT_RUN = dict(
    tags=["Agent"], summary="Run a Claude Code agent",
    description="""
Runs a whole agent loop and returns when it finishes. Unlike the chat routes, the tools here are
Claude Code's own and **run on this machine**, so only expose this to callers you trust.

Fields: `prompt` (string or content blocks, required), `model`, `system` (replaces Claude Code's
prompt) or `append_system` (added to it), `tools` (built-in names such as `Read`, `Bash`, `WebFetch`),
`mcp_servers` (`{name: stdio or http config}`), `cwd`, `output_schema` (JSON schema for
`structured_output`), `max_turns`, `permission_mode`, `session_id` (keeps the agent's context for
follow-up runs).
""",
    openapi_extra=_body({"prompt": "Open https://example.com and summarise it in one sentence.",
                         "model": "sonnet", "tools": ["WebFetch"],
                         "output_schema": {"type": "object", "properties": {"summary": {"type": "string"}}}}),
    responses={200: _json("The run's result, structured output when a schema was given, and every step.", {
        "id": "run_…", "is_error": False, "subtype": "success", "result": "…",
        "structured_output": {"summary": "Example Domain is a placeholder site."},
        "num_turns": 3, "duration_ms": 2689, "usage": {}, "steps": [
            {"type": "tool_use", "name": "WebFetch", "input": {"url": "https://example.com"}},
            {"type": "tool_result", "is_error": False, "content": "…"},
            {"type": "text", "text": "…"}], "session_id": None})},
)
