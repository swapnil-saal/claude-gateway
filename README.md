# claude-gateway

A small local server that turns your **Claude Code login** into an API endpoint.
It speaks the **Anthropic Messages API** and the **OpenAI Chat Completions API**, so the
official `anthropic` and `openai` SDKs, and anything built on them, work against it
unchanged. Requests run on your Claude subscription through Claude Code; no API key or
separate billing is needed.

```python
import anthropic, openai
a = anthropic.Anthropic(base_url="http://127.0.0.1:8787", api_key=token)
o = openai.OpenAI(base_url="http://127.0.0.1:8787/v1", api_key=token)
```

Why it exists: personal tools and side projects that want a plain chat API, with the plan
you already pay for. It is **for personal use on your own machine**. It is not a way to
share or resell a subscription, and it must never be exposed to the internet.

## Requirements

- macOS or Linux. Windows can run the server in the foreground only.
- [uv](https://docs.astral.sh/uv/) and Python 3.12+.
- Claude Code logged in on this machine (`claude` then `/login`) with a plan that includes it.
  The gateway uses the Claude Code CLI bundled with the Agent SDK, so a separate `claude`
  install is not required, but the login is.

## Install

```sh
git clone https://github.com/<you>/claude-gateway
cd claude-gateway
uv sync
uv run claude-gateway install
```

`install` registers a login service (launchd on macOS, a systemd user unit on Linux),
starts it, waits until it answers, and prints the token. From then on it starts at login
and restarts if it crashes.

To run it in the foreground instead, for a quick try or while developing:

```sh
uv run claude-gateway
```

## Manage

```sh
uv run claude-gateway status        # service state, health, warm processes, sessions, plan window
uv run claude-gateway logs -f       # follow the log (also -n 200)
uv run claude-gateway stop          # stop until `start`, or until the next login
uv run claude-gateway start
uv run claude-gateway restart       # after pulling an update
uv run claude-gateway token         # the API token clients send
uv run claude-gateway uninstall     # remove the service; keeps ~/.claude-gateway
```

Everything the gateway owns lives in `~/.claude-gateway/`: the `token` file (created on
first run, mode 600), `gateway.log`, and an empty working directory for the Claude Code
processes. Delete the token file to rotate it, then restart.

Settings are environment variables, read when the server starts. `install` captures the
ones set in the shell that runs it, so `CLAUDE_GATEWAY_PORT=9000 uv run claude-gateway install`
sticks.

| Variable | Default | Meaning |
|---|---|---|
| `CLAUDE_GATEWAY_HOST` | `127.0.0.1` | Bind address. Keep it local, or use a Tailscale IP to reach it from your own devices. |
| `CLAUDE_GATEWAY_PORT` | `8787` | |
| `CLAUDE_GATEWAY_TOKEN` | generated | Bearer token clients must send. |
| `CLAUDE_GATEWAY_DEFAULT_MODEL` | `sonnet` | Model when a request names one Claude Code does not know. |
| `CLAUDE_GATEWAY_SPARES` | `2` | Warm Claude Code processes kept ready for plain requests. |
| `CLAUDE_GATEWAY_MAX_IDLE` | `4` | Warm processes across all pools. |
| `CLAUDE_GATEWAY_CONCURRENCY` | `3` | Requests served at once; others queue for up to 120 s. |
| `CLAUDE_GATEWAY_MAX_POOLS` | `6` | Distinct system-prompt/tool combinations kept warm. |
| `CLAUDE_GATEWAY_POOL_TTL` | `300` | Seconds an unused pool keeps its spare. |
| `CLAUDE_GATEWAY_SESSION_TTL` | `900` | Seconds an idle session keeps its process. |
| `CLAUDE_GATEWAY_HOME` | `~/.claude-gateway` | Where the token, log and working directory live. |

## Use

Send the token as `x-api-key` or `Authorization: Bearer`, whichever your client does.
Model ids are the Anthropic ones (`claude-sonnet-5-5`, `claude-haiku-4-5`, `claude-opus-5-5`,
`claude-fable-5-1`) or the short aliases `sonnet`, `haiku`, `opus`, `fable`. `GET /v1/models` lists them.

### Anthropic SDK

```python
import anthropic
client = anthropic.Anthropic(base_url="http://127.0.0.1:8787", api_key=token)

msg = client.messages.create(model="claude-sonnet-5-5", max_tokens=1024,
                             messages=[{"role": "user", "content": "Hello"}])
print(msg.content[0].text)         # a real anthropic.types.Message

with client.messages.stream(model="haiku", max_tokens=512, messages=[...]) as s:
    for text in s.text_stream:
        print(text, end="")
```

TypeScript: `new Anthropic({ baseURL: "http://127.0.0.1:8787", apiKey: token })`.

### OpenAI SDK and OpenAI-compatible apps

```python
import openai
client = openai.OpenAI(base_url="http://127.0.0.1:8787/v1", api_key=token)
r = client.chat.completions.create(model="claude-haiku-4-5",
                                   messages=[{"role": "user", "content": "Hello"}])
print(r.choices[0].message.content)
```

Apps that take an "OpenAI base URL" want `http://127.0.0.1:8787/v1`. The routes are also
served without the `/v1` prefix, so `http://127.0.0.1:8787` works as well.

Supported: `system` and `developer` messages, streaming with `[DONE]` and
`stream_options.include_usage`, `tools` with `tool_choice` and `tool` messages, streamed
tool-call argument deltas, `image_url` as a data URL or http URL, `stop`,
`max_completion_tokens`, `response_format` of `json_object` or `json_schema` (the model is
instructed and markdown fences are stripped), `reasoning_effort` (mapped to thinking,
returned as `reasoning_content`), the legacy `functions` API, and sessions via `x-session-id`.
Errors use OpenAI's `{"error": {...}}` shape with codes like `invalid_api_key` and
`model_not_found`. Not supported: `n > 1`, `logprobs`, audio.

### Sessions

Send `x-session-id: <any id>` (`extra_headers=` in either SDK) and the gateway keeps one
Claude Code process for that conversation. Keep sending the full `messages` list as usual;
only the newest message is forwarded, which keeps long chats fast. Sessions end after
`CLAUDE_GATEWAY_SESSION_TTL` idle seconds or with `DELETE /v1/sessions/{id}`.

### Agent runs

`POST /v1/agent/run` runs a whole Claude Code agent loop with built-in tools, MCP servers,
a working directory and JSON-schema output.

```sh
curl -s localhost:8787/v1/agent/run -H "x-api-key: $(uv run claude-gateway token)" -d '{
  "prompt": "Open https://example.com and summarise it",
  "tools": ["WebFetch"], "model": "sonnet",
  "output_schema": {"type":"object","properties":{"summary":{"type":"string"}}}
}'
```

### Postman

Import `postman/claude-gateway.postman_collection.json`, copy
`postman/local.postman_environment.example.json` to `local.postman_environment.json`,
paste your token into it, import that too and select the **claude-gateway local**
environment. Run `Tools / 1. Tool call` before `2. Tool result`: it saves the `tool_use`
id into `{{toolUseId}}`.

## Compatibility with the Messages API

| Feature | Status |
|---|---|
| `messages`, `system` (string or blocks), multi-turn history | ✅ earlier turns are written out as a transcript for a fresh process |
| `stream` SSE (`message_start` … `message_stop`, `ping`) | ✅ the model's own events, passed through |
| `tools`, `tool_use` / `tool_result`, `tool_choice` (auto/any/tool/none) | ✅ via deferred MCP tools; the model never runs them |
| images, documents | ✅ |
| `thinking` (enabled/adaptive) | ✅ |
| `stop_sequences` | ✅ applied by the gateway |
| assistant prefill | ⚠️ approximated: the model is asked to start with the text, which is then stripped |
| errors → SDK exception types (400/401/404/429/529) | ✅ |
| `/v1/models`, `/v1/messages/count_tokens` | ✅ (the token count is an estimate) |
| `max_tokens`, `temperature`, `top_p`, `top_k` | ⚠️ accepted but not enforced |
| server tools (web search, code execution), `output_config`, batches, files API | ❌ use `/v1/agent/run` instead |
| `usage.input_tokens` | reported by Claude Code; includes its small tool-definition overhead |

Plan limits are shared with your interactive Claude Code. `claude-gateway status` and
`GET /usage` show the 5-hour and weekly windows, and a request rejected by the plan returns
a 429 `rate_limit_error`.

## How it works

Each request is answered by a Claude Code process driven through the Claude Agent SDK.
Starting one costs 1-3 s, so the gateway keeps processes connected ahead of time:

- **Spares.** The default pool (no system prompt, no tools) keeps `CLAUDE_GATEWAY_SPARES`
  processes ready. A stateless request takes one, uses it once and discards it, and a
  replacement starts in the background. Nothing carries over between requests.
- **Pools** are keyed by system prompt, tools and thinking, which are fixed when a process
  starts. Any other pool starts empty, so its first request is cold (about 1 s extra).
  From its second use it keeps one spare, dropped after `CLAUDE_GATEWAY_POOL_TTL` idle
  seconds. At most `CLAUDE_GATEWAY_MAX_IDLE` spares exist across all pools, so a burst of
  one-off prompts cannot leave processes running. The model is switched per request at no cost.
- **Sessions** keep their process between turns.

On an M-series Mac, warm time to first token with Haiku is about 0.6-0.9 s, which is
essentially the model's own latency.

Claude Code always tells the model it is an agent and describes the hosting process
(working directory, platform, date), even with an empty system prompt. Left alone, Haiku
answers "hi" with a long introduction of itself and that directory, so the gateway puts a
short preamble ahead of every chat system prompt telling the model to ignore that context
and answer as it would through the API.

Gateway processes ignore your user settings, hooks and `CLAUDE.md` files, and do not write
transcripts to `~/.claude/projects`, so they never show up in tools that watch it.

## Security

- The gateway is a door to your subscription. Keep it on `127.0.0.1` or a private network
  such as Tailscale. Never put it behind a public port or a tunnel.
- The token is the only authentication. Treat it like a password; rotate it by deleting
  `~/.claude-gateway/token` and restarting.
- Client tools passed in requests are never executed by the gateway; the model's tool calls
  are returned to the caller, exactly as with the real API. `/v1/agent/run` is different:
  it can run Claude Code's built-in tools on your machine, so only expose it to callers you trust.

## Development

```sh
uv sync
uv run claude-gateway                                   # foreground, logs to the terminal
uv run python tests/e2e.py http://127.0.0.1:8787        # 24 checks through the anthropic SDK
uv run python tests/e2e_openai.py http://127.0.0.1:8787 # Chat Completions wire format (+ openai SDK when installed)
```

Layout: `gateway/server.py` holds the routes, `anthropic_format.py` the Messages API
translation, `openai_format.py` the Chat Completions translation, `pool.py` the warm
process pools, `config.py` the settings and `cli.py` the command.

Issues and pull requests are welcome. Keep changes small and run both test files against
a running gateway before opening a PR.

## License

[MIT](LICENSE). Not affiliated with Anthropic. Use it in line with Anthropic's terms for
your plan.
