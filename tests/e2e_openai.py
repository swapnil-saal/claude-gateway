"""End-to-end check of the OpenAI-compatible endpoint against a running gateway.

    uv run python tests/e2e_openai.py [base_url]

Uses the `openai` SDK when it is installed, otherwise plain HTTP in the same wire format.
"""
import json
import sys
import time
from pathlib import Path

import httpx

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8787"
TOKEN = (Path.home() / ".claude-gateway" / "token").read_text().strip()
MODEL = "claude-haiku-4-5"
failures = []
http = httpx.Client(base_url=BASE, headers={"authorization": f"Bearer {TOKEN}"}, timeout=120)


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not cond:
        failures.append(name)


def post(**body):
    r = http.post("/v1/chat/completions", json=body)
    return r.status_code, r.json()


def stream(**body):
    with http.stream("POST", "/v1/chat/completions", json={**body, "stream": True}) as r:
        for line in r.iter_lines():
            if line.startswith("data: "):
                yield line[6:]


TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]

# 1. plain completion
t = time.time()
code, c = post(model=MODEL, messages=[{"role": "user", "content": "Reply with exactly: pong"}])
msg = c["choices"][0]["message"]
check("completion object", code == 200 and c["object"] == "chat.completion" and c["id"].startswith("chatcmpl-"))
check("content", "pong" in (msg["content"] or "").lower(), msg["content"])
check("finish_reason stop", c["choices"][0]["finish_reason"] == "stop")
check("usage", c["usage"]["prompt_tokens"] > 0 and c["usage"]["total_tokens"] > c["usage"]["completion_tokens"] > 0,
      json.dumps(c["usage"]))
print(f"     total {time.time() - t:.2f}s")

# 2. system + developer messages, max_completion_tokens, stop
code, c = post(model=MODEL, max_completion_tokens=50, messages=[
    {"role": "system", "content": "Always answer in French, one word."},
    {"role": "user", "content": "What colour is the sky?"}])
check("system message honoured", any(w in c["choices"][0]["message"]["content"].lower() for w in ("bleu", "azur")),
      c["choices"][0]["message"]["content"])
code, c = post(model=MODEL, stop=["STOP"], messages=[{"role": "user", "content": "Write exactly: alpha beta STOP gamma"}])
check("stop honoured", "gamma" not in c["choices"][0]["message"]["content"] and c["choices"][0]["finish_reason"] == "stop",
      repr(c["choices"][0]["message"]["content"]))

# 3. streaming chunks
t, first, parts, finish, last = time.time(), None, [], None, None
for data in stream(model=MODEL, messages=[{"role": "user", "content": "Count from 1 to 5, comma separated."}]):
    if data == "[DONE]":
        last = data
        continue
    ch = json.loads(data)
    first = first or time.time() - t
    for choice in ch["choices"]:
        parts.append(choice["delta"].get("content") or "")
        finish = choice.get("finish_reason") or finish
check("stream chunks", ch["object"] == "chat.completion.chunk" and "1" in "".join(parts) and "5" in "".join(parts),
      "".join(parts))
check("stream finish + [DONE]", finish == "stop" and last == "[DONE]", f"{finish} {last}")
print(f"     warm time-to-first-chunk {first:.2f}s, total {time.time() - t:.2f}s")

# 4. stream_options.include_usage: trailing usage chunk with empty choices
chunks = [json.loads(d) for d in stream(model=MODEL, stream_options={"include_usage": True},
                                        messages=[{"role": "user", "content": "Say hi"}]) if d != "[DONE]"]
check("include_usage chunk", chunks[-1]["choices"] == [] and chunks[-1]["usage"]["completion_tokens"] > 0,
      json.dumps(chunks[-1].get("usage")))

# 5. tool call, then tool result
q = {"role": "user", "content": "What's the weather in Paris right now?"}
code, c = post(model=MODEL, tools=TOOLS, messages=[q])
msg = c["choices"][0]["message"]
calls = msg.get("tool_calls") or []
check("tool_calls", c["choices"][0]["finish_reason"] == "tool_calls" and calls and calls[0]["function"]["name"] == "get_weather"
      and "paris" in calls[0]["function"]["arguments"].lower(), json.dumps(calls)[:120])
if calls:
    code, c2 = post(model=MODEL, tools=TOOLS, messages=[
        q, msg, {"role": "tool", "tool_call_id": calls[0]["id"], "content": "31°C, sunny"}])
    check("tool result answered", "31" in c2["choices"][0]["message"]["content"], c2["choices"][0]["message"]["content"][:80])

# 6. streamed tool call: arguments arrive as deltas
name, args, finish = None, "", None
for data in stream(model=MODEL, tools=TOOLS, tool_choice={"type": "function", "function": {"name": "get_weather"}},
                   messages=[{"role": "user", "content": "Tell me about Tokyo."}]):
    if data == "[DONE]":
        break
    for choice in json.loads(data)["choices"]:
        for tc in choice["delta"].get("tool_calls") or []:
            name = tc.get("function", {}).get("name") or name
            args += tc.get("function", {}).get("arguments") or ""
        finish = choice.get("finish_reason") or finish
check("streamed tool call", name == "get_weather" and "tokyo" in json.loads(args)["city"].lower() and finish == "tool_calls",
      f"{name} {args} {finish}")

# 7. response_format json_object
code, c = post(model=MODEL, response_format={"type": "json_object"},
               messages=[{"role": "user", "content": "Give the capital of France as JSON with key 'capital'."}])
try:
    parsed = json.loads(c["choices"][0]["message"]["content"])
except json.JSONDecodeError:
    parsed = {}
check("response_format json_object", parsed.get("capital", "").lower() == "paris", c["choices"][0]["message"]["content"][:80])

# 8. image as data URL
import base64, struct, zlib
size = 16
raw = b"".join(b"\x00" + b"\xff\x00\x00" * size for _ in range(size))
chunk = lambda t_, d: struct.pack(">I", len(d)) + t_ + d + struct.pack(">I", zlib.crc32(t_ + d) & 0xFFFFFFFF)
png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)) + \
    chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
code, c = post(model=MODEL, messages=[{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}},
    {"type": "text", "text": "What colour is this image? One word."}]}])
check("image_url input", "red" in c["choices"][0]["message"]["content"].lower(), c["choices"][0]["message"]["content"])

# 9. errors in OpenAI shape
r = httpx.post(f"{BASE}/v1/chat/completions", headers={"authorization": "Bearer nope"},
               json={"model": MODEL, "messages": [{"role": "user", "content": "x"}]})
check("bad key -> 401 invalid_api_key", r.status_code == 401 and r.json()["error"]["code"] == "invalid_api_key", r.text)
code, c = post(model=MODEL, messages=[])
check("empty messages -> 400", code == 400 and c["error"]["type"] == "invalid_request_error", json.dumps(c))
code, c = post(model="no-such-model-xyz", messages=[{"role": "user", "content": "x"}])
check("unknown model -> 404", code == 404 and c["error"]["code"] == "model_not_found", json.dumps(c)[:100])

# 10. models list
r = http.get("/v1/models").json()
check("models list", r["object"] == "list" and all(m["object"] == "model" for m in r["data"]),
      ", ".join(m["id"] for m in r["data"]))

# 11. the openai SDK itself, when available
try:
    import openai
except ImportError:
    print("     (openai SDK not installed; wire-format checks only)")
else:
    client = openai.OpenAI(base_url=f"{BASE}/v1", api_key=TOKEN, max_retries=0)
    c = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": "Reply with exactly: pong"}])
    check("openai SDK create", "pong" in c.choices[0].message.content.lower(), c.choices[0].message.content)
    got = "".join(ch.choices[0].delta.content or "" for ch in client.chat.completions.create(
        model=MODEL, stream=True, messages=[{"role": "user", "content": "Say hi"}]) if ch.choices)
    check("openai SDK stream", bool(got), got)

print()
print("ALL PASSED" if not failures else f"FAILED: {', '.join(failures)}")
sys.exit(1 if failures else 0)
