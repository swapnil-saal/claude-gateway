"""End-to-end check against a running gateway, through the official anthropic SDK.

    uv run python tests/e2e.py [base_url]
"""
import base64
import struct
import sys
import time
import zlib
from pathlib import Path

import anthropic
import httpx
from anthropic.types import Message, TextBlock, ToolUseBlock

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8787"
TOKEN = (Path.home() / ".claude-gateway" / "token").read_text().strip()
client = anthropic.Anthropic(base_url=BASE, api_key=TOKEN, max_retries=0)
MODEL = "claude-haiku-4-5-20251001"
failures = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not cond:
        failures.append(name)


def text(msg: Message) -> str:
    return "".join(b.text for b in msg.content if isinstance(b, TextBlock))


def red_png(size=16):
    raw = b"".join(b"\x00" + b"\xff\x00\x00" * size for _ in range(size))
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)) + \
        chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


# 1. plain message, typed response
t = time.time()
m = client.messages.create(model=MODEL, max_tokens=100, messages=[{"role": "user", "content": "Reply with exactly: pong"}])
check("create returns anthropic Message", isinstance(m, Message) and m.type == "message" and m.role == "assistant")
check("text content", "pong" in text(m).lower(), text(m))
check("stop_reason end_turn", m.stop_reason == "end_turn", m.stop_reason)
check("usage + id + model", m.usage.input_tokens > 0 and m.usage.output_tokens > 0 and m.id.startswith("msg_")
      and "haiku" in m.model, f"{m.id} {m.model} {m.usage.input_tokens}/{m.usage.output_tokens}")
print(f"     total {time.time() - t:.2f}s")

# 2. streaming with the SDK helper
t, first = time.time(), None
with client.messages.stream(model=MODEL, max_tokens=200,
                            messages=[{"role": "user", "content": "Count from 1 to 5, comma separated."}]) as s:
    chunks = []
    for piece in s.text_stream:
        first = first or time.time() - t
        chunks.append(piece)
    final = s.get_final_message()
check("stream text_stream", "1" in "".join(chunks) and "5" in "".join(chunks), "".join(chunks))
check("stream final message typed", isinstance(final, Message) and final.stop_reason == "end_turn")
print(f"     warm time-to-first-token {first:.2f}s, total {time.time() - t:.2f}s")

# 3. raw SSE event order
with client.messages.create(model=MODEL, max_tokens=50, stream=True,
                            messages=[{"role": "user", "content": "Say hi"}]) as raw:
    kinds = [e.type for e in raw]
check("raw event order", kinds[0] == "message_start" and kinds[-1] == "message_stop"
      and "content_block_start" in kinds and "message_delta" in kinds, ",".join(dict.fromkeys(kinds)))

# 4. system prompt
m = client.messages.create(model=MODEL, max_tokens=50, system="Always answer in French, one word.",
                           messages=[{"role": "user", "content": "What colour is the sky?"}])
check("system prompt honoured", any(w in text(m).lower() for w in ("bleu", "azur")), text(m))

# 5. stateless multi-turn history
m = client.messages.create(model=MODEL, max_tokens=50, messages=[
    {"role": "user", "content": "My code word is PINEAPPLE-7."},
    {"role": "assistant", "content": "Noted."},
    {"role": "user", "content": "What is my code word? Reply with it only."}])
check("history carried", "PINEAPPLE-7" in text(m), text(m))

# 6. client tools: tool_use then tool_result round trip
tools = [{"name": "get_weather", "description": "Current weather for a city",
          "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}]
q = {"role": "user", "content": "What's the weather in Paris right now?"}
m = client.messages.create(model=MODEL, max_tokens=300, tools=tools, messages=[q])
use = next((b for b in m.content if isinstance(b, ToolUseBlock)), None)
check("tool_use block", use is not None and use.name == "get_weather" and "paris" in str(use.input).lower(),
      f"{m.stop_reason} {use and use.input}")
check("stop_reason tool_use", m.stop_reason == "tool_use")
if use:
    m2 = client.messages.create(model=MODEL, max_tokens=300, tools=tools, messages=[
        q, {"role": "assistant", "content": m.content},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": use.id, "content": "31°C, sunny"}]}])
    check("tool_result answered", "31" in text(m2) and m2.stop_reason == "end_turn", text(m2)[:80])

# 7. tool_choice forcing
m = client.messages.create(model=MODEL, max_tokens=300, tools=tools, tool_choice={"type": "tool", "name": "get_weather"},
                           messages=[{"role": "user", "content": "Tell me about Tokyo."}])
check("tool_choice tool", any(isinstance(b, ToolUseBlock) for b in m.content), m.stop_reason)

# 8. stop_sequences
m = client.messages.create(model=MODEL, max_tokens=100, stop_sequences=["STOP"],
                           messages=[{"role": "user", "content": "Write exactly: alpha beta STOP gamma delta"}])
check("stop_sequence", m.stop_reason == "stop_sequence" and m.stop_sequence == "STOP" and "gamma" not in text(m),
      repr(text(m)))

# 9. image input
img = base64.b64encode(red_png()).decode()
m = client.messages.create(model=MODEL, max_tokens=30, messages=[{"role": "user", "content": [
    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img}},
    {"type": "text", "text": "What colour is this image? One word."}]}])
check("image input", "red" in text(m).lower(), text(m))

# 10. sessions keep context without resending history
sid = f"e2e-{int(time.time())}"
h = {"x-session-id": sid}
client.messages.create(model=MODEL, max_tokens=30, extra_headers=h,
                       messages=[{"role": "user", "content": "Remember the number 9137. Reply OK."}])
t = time.time()
m = client.messages.create(model=MODEL, max_tokens=30, extra_headers=h,
                           messages=[{"role": "user", "content": "Which number did I ask you to remember? Digits only."}])
check("session context", "9137" in text(m), f"{text(m)} in {time.time() - t:.2f}s")

# 11. tools inside a session: native tool_result answers the deferred call
sid2 = sid + "-tools"
h2 = {"x-session-id": sid2}
m = client.messages.create(model=MODEL, max_tokens=300, tools=tools, extra_headers=h2, messages=[q])
use = next((b for b in m.content if isinstance(b, ToolUseBlock)), None)
if use:
    m2 = client.messages.create(model=MODEL, max_tokens=300, tools=tools, extra_headers=h2, messages=[
        q, {"role": "assistant", "content": m.content},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": use.id, "content": "12°C, rain"}]}])
    check("session tool round trip", "12" in text(m2), text(m2)[:80])
else:
    check("session tool round trip", False, "no tool_use")

# 12. thinking
m = client.messages.create(model=MODEL, max_tokens=2000, thinking={"type": "enabled", "budget_tokens": 1024},
                           messages=[{"role": "user", "content": "What is 17*23? Answer with the number."}])
check("thinking block + answer", m.content[0].type == "thinking" and "391" in text(m),
      f"{[b.type for b in m.content]} {text(m)}")

# 13. errors map to SDK exception types
for name, call, exc in [
    ("bad key -> AuthenticationError",
     lambda: anthropic.Anthropic(base_url=BASE, api_key="nope", max_retries=0).messages.create(
         model=MODEL, max_tokens=5, messages=[{"role": "user", "content": "x"}]), anthropic.AuthenticationError),
    ("first message assistant -> BadRequestError",
     lambda: client.messages.create(model=MODEL, max_tokens=5, messages=[{"role": "assistant", "content": "x"}]),
     anthropic.BadRequestError),
]:
    try:
        call()
        check(name, False, "no error")
    except exc as e:
        check(name, True, type(e).__name__)
    except Exception as e:
        check(name, False, repr(e)[:120])

# the SDK refuses to send a request without max_tokens, so check the gateway's own validation raw
r = httpx.post(f"{BASE}/v1/messages", headers={"x-api-key": TOKEN},
               json={"model": MODEL, "messages": [{"role": "user", "content": "x"}]})
check("missing max_tokens -> 400 invalid_request_error",
      r.status_code == 400 and r.json()["error"]["type"] == "invalid_request_error", r.text[:100])

# 14. models list
ms = client.models.list()
check("models.list", len(ms.data) > 0 and ms.data[0].type == "model", ", ".join(x.id for x in ms.data))

# 15. agent run with a built-in tool and structured output
r = httpx.post(f"{BASE}/v1/agent/run", headers={"x-api-key": TOKEN}, timeout=300, json={
    "prompt": f"Read {Path(__file__).resolve().parent.parent / 'pyproject.toml'} and report the project name.",
    "model": "haiku", "tools": ["Read"],
    "output_schema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}}).json()
check("agent run + structured output", (r.get("structured_output") or {}).get("name") == "claude-gateway",
      f"{r.get('structured_output')} turns={r.get('num_turns')} {r.get('duration_ms')}ms")

for s_id in (sid, sid2):                     # don't leave session processes running for their idle TTL
    httpx.delete(f"{BASE}/v1/sessions/{s_id}", headers={"x-api-key": TOKEN})

print(f"\n{'ALL PASSED' if not failures else f'{len(failures)} FAILED: ' + ', '.join(failures)}")
sys.exit(1 if failures else 0)
