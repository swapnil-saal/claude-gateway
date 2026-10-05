"""OpenAI Chat Completions API <-> Anthropic Messages API.

A chat completion request is rewritten into a Messages request and runs through
the normal path; the shaped Messages events are then turned back into a
`chat.completion` object or `chat.completion.chunk` stream.
"""
import base64
import json
import time

from .anthropic_format import APIError

DEFAULT_MAX_TOKENS = 4096
FINISH = {"end_turn": "stop", "stop_sequence": "stop", "max_tokens": "length", "tool_use": "tool_calls"}
THINKING_BUDGET = {"minimal": 1024, "low": 1024, "medium": 4096, "high": 16000}
ERROR_TYPES = {"api_error": "server_error", "overloaded_error": "server_error"}
ERROR_CODES = {"authentication_error": "invalid_api_key", "not_found_error": "model_not_found",
               "rate_limit_error": "rate_limit_exceeded", "request_too_large": "context_length_exceeded"}


def error_body(e: APIError) -> dict:
    return {"error": {"message": e.message, "type": ERROR_TYPES.get(e.type, e.type),
                      "param": None, "code": ERROR_CODES.get(e.type)}}


# ---------------------------------------------------------------- request
def _parts(content) -> list[dict]:
    """Content as a list of parts; a string is one text part."""
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return list(content)


def _text_of(content) -> str:
    return "".join(p.get("text", "") for p in _parts(content) if p.get("type") == "text")


def _image(url: str) -> dict:
    if url.startswith("data:"):
        head, _, data = url.partition(",")
        media = head[5:].split(";")[0] or "image/png"
        if ";base64" not in head:
            data = base64.b64encode(data.encode()).decode()
        return {"type": "image", "source": {"type": "base64", "media_type": media, "data": data}}
    return {"type": "image", "source": {"type": "url", "url": url}}


def _user_blocks(content) -> list[dict]:
    out = []
    for p in _parts(content):
        kind = p.get("type")
        if kind == "text":
            out.append({"type": "text", "text": p["text"]})
        elif kind == "image_url":
            out.append(_image(p["image_url"]["url"] if isinstance(p["image_url"], dict) else p["image_url"]))
        elif kind == "input_audio":
            raise APIError("invalid_request_error", "messages: audio input is not supported")
    return out


def _assistant_blocks(m: dict) -> list[dict]:
    out = [{"type": "text", "text": t} for t in [_text_of(m.get("content"))] if t]
    for call in m.get("tool_calls") or []:
        fn = call.get("function") or {}
        args = fn.get("arguments") or "{}"
        try:
            inp = json.loads(args) if isinstance(args, str) else args
        except json.JSONDecodeError:
            inp = {"_raw": args}
        out.append({"type": "tool_use", "id": call["id"], "name": fn.get("name", ""), "input": inp})
    if fc := m.get("function_call"):                                   # legacy functions API
        out.append({"type": "tool_use", "id": "call_legacy", "name": fc["name"],
                    "input": json.loads(fc.get("arguments") or "{}")})
    return out


def _tool_choice(choice, legacy_fc):
    choice = choice if choice is not None else legacy_fc
    if choice in (None, "auto"):
        return None
    if choice == "none":
        return {"type": "none"}
    if choice == "required":
        return {"type": "any"}
    if isinstance(choice, dict):
        name = (choice.get("function") or {}).get("name") or choice.get("name")
        if name:
            return {"type": "tool", "name": name}
    raise APIError("invalid_request_error", f"tool_choice: unsupported value {choice!r}")


def _response_format_note(fmt) -> str:
    kind = (fmt or {}).get("type")
    if kind == "json_object":
        return "Respond with a single valid JSON object and nothing else: no prose, no code fences."
    if kind == "json_schema":
        schema = (fmt.get("json_schema") or {}).get("schema") or {}
        return ("Respond with a single JSON value that validates against this JSON schema and nothing else, "
                f"no prose and no code fences:\n{json.dumps(schema)}")
    return ""


def to_messages(body: dict) -> dict:
    """A Messages API request body equivalent to a chat completion request."""
    if not isinstance(body.get("messages"), list) or not body["messages"]:
        raise APIError("invalid_request_error", "messages: at least one message is required")
    if (body.get("n") or 1) != 1:
        raise APIError("invalid_request_error", "n: only 1 is supported")

    system: list[str] = []
    messages: list[dict] = []

    def add(role, blocks):
        if not blocks:
            return
        if messages and messages[-1]["role"] == role:          # the Messages API wants alternating roles
            messages[-1]["content"] += blocks
        else:
            messages.append({"role": role, "content": blocks})

    for m in body["messages"]:
        role = m.get("role")
        if role in ("system", "developer"):
            if t := _text_of(m.get("content")):
                system.append(t)
        elif role == "user":
            add("user", _user_blocks(m.get("content")))
        elif role == "assistant":
            add("assistant", _assistant_blocks(m))
        elif role in ("tool", "function"):
            add("user", [{"type": "tool_result", "tool_use_id": m.get("tool_call_id") or "call_legacy",
                          "content": _text_of(m.get("content")) if not isinstance(m.get("content"), str)
                          else m["content"]}])
        else:
            raise APIError("invalid_request_error", f"messages: unknown role {role!r}")
    if note := _response_format_note(body.get("response_format")):
        system.append(note)
    if messages and messages[0]["role"] != "user":              # a leading assistant turn has no API meaning
        messages.insert(0, {"role": "user", "content": [{"type": "text", "text": "(conversation start)"}]})

    out = {"model": body.get("model") or "", "messages": messages,
           "max_tokens": body.get("max_completion_tokens") or body.get("max_tokens") or DEFAULT_MAX_TOKENS,
           "stream": bool(body.get("stream"))}
    if system:
        out["system"] = "\n\n".join(system)
    for k in ("temperature", "top_p"):
        if body.get(k) is not None:
            out[k] = body[k]
    if stop := body.get("stop"):
        out["stop_sequences"] = [stop] if isinstance(stop, str) else list(stop)

    tools = [t["function"] for t in body.get("tools") or [] if t.get("type") == "function"] + \
            list(body.get("functions") or [])
    if tools:
        out["tools"] = [{"name": t["name"], "description": t.get("description", ""),
                         "input_schema": t.get("parameters") or {"type": "object", "properties": {}}} for t in tools]
    if (tc := _tool_choice(body.get("tool_choice"), body.get("function_call"))) is not None:
        out["tool_choice"] = tc
    if (effort := body.get("reasoning_effort")) and effort != "none":
        out["thinking"] = {"type": "enabled", "budget_tokens": THINKING_BUDGET.get(effort, 4096)}
    return out


# ---------------------------------------------------------------- response
def json_mode(body: dict) -> bool:
    return (body.get("response_format") or {}).get("type") in ("json_object", "json_schema")


def unfence(text: str) -> str:
    """JSON without the markdown fence a model sometimes adds despite instructions."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else ""
        t = t.rsplit("```", 1)[0]
    return t.strip()


def _usage(u: dict) -> dict:
    prompt = (u.get("input_tokens") or 0) + (u.get("cache_read_input_tokens") or 0) + \
             (u.get("cache_creation_input_tokens") or 0)
    out = {"prompt_tokens": prompt, "completion_tokens": u.get("output_tokens") or 0}
    out["total_tokens"] = out["prompt_tokens"] + out["completion_tokens"]
    if cached := u.get("cache_read_input_tokens"):
        out["prompt_tokens_details"] = {"cached_tokens": cached}
    return out


def _call(block: dict, index: int) -> dict:
    return {"id": block["id"], "type": "function", "index": index,
            "function": {"name": block["name"], "arguments": json.dumps(block.get("input") or {})}}


def completion(msg: dict, model: str, json_only: bool = False) -> dict:
    """A chat.completion object from an accumulated Messages API message."""
    text = "".join(b["text"] for b in msg["content"] if b["type"] == "text")
    if json_only:
        text = unfence(text)
    thinking = "".join(b.get("thinking", "") for b in msg["content"] if b["type"] == "thinking")
    calls = [_call(b, i) for i, b in enumerate(b for b in msg["content"] if b["type"] == "tool_use")]
    message = {"role": "assistant", "content": text or None, "refusal": None}
    if calls:
        message["tool_calls"] = calls
    if thinking:
        message["reasoning_content"] = thinking
    return {"id": "chatcmpl-" + msg["id"].removeprefix("msg_"), "object": "chat.completion",
            "created": int(time.time()), "model": msg.get("model") or model, "system_fingerprint": None,
            "choices": [{"index": 0, "message": message, "logprobs": None,
                         "finish_reason": FINISH.get(msg.get("stop_reason") or "", "stop")}],
            "usage": _usage(msg.get("usage") or {})}


class Chunker:
    """Turns shaped Messages API stream events into chat.completion.chunk objects."""

    def __init__(self, model: str, include_usage: bool = False, json_only: bool = False):
        self.model, self.include_usage = model, include_usage
        self.json_only, self.held = json_only, ""     # JSON mode: text is sent whole, fences removed
        self.id, self.created = "chatcmpl-" + "0" * 24, int(time.time())
        self.input_tokens = 0
        self.calls: dict[int, int] = {}        # content block index -> tool call index
        self.kinds: dict[int, str] = {}

    def _chunk(self, delta: dict | None, finish=None, usage=None) -> dict:
        out = {"id": self.id, "object": "chat.completion.chunk", "created": self.created, "model": self.model,
               "system_fingerprint": None,
               "choices": [{"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish}]
               if delta is not None else []}
        if self.include_usage or usage is not None and delta is not None:
            out["usage"] = usage
        return out

    def feed(self, ev: dict) -> list[dict | str]:
        """Chunks to send; the string "[DONE]" ends the stream."""
        kind = ev["type"]
        if kind == "message_start":
            msg = ev["message"]
            self.id = "chatcmpl-" + msg["id"].removeprefix("msg_")
            self.model = msg.get("model") or self.model
            self.input_tokens = (msg.get("usage") or {}).get("input_tokens", 0)
            self.start_usage = msg.get("usage") or {}
            return [self._chunk({"role": "assistant", "content": ""})]
        if kind == "content_block_start":
            block, i = ev["content_block"], ev["index"]
            self.kinds[i] = block["type"]
            if block["type"] == "tool_use":
                self.calls[i] = len(self.calls)
                return [self._chunk({"tool_calls": [{"index": self.calls[i], "id": block["id"], "type": "function",
                                                     "function": {"name": block["name"], "arguments": ""}}]})]
            return []
        if kind == "content_block_delta":
            d, i = ev["delta"], ev["index"]
            if d["type"] == "text_delta" and d["text"]:
                if self.json_only:
                    self.held += d["text"]
                    return []
                return [self._chunk({"content": d["text"]})]
            if d["type"] == "input_json_delta" and d["partial_json"]:
                return [self._chunk({"tool_calls": [{"index": self.calls[i],
                                                     "function": {"arguments": d["partial_json"]}}]})]
            if d["type"] == "thinking_delta" and d["thinking"]:
                return [self._chunk({"reasoning_content": d["thinking"]})]
            return []
        if kind == "message_delta":
            usage = _usage({**self.start_usage, **(ev.get("usage") or {})})
            finish = FINISH.get(ev["delta"].get("stop_reason") or "", "stop")
            out = [self._chunk({"content": unfence(self.held)})] if self.held else []
            out.append(self._chunk({}, finish=finish, usage=None if self.include_usage else usage))
            if self.include_usage:
                out.append(self._chunk(None, usage=usage))
            return out
        if kind == "message_stop":
            return ["[DONE]"]
        return []
