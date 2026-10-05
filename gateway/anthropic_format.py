"""Anthropic Messages API <-> Claude Code.

Requests become a pool spec (system prompt, client tools, thinking) plus one
user message. Claude Code passes the model's raw stream events through, so the
response side is mostly clean-up: hide the MCP prefix on client tools, drop
fields the public API doesn't have, and apply stop_sequences, which Claude Code
doesn't support.
"""
import json

from .pool import CLIENT_TOOL_PREFIX


class APIError(Exception):
    STATUS = {"invalid_request_error": 400, "authentication_error": 401, "permission_error": 403,
              "not_found_error": 404, "request_too_large": 413, "rate_limit_error": 429,
              "api_error": 500, "overloaded_error": 529}

    def __init__(self, type_: str, message: str):
        super().__init__(message)
        self.type, self.message = type_, message

    @property
    def status(self):
        return self.STATUS.get(self.type, 500)

    def body(self):
        return {"type": "error", "error": {"type": self.type, "message": self.message}}


# ---------------------------------------------------------------- request
def validate(body: dict):
    for name in ("model", "max_tokens", "messages"):
        if name not in body:
            raise APIError("invalid_request_error", f"{name}: Field required")
    msgs = body["messages"]
    if not isinstance(msgs, list) or not msgs:
        raise APIError("invalid_request_error", "messages: at least one message is required")
    if msgs[0].get("role") != "user":
        raise APIError("invalid_request_error", "messages: first message must use the \"user\" role")
    for i, m in enumerate(msgs):
        if m.get("role") not in ("user", "assistant"):
            raise APIError("invalid_request_error", f"messages.{i}.role: Input should be 'user' or 'assistant'")


def system_text(system) -> str:
    if isinstance(system, list):
        return "\n\n".join(b.get("text", "") for b in system if b.get("type") == "text")
    return system or ""


def chat_spec(body: dict) -> dict:
    spec = {"system": system_text(body.get("system"))}
    tools = body.get("tools") or []
    if (body.get("tool_choice") or {}).get("type") == "none":
        tools = []
    client_tools = []
    for t in tools:
        if t.get("type") not in (None, "custom"):
            raise APIError("invalid_request_error",
                           f"tools: server tool {t.get('type')!r} is not supported by the gateway")
        client_tools.append({"name": t["name"], "description": t.get("description", ""),
                             "input_schema": t.get("input_schema") or {"type": "object"}})
    if client_tools:
        spec["client_tools"] = client_tools
    thinking = body.get("thinking")
    if thinking and thinking.get("type") in ("enabled", "adaptive"):
        spec["thinking"] = {**thinking, "display": thinking.get("display", "summarized")}
    return spec


def _blocks(content) -> list[dict]:
    return [{"type": "text", "text": content}] if isinstance(content, str) else list(content)


def _result_text(content) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(b.get("text", f"[{b.get('type')} block]") for b in content or [])


def _render(block: dict) -> dict | None:
    """A history block as something a fresh process can read; images stay native."""
    kind = block.get("type")
    if kind in ("text", "image", "document"):
        return {k: v for k, v in block.items() if k != "cache_control"}
    if kind == "tool_use":
        return {"type": "text", "text": f"[called tool {block['name']} (id {block['id']}) "
                                        f"with input {json.dumps(block.get('input', {}))}]"}
    if kind == "tool_result":
        err = " (error)" if block.get("is_error") else ""
        return {"type": "text", "text": f"[result{err} of tool call {block['tool_use_id']}]\n"
                                        f"{_result_text(block.get('content'))}"}
    return None                                # thinking, redacted_thinking, ...


def _tool_choice_note(body: dict) -> str:
    choice = body.get("tool_choice") or {}
    if choice.get("type") == "any":
        return "You must respond by calling one of your tools."
    if choice.get("type") == "tool":
        return f"You must respond by calling the {choice['name']} tool."
    return ""


def prefill_of(body: dict) -> str:
    """Text of a trailing assistant message, which the reply must continue."""
    last = body["messages"][-1]
    if last["role"] != "assistant":
        return ""
    return "".join(b.get("text", "") for b in _blocks(last["content"]) if b.get("type") == "text")


def prompt(body: dict, continuing_session: bool) -> list[dict]:
    """Content blocks for the single user message sent to Claude Code.

    A continuing session already holds the earlier turns, so only the newest
    user message is sent. Otherwise the earlier turns are written out as a
    transcript ahead of it.
    """
    msgs = body["messages"]
    prefill = prefill_of(body)
    if msgs[-1]["role"] == "assistant":
        msgs = msgs[:-1]
    last, history = msgs[-1], ([] if continuing_session else msgs[:-1])

    out: list[dict] = []
    if history:
        out.append({"type": "text", "text": (
            "<conversation_history>\nThe earlier turns of this conversation, oldest first. You are the "
            "Assistant. Reply to the latest User message that follows. Call tools only through your tool "
            "mechanism, never by writing a call as text.\n")})
        for m in history:
            out.append({"type": "text", "text": f"\n{m['role'].capitalize()}:"})
            out += [r for b in _blocks(m["content"]) if (r := _render(b))]
        out.append({"type": "text", "text": "\n</conversation_history>\n\nUser:"})

    for b in _blocks(last["content"]):
        if continuing_session and b.get("type") == "tool_result":
            out.append({k: v for k, v in b.items() if k != "cache_control"})    # answers the deferred call
        elif r := _render(b):
            out.append(r)

    notes = [n for n in (_tool_choice_note(body),) if n]
    if prefill:
        notes.append(f"Begin your reply exactly with the following text, then continue it: {prefill}")
    if notes:
        out.append({"type": "text", "text": "\n\n" + "\n".join(notes)})
    return out or [{"type": "text", "text": ""}]


# ---------------------------------------------------------------- response
class Shaper:
    """Cleans raw stream events into public API events and applies stop_sequences.

    feed() returns the events to send; `done` is set once message_stop is emitted.
    """

    def __init__(self, stop_sequences=None, prefill="", thinking=False):
        self.stops = [s for s in stop_sequences or [] if s]
        self.hold = max((len(s) for s in self.stops), default=1) - 1
        self.strip = prefill                   # the API never echoes a prefill; the model is asked to repeat it
        self.pending: dict[int, str] = {}      # text held back while a stop sequence might still be forming
        self.started = self.done = False
        # Some models think even when asked not to, and Claude Code then streams an empty
        # thinking block. The API never returns thinking the caller did not enable, so those
        # blocks are dropped and the remaining ones renumbered to stay contiguous.
        self.thinking = thinking
        self.skipped: set[int] = set()         # raw indices of dropped blocks
        self.index: dict[int, int] = {}        # raw index -> index as sent

    def feed(self, ev: dict) -> list[dict]:
        if self.done:
            return []
        kind = ev.get("type")
        if kind == "message_start":
            if self.started:                   # a second model call in one request: not expected in chat mode
                return []
            self.started = True
            msg = {k: v for k, v in ev["message"].items() if k != "container"}
            return [{"type": "message_start", "message": msg}]
        if not self.started:
            return []
        if kind in ("content_block_start", "content_block_delta", "content_block_stop"):
            raw = ev["index"]
            if kind == "content_block_start":
                if not self.thinking and ev["content_block"].get("type") in ("thinking", "redacted_thinking"):
                    self.skipped.add(raw)
                else:
                    self.index[raw] = len(self.index)
            if raw in self.skipped:
                return []
            ev = {**ev, "index": self.index.get(raw, raw)}
        if kind == "content_block_start":
            block = dict(ev["content_block"])
            block.pop("caller", None)
            if block.get("type") == "tool_use" and block.get("name", "").startswith(CLIENT_TOOL_PREFIX):
                block["name"] = block["name"][len(CLIENT_TOOL_PREFIX):]
            if block.get("type") == "text":
                self.pending[ev["index"]] = ""
            return [{"type": kind, "index": ev["index"], "content_block": block}]
        if kind == "content_block_delta":
            delta = {k: v for k, v in ev["delta"].items() if k != "estimated_tokens"}
            if delta.get("type") == "text_delta":
                return self._text(ev["index"], delta["text"])
            return [{"type": kind, "index": ev["index"], "delta": delta}]
        if kind == "content_block_stop":
            rest = self.pending.pop(ev["index"], "")
            return ([self._delta(ev["index"], rest)] if rest else []) + [{"type": kind, "index": ev["index"]}]
        if kind == "message_delta":
            delta = {k: v for k, v in ev["delta"].items() if k not in ("stop_details", "container")}
            return [{"type": kind, "delta": delta, "usage": ev.get("usage", {})}]
        if kind == "message_stop":
            self.done = True
            return [{"type": "message_stop"}]
        return []

    def _text(self, index: int, text: str) -> list[dict]:
        if self.strip:
            n = next((i for i, (a, b) in enumerate(zip(text, self.strip)) if a != b), min(len(text), len(self.strip)))
            if n == len(text) and n < len(self.strip):
                self.strip = self.strip[n:]
                return []
            text = text[n:] if n == len(self.strip) else text
            self.strip = ""
        buf = self.pending.get(index, "") + text
        # Holding back hold = longest stop - 1 chars means any match lies wholly inside buf.
        hits = [(at, s) for s in self.stops if (at := buf.find(s)) != -1]
        if hits:
            at, s = min(hits)
            self.done = True
            return ([self._delta(index, buf[:at])] if at else []) + [
                {"type": "content_block_stop", "index": index},
                {"type": "message_delta", "delta": {"stop_reason": "stop_sequence", "stop_sequence": s},
                 "usage": {"output_tokens": 0}},
                {"type": "message_stop"}]
        send, self.pending[index] = (buf[:-self.hold], buf[-self.hold:]) if self.hold else (buf, "")
        return [self._delta(index, send)] if send else []

    @staticmethod
    def _delta(index, text):
        return {"type": "content_block_delta", "index": index, "delta": {"type": "text_delta", "text": text}}


class Accumulator:
    """Builds the non-streaming Message object from shaped events."""

    def __init__(self):
        self.message: dict | None = None
        self.blocks: dict[int, dict] = {}
        self.json_parts: dict[int, str] = {}

    def add(self, ev: dict):
        kind = ev["type"]
        if kind == "message_start":
            self.message = dict(ev["message"])
        elif kind == "content_block_start":
            self.blocks[ev["index"]] = dict(ev["content_block"])
        elif kind == "content_block_delta":
            b, d = self.blocks[ev["index"]], ev["delta"]
            match d.get("type"):
                case "text_delta":
                    b["text"] = b.get("text", "") + d["text"]
                case "input_json_delta":
                    self.json_parts[ev["index"]] = self.json_parts.get(ev["index"], "") + d["partial_json"]
                case "thinking_delta":
                    b["thinking"] = b.get("thinking", "") + d["thinking"]
                case "signature_delta":
                    b["signature"] = d["signature"]
                case "citations_delta":
                    b.setdefault("citations", []).append(d["citation"])
        elif kind == "content_block_stop":
            if (raw := self.json_parts.pop(ev["index"], None)) is not None:
                self.blocks[ev["index"]]["input"] = json.loads(raw) if raw.strip() else {}
        elif kind == "message_delta":
            self.message.update(ev["delta"])
            self.message["usage"] = {**self.message.get("usage", {}), **ev.get("usage", {})}

    def result(self) -> dict:
        msg = self.message
        msg["content"] = [self.blocks[i] for i in sorted(self.blocks)]
        return msg
