"""claude-gateway: the Anthropic Messages API, served by your own Claude Code login.

POST /v1/messages                 Messages API (stream or not); header x-session-id keeps context across calls
POST /v1/chat/completions         the same, in OpenAI Chat Completions format (openai SDK, any OpenAI client)
POST /v1/messages/count_tokens    rough estimate (Claude Code has no token counter)
GET  /v1/models[/{id}]            model list
POST /v1/agent/run                full Claude Code agent run: built-in tools, MCP servers, cwd, JSON schema output
DELETE /v1/sessions/{id}          end a session
GET  /usage                       plan rate-limit windows, pools, sessions
GET  /health                      liveness, no auth
"""
import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager

from claude_agent_sdk import AssistantMessage, RateLimitEvent, ResultMessage, StreamEvent, TextBlock, ToolResultBlock, ToolUseBlock, UserMessage
from fastapi import Depends, FastAPI, Request, Security
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

from . import config, docs, openai_format
from .anthropic_format import Accumulator, APIError, Shaper, chat_spec, prefill_of, prompt, system_text, validate
from .pool import Registry

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("gateway")

registry = Registry()
slots = asyncio.Semaphore(config.CONCURRENCY)
rate_limits: dict[str, dict] = {}
QUEUE_TIMEOUT = 120
DEFAULT_SPEC = {"system": ""}


@asynccontextmanager
async def lifespan(_app):
    registry.warm(DEFAULT_SPEC)
    reaper = asyncio.create_task(registry.reap())
    log.info("claude-gateway on http://%s:%s (token in %s)", config.HOST, config.PORT, config.HOME / "token")
    yield
    reaper.cancel()
    await registry.aclose()


app = FastAPI(title="claude-gateway", version="0.1.0", description=docs.DESCRIPTION, openapi_tags=docs.TAGS,
              lifespan=lifespan)


@app.exception_handler(APIError)
async def api_error(req, e: APIError):
    body = openai_format.error_body(e) if "/chat/" in req.url.path else e.body()
    return JSONResponse(body, status_code=e.status, headers={"request-id": _request_id()})


def _request_id():
    return "req_gw_" + uuid.uuid4().hex[:24]


_api_key = APIKeyHeader(name="x-api-key", auto_error=False, description="The gateway token (`claude-gateway token`)")
_bearer = HTTPBearer(auto_error=False, description="The same token, as OpenAI clients send it")


def auth(key: str | None = Security(_api_key), cred: HTTPAuthorizationCredentials | None = Security(_bearer)):
    if (key or (cred.credentials if cred else "")) != config.TOKEN:
        raise APIError("authentication_error", "invalid x-api-key")


# ---------------------------------------------------------------- running a request
ASSISTANT_ERRORS = {"authentication_failed": "authentication_error", "billing_error": "permission_error",
                    "rate_limit": "rate_limit_error", "invalid_request": "invalid_request_error",
                    "server_error": "api_error", "unknown": "api_error"}
STATUS_ERRORS = {400: "invalid_request_error", 401: "authentication_error", 403: "permission_error",
                 404: "not_found_error", 413: "request_too_large", 429: "rate_limit_error", 529: "overloaded_error"}


def _user_message(content):
    return {"type": "user", "message": {"role": "user", "content": content}, "parent_tool_use_id": None,
            "session_id": "default"}


async def _one(item):
    yield item


async def _slot():
    try:
        await asyncio.wait_for(slots.acquire(), QUEUE_TIMEOUT)
    except TimeoutError:
        raise APIError("overloaded_error", f"all {config.CONCURRENCY} gateway slots busy for {QUEUE_TIMEOUT}s")


def _result_error(m: ResultMessage) -> APIError:
    text = "; ".join(m.errors or []) or m.result or m.subtype
    return APIError(STATUS_ERRORS.get(m.api_error_status or 0, "api_error"), text)


def _note_rate_limit(m: RateLimitEvent):
    info = m.rate_limit_info
    rate_limits[info.rate_limit_type or "unknown"] = info.raw
    if info.status == "rejected":
        raise APIError("rate_limit_error", f"Claude plan limit reached ({info.rate_limit_type}); "
                                           f"resets at {time.strftime('%H:%M', time.localtime(info.resets_at or 0))}")


async def chat_events(body: dict, session_id: str | None):
    """Shaped Messages API stream events for one request."""
    validate(body)
    spec = chat_spec(body)
    await _slot()
    session = None
    try:
        if session_id:
            session, continuing = await registry.session(session_id, spec)
            warm = True
            await session.lock.acquire()
            worker = session.worker
        else:
            worker, warm = await registry.pool(spec).acquire()
            continuing = False
        t0, t_model, t_first = time.monotonic(), None, None
        ok, status = False, "failed"
        try:
            try:
                await worker.set_model(body["model"])
            except Exception as e:
                raise APIError("not_found_error", f"model: {body['model']} ({e})")
            t_model = time.monotonic()
            await worker.client.query(_one(_user_message(prompt(body, continuing))))
            shaper = Shaper(body.get("stop_sequences"), prefill_of(body), thinking=bool(spec.get("thinking")))
            async with asyncio.timeout(config.REQUEST_TIMEOUT):
                async for m in worker.client.receive_response():
                    if isinstance(m, StreamEvent):
                        if t_first is None and m.event.get("type") == "message_start":
                            t_first = time.monotonic()
                        if m.parent_tool_use_id is None:
                            for ev in shaper.feed(m.event):
                                yield ev
                        if shaper.done and not session:
                            break              # stateless worker is thrown away; no need to drain
                    elif isinstance(m, RateLimitEvent):
                        _note_rate_limit(m)
                    elif isinstance(m, AssistantMessage) and m.error and not shaper.started:
                        raise APIError(ASSISTANT_ERRORS.get(m.error, "api_error"), f"upstream error: {m.error}")
                    elif isinstance(m, ResultMessage) and m.is_error and not shaper.done:
                        raise _result_error(m)
            if not shaper.done:
                raise APIError("api_error", "Claude Code ended the turn without a complete message")
            ok, status = True, "ok"
        except (GeneratorExit, asyncio.CancelledError):
            status = "cancelled"               # client went away mid-stream
            raise
        finally:
            log.info("%s model=%s %s set_model=%.2fs first_event=%s total=%.2fs",
                     f"session {session_id}" if session else ("warm" if warm else "COLD"), worker.model,
                     status, (t_model or t0) - t0,
                     f"{t_first - t0:.2f}s" if t_first else "-", time.monotonic() - t0)
            if session:
                session.lock.release()
                session.last_used, session.turns = time.monotonic(), session.turns + (1 if ok else 0)
                if not ok:
                    registry.drop_session(session_id)
            else:
                worker.discard()
    finally:
        slots.release()


def _sse(ev: dict) -> bytes:
    return f"event: {ev['type']}\ndata: {json.dumps(ev, separators=(',', ':'))}\n\n".encode()


@app.post("/v1/messages", dependencies=[Depends(auth)], **docs.MESSAGES)
async def messages(request: Request):
    try:
        body = await request.json()
    except json.JSONDecodeError:
        raise APIError("invalid_request_error", "request body is not valid JSON")
    session_id = request.headers.get("x-session-id")
    events = chat_events(body, session_id)
    headers = {"request-id": _request_id()}

    if not body.get("stream"):
        acc = Accumulator()
        async for ev in events:
            acc.add(ev)
        return JSONResponse(acc.result(), headers=headers)

    first = await anext(events)                # errors before the first event still get a proper status code

    async def stream():
        yield _sse(first)
        yield _sse({"type": "ping"})
        try:
            async for ev in events:
                yield _sse(ev)
        except APIError as e:
            yield _sse(e.body())
        finally:
            await events.aclose()

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={**headers, "cache-control": "no-cache"})


def _oai_sse(item) -> bytes:
    data = item if isinstance(item, str) else json.dumps(item, separators=(",", ":"))
    return f"data: {data}\n\n".encode()


@app.post("/v1/chat/completions", dependencies=[Depends(auth)], **docs.CHAT)
@app.post("/chat/completions", dependencies=[Depends(auth)], include_in_schema=False)   # base URL without /v1
async def chat_completions(request: Request):
    try:
        raw = await request.json()
    except json.JSONDecodeError:
        raise APIError("invalid_request_error", "request body is not valid JSON")
    body = openai_format.to_messages(raw)
    events = chat_events(body, request.headers.get("x-session-id"))
    headers = {"request-id": _request_id()}

    if not body["stream"]:
        acc = Accumulator()
        async for ev in events:
            acc.add(ev)
        return JSONResponse(openai_format.completion(acc.result(), body["model"], openai_format.json_mode(raw)),
                            headers=headers)

    first = await anext(events)                # errors before the first event still get a proper status code
    chunker = openai_format.Chunker(body["model"], bool((raw.get("stream_options") or {}).get("include_usage")),
                                    openai_format.json_mode(raw))

    async def stream():
        try:
            for c in chunker.feed(first):
                yield _oai_sse(c)
            async for ev in events:
                for c in chunker.feed(ev):
                    yield _oai_sse(c)
        except APIError as e:
            yield _oai_sse(openai_format.error_body(e))
            yield _oai_sse("[DONE]")
        finally:
            await events.aclose()

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={**headers, "cache-control": "no-cache"})


@app.post("/v1/messages/count_tokens", dependencies=[Depends(auth)], **docs.COUNT_TOKENS)
async def count_tokens(request: Request):
    body = await request.json()
    text = json.dumps(body.get("messages", [])) + system_text(body.get("system")) + json.dumps(body.get("tools", []))
    return {"input_tokens": max(1, len(text) // 4)}


@app.delete("/v1/sessions/{session_id}", dependencies=[Depends(auth)], **docs.END_SESSION)
async def end_session(session_id: str):
    return {"id": session_id, "deleted": registry.drop_session(session_id)}


# ---------------------------------------------------------------- models / status
def _model(mid, name):                         # Anthropic and OpenAI model objects, superimposed
    return {"type": "model", "object": "model", "id": mid, "display_name": name,
            "created_at": "2026-01-01T00:00:00Z", "created": 1767225600, "owned_by": "anthropic"}


@app.get("/v1/models", dependencies=[Depends(auth)], **docs.MODELS)
@app.get("/models", dependencies=[Depends(auth)], include_in_schema=False)
async def models():
    data = [_model(*m) for m in config.MODELS]
    return {"object": "list", "data": data, "has_more": False, "first_id": data[0]["id"], "last_id": data[-1]["id"]}


@app.get("/v1/models/{model_id}", dependencies=[Depends(auth)], **docs.MODEL)
@app.get("/models/{model_id}", dependencies=[Depends(auth)], include_in_schema=False)
async def model(model_id: str):
    for mid, name in config.MODELS:
        if mid == model_id:
            return _model(mid, name)
    raise APIError("not_found_error", f"model: {model_id}")


@app.get("/usage", dependencies=[Depends(auth)], **docs.USAGE)
async def usage():
    return {"rate_limits": rate_limits, "concurrency": config.CONCURRENCY,
            "busy_slots": config.CONCURRENCY - slots._value, **registry.stats()}


@app.get("/health", **docs.HEALTH)
async def health():
    return {"ok": True, "warm_spares": sum(len(p.idle) for p in registry.pools.values())}


# ---------------------------------------------------------------- agent runs
@app.post("/v1/agent/run", dependencies=[Depends(auth)], **docs.AGENT_RUN)
async def agent_run(request: Request):
    """A whole Claude Code agent loop. Body:
    prompt (str | content blocks, required), model, system (replaces Claude Code's prompt) or
    append_system (added to it), tools (built-in names, e.g. ["Read", "WebFetch"]),
    mcp_servers ({name: stdio/http config}), cwd, output_schema (JSON schema), max_turns,
    permission_mode, session_id (keeps the agent's context for follow-up runs).
    """
    body = await request.json()
    if not body.get("prompt"):
        raise APIError("invalid_request_error", "prompt: Field required")
    spec = {"agent": True, "system": body.get("system"), "append_system": body.get("append_system"),
            "builtin_tools": body.get("tools") or [], "mcp_servers": body.get("mcp_servers") or {},
            "cwd": body.get("cwd"), "max_turns": body.get("max_turns"),
            "permission_mode": body.get("permission_mode"),
            "output_format": {"type": "json_schema", "schema": body["output_schema"]}
            if body.get("output_schema") else None}
    session_id = body.get("session_id")
    content = body["prompt"] if isinstance(body["prompt"], list) else [{"type": "text", "text": body["prompt"]}]

    await _slot()
    session, t0, steps = None, time.monotonic(), []
    try:
        if session_id:
            session, _ = await registry.session("agent:" + session_id, spec)
            await session.lock.acquire()
            worker = session.worker
        else:
            worker, _ = await registry.pool(spec).acquire()
        ok = False
        try:
            await worker.set_model(body.get("model") or config.DEFAULT_MODEL)
            await worker.client.query(_one(_user_message(content)))
            result = None
            async with asyncio.timeout(config.REQUEST_TIMEOUT):
                async for m in worker.client.receive_response():
                    if isinstance(m, AssistantMessage):
                        for b in m.content:
                            if isinstance(b, TextBlock):
                                steps.append({"type": "text", "text": b.text})
                            elif isinstance(b, ToolUseBlock):
                                steps.append({"type": "tool_use", "name": b.name, "input": b.input})
                    elif isinstance(m, UserMessage) and isinstance(m.content, list):
                        for b in m.content:
                            if isinstance(b, ToolResultBlock):
                                text = b.content if isinstance(b.content, str) else json.dumps(b.content)
                                steps.append({"type": "tool_result", "is_error": bool(b.is_error),
                                              "content": (text or "")[:2000]})
                    elif isinstance(m, RateLimitEvent):
                        _note_rate_limit(m)
                    elif isinstance(m, ResultMessage):
                        result = m
            if result is None:
                raise APIError("api_error", "agent run ended without a result")
            ok = not result.is_error
            return {"id": "run_" + uuid.uuid4().hex[:20], "is_error": result.is_error, "subtype": result.subtype,
                    "result": result.result, "structured_output": result.structured_output,
                    "num_turns": result.num_turns, "duration_ms": round((time.monotonic() - t0) * 1000),
                    "usage": result.usage, "steps": steps, "session_id": session_id}
        finally:
            if session:
                session.lock.release()
                session.last_used = time.monotonic()
                if not ok:
                    registry.drop_session("agent:" + session_id)
            else:
                worker.discard()
    finally:
        slots.release()


def main():                                    # kept for `python -m gateway.server`
    from .cli import cmd_serve
    cmd_serve(None)


if __name__ == "__main__":
    main()
