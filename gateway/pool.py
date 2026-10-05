"""Warm Claude Code processes.

Starting a Claude Code process and completing its handshake takes 1-3 s, so
processes are started ahead of time and kept connected until a request needs
one.

A process remembers everything it has been sent, so a stateless request takes
a spare, uses it once and throws it away while a replacement starts in the
background. A request carrying a session id keeps one process across turns.

A process's system prompt and tools are fixed when it starts, so spares are
grouped into pools by that spec. The model is switched per request.

Only the default pool keeps CLAUDE_GATEWAY_SPARES ready from the start. Any other
pool starts empty and gets one spare once its spec is used a second time, so a
one-off system prompt leaves nothing running. Idle spares across all pools are
capped at CLAUDE_GATEWAY_MAX_IDLE.
"""
import asyncio
import hashlib
import json
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient, HookMatcher, SdkMcpTool, create_sdk_mcp_server

from . import config

log = logging.getLogger("gateway.pool")
CLIENT_TOOL_SERVER = "c"                       # client tools appear to the model as mcp__c__<name>
CLIENT_TOOL_PREFIX = f"mcp__{CLIENT_TOOL_SERVER}__"

# Claude Code always tells the model it is an agent built on the Agent SDK and adds the working
# directory, platform, shell, OS and date, even with an empty system prompt. Left alone, models
# (Haiku especially) introduce themselves and describe that environment, which makes a reply to
# "hi" 5-10x longer than through the API. This preamble goes ahead of the caller's system prompt.
CHAT_PREAMBLE = (
    "You are serving requests made to the Anthropic Messages API. The agent, working-directory, "
    "platform, shell and date details in your context describe the hosting process, not the user: "
    "ignore them and do not mention them, Claude Code or the Agent SDK. You have no file access and "
    "no tools other than those listed. Reply as you would through the API; do not introduce "
    "yourself or list what you can do unless asked."
)


def spec_key(spec: dict) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]


async def _defer_client_tool(_input, _tool_use_id, _ctx):
    # Stop the run at the tool call so the caller executes it, as with the real API.
    return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "defer"}}


async def _never_called(_args):
    return {"content": [{"type": "text", "text": "client tool executed in gateway"}], "is_error": True}


def build_options(spec: dict) -> ClaudeAgentOptions:
    """Turn a JSON spec into Claude Code options.

    Chat specs: system, client_tools, thinking.
    Agent specs add: builtin_tools, mcp_servers, cwd, output_format, max_turns, permission_mode, append_system.
    """
    agent = spec.get("agent", False)
    mcp_servers = dict(spec.get("mcp_servers") or {})
    allowed = list(spec.get("builtin_tools") or [])
    hooks = None
    if client_tools := spec.get("client_tools"):
        tools = [SdkMcpTool(t["name"], t.get("description", ""), t.get("input_schema") or {"type": "object"},
                            _never_called) for t in client_tools]
        mcp_servers[CLIENT_TOOL_SERVER] = create_sdk_mcp_server(CLIENT_TOOL_SERVER, tools=tools)
        allowed += [CLIENT_TOOL_PREFIX + t["name"] for t in client_tools]
        hooks = {"PreToolUse": [HookMatcher(matcher=f"{CLIENT_TOOL_PREFIX}.*", hooks=[_defer_client_tool])]}
    allowed += [f"mcp__{name}__*" for name in spec.get("mcp_servers") or {}]

    if agent:
        system = ({"type": "preset", "preset": "claude_code", "append": spec["append_system"]}
                  if spec.get("append_system") else spec.get("system"))
    else:                                      # chat: hide the hosting process from the model
        system = CHAT_PREAMBLE + (f"\n\n{spec['system']}" if spec.get("system") else "")

    return ClaudeAgentOptions(
        tools=spec.get("builtin_tools") or [],
        allowed_tools=allowed,
        mcp_servers=mcp_servers,
        strict_mcp_config=True,
        system_prompt=system,
        setting_sources=[],                     # ignore user hooks, CLAUDE.md and settings
        include_partial_messages=True,
        thinking=spec.get("thinking") or {"type": "disabled"},
        max_turns=spec.get("max_turns") or (None if agent else 1),
        permission_mode=spec.get("permission_mode") or "default",
        output_format=spec.get("output_format"),
        cwd=spec.get("cwd") or str(config.WORKDIR),
        model=config.DEFAULT_MODEL,
        hooks=hooks,
        # Keep gateway traffic out of ~/.claude/projects (and out of anything tailing it).
        extra_args={"no-session-persistence": None},
        stderr=lambda line: log.debug("cli: %s", line),
    )


@dataclass
class Worker:
    client: ClaudeSDKClient
    spec_key: str
    model: str = config.DEFAULT_MODEL
    born: float = field(default_factory=time.monotonic)

    async def set_model(self, model: str):
        model = config.ALIASES.get(model, model)
        if model == self.model:
            return
        for attempt in range(2):               # the CLI's model check against the API fails now and then
            try:
                await self.client.set_model(model)
                break
            except Exception:
                if attempt:
                    raise
        self.model = model
        remember_model(model)                  # only models that exist are worth pre-checking in new workers

    def discard(self):
        asyncio.create_task(self._close())

    async def _close(self):
        try:
            await asyncio.wait_for(self.client.disconnect(), 10)
        except BaseException as e:            # a dying process may raise anything; never let it escape
            log.debug("disconnect failed: %r", e)


async def spawn(spec: dict, key: str) -> Worker:
    client = ClaudeSDKClient(build_options(spec))
    t = time.monotonic()
    await asyncio.wait_for(client.connect(), config.CONNECT_TIMEOUT)
    worker = Worker(client, key, model="")
    # A process checks each non-alias model with the API the first time it switches to it (~1.2 s).
    # Do that now for the models requests have been using, then settle on the default.
    for model in [*recent_models, config.DEFAULT_MODEL]:
        try:
            await asyncio.wait_for(worker.set_model(model), config.CONNECT_TIMEOUT)
        except Exception as e:
            log.warning("could not pre-check model %s: %r", model, e)
            if "not found" in str(e).lower() and model in recent_models:
                recent_models.remove(model)
    log.info("spawned worker for pool %s in %.2fs", key, time.monotonic() - t)
    return worker


recent_models: list[str] = []                  # non-alias models seen lately, newest last


def remember_model(model: str):
    if model in ("sonnet", "opus", "haiku", "fable", config.DEFAULT_MODEL) or model in config.ALIASES.values():
        return
    if model in recent_models:
        recent_models.remove(model)
    recent_models.append(model)
    del recent_models[:-3]


class Pool:
    def __init__(self, spec: dict, spares: int, has_room):
        self.spec, self.key, self.spares = spec, spec_key(spec), spares
        self.has_room = has_room               # False once the gateway-wide idle cap is reached
        self.idle: list[Worker] = []
        self.spawning = 0
        self.uses = 0
        self.last_used = time.monotonic()

    def top_up(self):
        while len(self.idle) + self.spawning < self.spares and self.has_room():
            self.spawning += 1
            asyncio.create_task(self._fill())

    async def _fill(self):
        try:
            self.idle.append(await spawn(self.spec, self.key))
        except Exception as e:
            log.warning("could not start a spare for pool %s: %r", self.key, e)
        finally:
            self.spawning -= 1

    async def acquire(self) -> tuple[Worker, bool]:
        """A connected worker, and whether it was already warm."""
        self.last_used = time.monotonic()
        self.uses += 1
        if self.uses >= 2:                     # reused spec: worth keeping one ready
            self.spares = max(self.spares, 1)
        now = time.monotonic()
        for w in [w for w in self.idle if now - w.born > config.SPARE_MAX_AGE]:
            self.idle.remove(w)
            w.discard()
        if self.idle:
            worker, warm = self.idle.pop(0), True
        else:
            worker, warm = await spawn(self.spec, self.key), False
        self.top_up()
        return worker, warm

    def close(self):
        for w in self.idle:
            w.discard()
        self.idle.clear()
        self.spares = 0


@dataclass
class Session:
    worker: Worker
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    last_used: float = field(default_factory=time.monotonic)
    turns: int = 0


class Registry:
    def __init__(self):
        self.pools: OrderedDict[str, Pool] = OrderedDict()
        self.sessions: OrderedDict[str, Session] = OrderedDict()
        self.pinned: set[str] = set()

    def pool(self, spec: dict, pin: bool = False) -> Pool:
        key = spec_key(spec)
        if key not in self.pools:
            self.pools[key] = Pool(spec, config.SPARES if pin else 0, self._has_room)
            while len(self.pools) > config.MAX_POOLS:
                old = next((k for k in self.pools if k not in self.pinned and k != key), None)
                if old is None:
                    break
                self.pools.pop(old).close()
        if pin:
            self.pinned.add(key)
        self.pools.move_to_end(key)
        return self.pools[key]

    def _has_room(self) -> bool:
        return sum(len(p.idle) + p.spawning for p in self.pools.values()) < config.MAX_IDLE

    def warm(self, spec: dict):
        self.pool(spec, pin=True).top_up()

    async def session(self, sid: str, spec: dict) -> tuple[Session, bool]:
        """The session's process, and whether it already existed with this spec."""
        key = spec_key(spec)
        s = self.sessions.get(sid)
        if s and s.worker.spec_key == key:
            self.sessions.move_to_end(sid)
            return s, True
        if s:                                  # system prompt or tools changed: start over
            self.drop_session(sid)
        worker, _ = await self.pool(spec).acquire()
        s = self.sessions[sid] = Session(worker)
        while len(self.sessions) > config.MAX_SESSIONS:
            self.drop_session(next(iter(self.sessions)))
        return s, False

    def drop_session(self, sid: str) -> bool:
        s = self.sessions.pop(sid, None)
        if s:
            s.worker.discard()
        return s is not None

    async def reap(self):
        while True:
            await asyncio.sleep(60)
            now = time.monotonic()
            for sid, s in list(self.sessions.items()):
                if now - s.last_used > config.SESSION_TTL and not s.lock.locked():
                    self.drop_session(sid)
            for key, p in list(self.pools.items()):
                if key not in self.pinned and now - p.last_used > config.POOL_TTL:
                    self.pools.pop(key).close()

    def close(self):
        for p in self.pools.values():
            p.close()
        for sid in list(self.sessions):
            self.drop_session(sid)

    def stats(self):
        return {
            "pools": [{"key": p.key, "idle": len(p.idle), "spawning": p.spawning, "spares": p.spares, "uses": p.uses,
                       "pinned": p.key in self.pinned, "thinking": bool(p.spec.get("thinking")),
                       "system": (p.spec.get("system") or "")[:60],
                       "client_tools": [t["name"] for t in p.spec.get("client_tools") or []],
                       "agent": p.spec.get("agent", False)} for p in self.pools.values()],
            "sessions": [{"id": sid, "turns": s.turns, "idle_s": round(time.monotonic() - s.last_used)}
                         for sid, s in self.sessions.items()],
        }
