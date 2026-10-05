"""Settings, read from the environment once at import.

CLAUDE_GATEWAY_HOST / _PORT       bind address (keep it on 127.0.0.1 or a tailnet IP)
CLAUDE_GATEWAY_TOKEN              bearer token; generated into ~/.claude-gateway/token when unset
CLAUDE_GATEWAY_SPARES             warm processes kept for the default pool (default 2)
CLAUDE_GATEWAY_MAX_IDLE           warm processes kept across all pools (default 4)
CLAUDE_GATEWAY_POOL_TTL           seconds an unused non-default pool keeps its spare (default 300)
CLAUDE_GATEWAY_CONCURRENCY        requests that may run at once (default 3)
CLAUDE_GATEWAY_MAX_POOLS          distinct pools (system prompt + tools) kept warm (default 6)
CLAUDE_GATEWAY_SESSION_TTL        seconds an idle session process lives (default 900)
CLAUDE_GATEWAY_DEFAULT_MODEL      model when a request's alias is unknown (default sonnet)
"""
import os
import secrets
from pathlib import Path

HOME = Path(os.environ.get("CLAUDE_GATEWAY_HOME", Path.home() / ".claude-gateway"))
# Empty cwd, so no CLAUDE.md or project settings get picked up. Session viewers group
# gateway processes under this folder's name.
WORKDIR = HOME / "claude-gateway"
HOST = os.environ.get("CLAUDE_GATEWAY_HOST", "127.0.0.1")
PORT = int(os.environ.get("CLAUDE_GATEWAY_PORT", "8787"))
SPARES = int(os.environ.get("CLAUDE_GATEWAY_SPARES", "2"))
CONCURRENCY = int(os.environ.get("CLAUDE_GATEWAY_CONCURRENCY", "3"))
MAX_IDLE = int(os.environ.get("CLAUDE_GATEWAY_MAX_IDLE", "4"))
POOL_TTL = int(os.environ.get("CLAUDE_GATEWAY_POOL_TTL", "300"))
MAX_POOLS = int(os.environ.get("CLAUDE_GATEWAY_MAX_POOLS", "6"))
SESSION_TTL = int(os.environ.get("CLAUDE_GATEWAY_SESSION_TTL", "900"))
MAX_SESSIONS = int(os.environ.get("CLAUDE_GATEWAY_MAX_SESSIONS", "8"))
SPARE_MAX_AGE = 3600             # recycle idle spares after an hour
CONNECT_TIMEOUT = 60
REQUEST_TIMEOUT = 600
DEFAULT_MODEL = os.environ.get("CLAUDE_GATEWAY_DEFAULT_MODEL", "sonnet")

# Claude Code switches to an alias instantly but checks other full ids with the API
# (~1.2 s, once per process), so ids that an alias already points to are sent as the alias.
ALIASES = {
    "claude-haiku-4-5-20251001": "haiku", "claude-haiku-4-5": "haiku",
    "claude-sonnet-5-5": "sonnet", "claude-opus-5-5": "opus",
}

MODELS = [  # what GET /v1/models reports; any id Claude Code accepts also works
    ("claude-opus-5-5", "Claude Opus 5.5"),
    ("claude-sonnet-5-5", "Claude Sonnet 5.5"),
    ("claude-haiku-4-5-20251001", "Claude Haiku 4.5"),
    ("claude-fable-5-1", "Claude Fable 5.1"),
]


def _token():
    if tok := os.environ.get("CLAUDE_GATEWAY_TOKEN"):
        return tok
    path = HOME / "token"
    if path.exists():
        return path.read_text().strip()
    HOME.mkdir(parents=True, exist_ok=True)
    tok = "cgw-" + secrets.token_urlsafe(24)
    path.write_text(tok + "\n")
    path.chmod(0o600)
    return tok


TOKEN = _token()
WORKDIR.mkdir(parents=True, exist_ok=True)
