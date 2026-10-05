"""The `claude-gateway` command.

    claude-gateway                 run in the foreground (same as `serve`); Ctrl+C stops it
    claude-gateway start           run in the background
    claude-gateway stop            stop the background server
    claude-gateway restart
    claude-gateway status          process, health, warm processes, sessions, plan limits
    claude-gateway logs [-f] [-n N]
    claude-gateway token           print the API token clients should send

The background server writes its pid to ~/.claude-gateway/gateway.pid and its output to
~/.claude-gateway/gateway.log. It does not start again after a reboot; run `start` again.
Settings come from CLAUDE_GATEWAY_* environment variables, read when the server starts.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

from . import config

LOG = config.HOME / "gateway.log"
PIDFILE = config.HOME / "gateway.pid"


def _base():
    return f"http://{config.HOST}:{config.PORT}"


def _get(path: str, timeout=3):
    req = urllib.request.Request(_base() + path, headers={"x-api-key": config.TOKEN})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def _healthy() -> dict | None:
    try:
        return _get("/health")
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


def _wait(cond, seconds=40) -> bool:
    for _ in range(seconds * 2):
        if cond():
            return True
        time.sleep(0.5)
    return False


def _pid() -> int | None:
    """Pid of the background server, if the one we recorded is still alive."""
    try:
        pid = int(PIDFILE.read_text().strip())
        os.kill(pid, 0)
        return pid
    except (FileNotFoundError, ValueError, ProcessLookupError, PermissionError):
        return None


# ---------------------------------------------------------------- commands
def cmd_serve(_args):
    import uvicorn
    from .server import app
    uvicorn.run(app, host=config.HOST, port=config.PORT, log_level="info")


def cmd_start(_args):
    if pid := _pid():
        sys.exit(f"already running (pid {pid}) on {_base()}")
    if _healthy():
        sys.exit(f"something is already answering on {_base()}; stop it first")
    config.HOME.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as log:
        # Its own session, so closing this terminal does not take it down.
        proc = subprocess.Popen([sys.executable, "-m", "gateway", "serve"], stdout=log, stderr=log,
                                stdin=subprocess.DEVNULL, start_new_session=True, cwd=config.HOME)
    PIDFILE.write_text(f"{proc.pid}\n")
    if not _wait(lambda: _healthy() or proc.poll() is not None) or proc.poll() is not None:
        PIDFILE.unlink(missing_ok=True)
        sys.exit(f"did not come up; see `claude-gateway logs` ({LOG})")
    print(f"claude-gateway running on {_base()} (pid {proc.pid})")
    print(f"token: {config.TOKEN}")
    print(f"logs:  {LOG}")


def cmd_stop(_args):
    pid = _pid()
    if not pid:
        PIDFILE.unlink(missing_ok=True)
        if _healthy():
            sys.exit(f"{_base()} is answering, but it was not started with `claude-gateway start`; "
                     "stop it where it runs (Ctrl+C in its terminal)")
        print("not running")
        return
    os.kill(pid, signal.SIGTERM)               # uvicorn shuts down cleanly and closes the Claude Code processes

    def gone():
        try:
            os.kill(pid, 0)
            return False
        except ProcessLookupError:
            return True
    if not _wait(gone, 15):
        os.kill(pid, signal.SIGKILL)
    for sig in (signal.SIGTERM, signal.SIGKILL):   # anything left in its process group (Claude Code children)
        try:
            os.killpg(pid, sig)
            time.sleep(1)
        except ProcessLookupError:
            break
    PIDFILE.unlink(missing_ok=True)
    print("stopped")


def cmd_restart(args):
    if _pid():
        cmd_stop(args)
    cmd_start(args)


def cmd_status(_args):
    pid, health = _pid(), _healthy()
    if pid:
        print(f"process:  running in the background (pid {pid})")
    elif health:
        print("process:  running, but not started with `claude-gateway start` (foreground or another tool)")
    else:
        print("process:  not running")
    if not health:
        print(f"endpoint: {_base()} not answering")
        return
    u = _get("/usage")
    print(f"endpoint: {_base()} ok, {health['warm_spares']} warm process(es), "
          f"{u['busy_slots']}/{u['concurrency']} slots busy")
    for p in u["pools"]:
        what = "agent" if p["agent"] else (f"system={p['system']!r}" if p["system"] else "default")
        extra = ", tools " + ",".join(p["client_tools"]) if p["client_tools"] else ""
        print(f"  pool {p['key']}: {p['idle']} idle, {p['spawning']} starting, {p['uses']} uses, {what}{extra}")
    for s in u["sessions"]:
        print(f"  session {s['id']}: {s['turns']} turns, idle {s['idle_s']}s")
    for kind, info in (u.get("rate_limits") or {}).items():
        used = info.get("utilization")
        resets = info.get("resetsAt") or info.get("resets_at")
        when = time.strftime("%a %H:%M", time.localtime(resets)) if resets else "?"
        print(f"  plan {kind}: {info.get('status', '?')}"
              + (f", {used:.0%} used" if isinstance(used, (int, float)) else "") + f", resets {when}")
    print(f"token:    {config.HOME / 'token'}")
    print(f"logs:     {LOG}")


def cmd_logs(args):
    if not LOG.exists():
        sys.exit(f"no log yet at {LOG}")
    cmd = ["tail", "-n", str(args.lines)] + (["-f"] if args.follow else []) + [str(LOG)]
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        pass


def cmd_token(_args):
    print(config.TOKEN)


def main(argv=None):
    p = argparse.ArgumentParser(prog="claude-gateway", description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = p.add_subparsers(dest="cmd")
    for name in ("serve", "start", "stop", "restart", "status", "token"):
        sub.add_parser(name)
    logs = sub.add_parser("logs")
    logs.add_argument("-f", "--follow", action="store_true")
    logs.add_argument("-n", "--lines", type=int, default=50)
    args = p.parse_args(argv)
    globals()[f"cmd_{args.cmd or 'serve'}"](args)


if __name__ == "__main__":
    main()
