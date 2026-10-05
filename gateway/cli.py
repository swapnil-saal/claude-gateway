"""The `claude-gateway` command.

    claude-gateway                 run in the foreground (same as `serve`)
    claude-gateway install         run as a login service: starts now, at every login, restarts on crash
    claude-gateway uninstall       stop and remove the login service (keeps the token and logs)
    claude-gateway start|stop|restart
    claude-gateway status          service state, health, warm processes, sessions, plan limits
    claude-gateway logs [-f] [-n N]
    claude-gateway token           print the API token clients should send

macOS uses a launchd agent, Linux a systemd user unit. Settings are read from
CLAUDE_GATEWAY_* environment variables; `install` captures the ones set when it runs.
"""
import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

from . import config

LABEL = "com.claude-gateway"
UNIT = "claude-gateway"
LOG = config.HOME / "gateway.log"


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


def _wait_healthy(seconds=40) -> bool:
    for _ in range(seconds * 2):
        if _healthy():
            return True
        time.sleep(0.5)
    return False


def _env() -> dict[str, str]:
    """Environment the service runs with: our settings, plus PATH so Claude Code finds its helpers."""
    env = {k: v for k, v in os.environ.items() if k.startswith("CLAUDE_GATEWAY_")}
    env["PATH"] = os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin")
    if "HOME" in os.environ:
        env["HOME"] = os.environ["HOME"]
    return env


def _program() -> list[str]:
    # The interpreter this command runs under has the package installed, whether via
    # `uv run`, `uv tool install` or a plain venv.
    return [sys.executable, "-m", "gateway", "serve"]


# ---------------------------------------------------------------- macOS (launchd)
class Launchd:
    plist = os.path.expanduser(f"~/Library/LaunchAgents/{LABEL}.plist")

    @property
    def domain(self):
        return f"gui/{os.getuid()}"

    def _write(self):
        def xml(s):
            return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        args = "".join(f"    <string>{xml(a)}</string>\n" for a in _program())
        env = "".join(f"    <key>{k}</key><string>{xml(v)}</string>\n" for k, v in _env().items())
        os.makedirs(os.path.dirname(self.plist), exist_ok=True)
        with open(self.plist, "w") as f:
            f.write(f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>{LABEL}</string>
  <key>ProgramArguments</key><array>
{args}  </array>
  <key>EnvironmentVariables</key><dict>
{env}  </dict>
  <key>WorkingDirectory</key><string>{config.HOME}</string>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>{LOG}</string>
  <key>StandardErrorPath</key><string>{LOG}</string>
</dict></plist>
""")

    def installed(self):
        return os.path.exists(self.plist)

    def install(self):
        self._write()
        self.start()

    def uninstall(self):
        self.stop()
        if self.installed():
            os.remove(self.plist)

    def start(self):
        if not self.installed():
            sys.exit("not installed: run `claude-gateway install` first")
        subprocess.run(["launchctl", "bootstrap", self.domain, self.plist], capture_output=True)
        subprocess.run(["launchctl", "kickstart", f"{self.domain}/{LABEL}"], capture_output=True)

    def stop(self):
        subprocess.run(["launchctl", "bootout", f"{self.domain}/{LABEL}"], capture_output=True)

    def restart(self):
        r = subprocess.run(["launchctl", "kickstart", "-k", f"{self.domain}/{LABEL}"], capture_output=True)
        if r.returncode:
            self.start()

    def pid(self) -> int | None:
        r = subprocess.run(["launchctl", "print", f"{self.domain}/{LABEL}"], capture_output=True, text=True)
        for line in r.stdout.splitlines():
            if line.strip().startswith("pid = "):
                return int(line.split("=")[1])
        return None


# ---------------------------------------------------------------- Linux (systemd --user)
class Systemd:
    unit = os.path.expanduser(f"~/.config/systemd/user/{UNIT}.service")

    def _ctl(self, *args, check=False):
        return subprocess.run(["systemctl", "--user", *args], capture_output=True, text=True, check=check)

    def _write(self):
        env = "".join(f"Environment={k}={v}\n" for k, v in _env().items())
        os.makedirs(os.path.dirname(self.unit), exist_ok=True)
        with open(self.unit, "w") as f:
            f.write(f"""[Unit]
Description=claude-gateway: Anthropic and OpenAI compatible API on your Claude Code login
After=network.target

[Service]
ExecStart={" ".join(_program())}
WorkingDirectory={config.HOME}
Restart=always
RestartSec=2
StandardOutput=append:{LOG}
StandardError=append:{LOG}
{env}
[Install]
WantedBy=default.target
""")

    def installed(self):
        return os.path.exists(self.unit)

    def install(self):
        self._write()
        self._ctl("daemon-reload")
        self._ctl("enable", "--now", UNIT)

    def uninstall(self):
        self._ctl("disable", "--now", UNIT)
        if self.installed():
            os.remove(self.unit)
        self._ctl("daemon-reload")

    def start(self):
        if not self.installed():
            sys.exit("not installed: run `claude-gateway install` first")
        self._ctl("start", UNIT)

    def stop(self):
        self._ctl("stop", UNIT)

    def restart(self):
        self._ctl("restart", UNIT)

    def pid(self) -> int | None:
        r = self._ctl("show", "-p", "MainPID", "--value", UNIT)
        return int(r.stdout.strip() or 0) or None


def service():
    system = platform.system()
    if system == "Darwin":
        return Launchd()
    if system == "Linux" and shutil.which("systemctl"):
        return Systemd()
    sys.exit(f"login services are supported on macOS and Linux (systemd); on {system} run `claude-gateway serve`")


# ---------------------------------------------------------------- commands
def cmd_serve(_args):
    import uvicorn
    from .server import app
    uvicorn.run(app, host=config.HOST, port=config.PORT, log_level="info")


def cmd_install(_args):
    svc = service()
    if _healthy() and not svc.pid():
        sys.exit(f"something else is already listening on {_base()}; stop it first")
    svc.install()
    if not _wait_healthy():
        sys.exit(f"the service did not come up; see `claude-gateway logs` ({LOG})")
    print(f"claude-gateway running on {_base()}  (starts at login, restarts on crash)")
    print(f"token: {config.TOKEN}")
    print(f"logs:  {LOG}")


def cmd_uninstall(_args):
    service().uninstall()
    print(f"claude-gateway stopped and removed; token and logs kept in {config.HOME}")


def cmd_start(_args):
    service().start()
    print("running on " + _base() if _wait_healthy() else f"did not come up; see `claude-gateway logs`")


def cmd_stop(_args):
    service().stop()
    for _ in range(20):
        if not _healthy():
            break
        time.sleep(0.5)
    print("stopped" if not _healthy() else "still answering: it may be running outside the service")


def cmd_restart(_args):
    service().restart()
    print("running on " + _base() if _wait_healthy() else f"did not come up; see `claude-gateway logs`")


def cmd_status(_args):
    try:
        svc, pid = service(), service().pid()
        print(f"service:  {'installed' if svc.installed() else 'not installed'}"
              + (f", running (pid {pid})" if pid else ", not running"))
    except SystemExit as e:
        print(f"service:  {e}")
    health = _healthy()
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
    for name in ("serve", "install", "uninstall", "start", "stop", "restart", "status", "token"):
        sub.add_parser(name)
    logs = sub.add_parser("logs")
    logs.add_argument("-f", "--follow", action="store_true")
    logs.add_argument("-n", "--lines", type=int, default=50)
    args = p.parse_args(argv)
    globals()[f"cmd_{args.cmd or 'serve'}"](args)


if __name__ == "__main__":
    main()
