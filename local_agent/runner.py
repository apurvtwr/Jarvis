"""`jarvis runner`: executes agent tools on this computer for a (possibly remote) server.

The runner opens an outbound WebSocket to ``/api/runners/connect`` (nothing listens
locally), registers a stable ``runner_id`` and hostname, and answers tool calls.

It does not trust the server. Every path is resolved locally (symlinks followed) and
must be inside both

* the grants the server sends with the call (what the user attached in a chat), and
* this machine's own ``runner.allow`` roots (default: the home directory), minus
  ``runner.deny`` (credential stores such as ~/.ssh by default),

so a compromised server cannot reach anything the user did not expose here.
``runner.allow_commands: false`` disables ``run_command`` locally. Approvals are
decided on the server; the runner never approves anything itself.

Only stdlib, ``tools`` and ``client`` are imported here: never ``core``.
"""
import json
import os
import socket
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

from . import tools
from .client import is_local_url

DEFAULT_DENY = ("~/.ssh", "~/.gnupg", "~/.config/jarvis", "~/.local/share/keyrings", "~/.password-store",
                "~/.aws", "~/.kube", "~/.docker", "~/.netrc", "~/.pki")
RUNNER_ID_FILE = Path(os.environ.get("JARVIS_RUNNER_ID_FILE", Path.home() / ".config/jarvis/runner-id"))


class AuthError(Exception):
    pass


def runner_id(path=None):
    """A stable id for this machine's runner, created on first use."""
    path = Path(path or RUNNER_ID_FILE)
    try:
        value = path.read_text().strip()
        if value:
            return value
    except OSError:
        pass
    value = uuid.uuid4().hex
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value + "\n")
    return value


def _roots(items):
    return [Path(p).expanduser().resolve() for p in items or []]


class Runner:
    """Executes one call message and builds the reply. Pure and synchronous: easy to test."""

    def __init__(self, allow=None, deny=DEFAULT_DENY, allow_commands=True, limits=None):
        self.allow = _roots(allow if allow else ["~"])
        self.deny = _roots(deny)
        self.allow_commands = allow_commands
        self.limits = {**tools.LIMITS, **(limits or {})}

    def permit(self, p):
        """Local policy: ``p`` (already resolved) must be inside an allow root and no deny root."""
        if not any(p.is_relative_to(root) for root in self.allow):
            raise ValueError("Path is outside the folders this runner exposes (runner.allow)")
        if any(p.is_relative_to(root) for root in self.deny):
            raise ValueError("Path is in a protected location on this computer (runner.deny)")
        return p

    def check(self, grants):
        roots = _roots(g for g in grants if isinstance(g, str) and os.path.isabs(g))

        def checker(path):
            # Same symlink-resolving boundary as core.allowed_path, then the local policy.
            return self.permit(tools.within(path, roots))
        return checker

    def execute(self, tool, args, grants):
        if tool == "resolve":  # the server asks where a grant lives before storing it
            path = (args or {}).get("path")
            if not isinstance(path, str):
                raise ValueError("A path is required")
            return str(self.permit(tools.grantable(path)))
        if tool not in tools.NAMES:
            raise ValueError("Unknown tool")
        if tool == "run_command" and not self.allow_commands:
            raise ValueError("Commands are disabled on this runner (runner.allow_commands)")
        return tools.execute(tool, args or {}, self.check(grants or []), self.limits)

    def handle(self, msg):
        call_id = msg.get("call_id")
        try:
            result = self.execute(msg.get("tool"), msg.get("args"), msg.get("grants"))
            return {"type": "result", "call_id": call_id, "ok": True, "result": result}
        except Exception as exc:
            return {"type": "result", "call_id": call_id, "ok": False, "error": str(exc) or exc.__class__.__name__}


def connect_url(server_url):
    """http(s)://host/base → ws(s)://host/base/api/runners/connect. Plain ws only on loopback."""
    parsed = urlparse(server_url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"Unsupported server URL: {server_url}")
    if parsed.scheme == "http" and not is_local_url(server_url):
        raise ValueError("Refusing to send the runner token over plain HTTP to a remote server; use https://")
    scheme = "wss" if parsed.scheme == "https" else "ws"
    return f"{scheme}://{parsed.netloc}{parsed.path.rstrip('/')}/api/runners/connect"


def serve_connection(ws, runner, rid, hostname, log=print, workers=4):
    """Run the protocol over an open connection with ``send(str)``/``recv()``/iteration
    (websockets' sync client). Returns when the connection closes."""
    ws.send(json.dumps({"type": "hello", "runner_id": rid, "hostname": hostname}))
    welcome = json.loads(ws.recv(timeout=15))
    if welcome.get("type") != "welcome":
        raise ConnectionError(f"Unexpected reply from server: {welcome!r}")
    log(f"Runner {hostname} ({rid[:8]}) connected")
    lock = threading.Lock()

    def answer(msg):
        reply = runner.handle(msg)
        with lock:
            try:
                ws.send(json.dumps(reply))
            except Exception:
                pass  # connection gone; the server times the call out

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="jarvis-runner") as pool:
        for raw in ws:
            try:
                msg = json.loads(raw)
            except ValueError:
                continue
            if isinstance(msg, dict) and msg.get("type") == "call":
                pool.submit(answer, msg)


def run_forever(server_url, token, runner, rid=None, hostname=None, log=print):
    """Connect, serve, and reconnect with backoff until interrupted or the token is rejected."""
    from websockets.exceptions import ConnectionClosed, InvalidStatus
    from websockets.sync.client import connect

    url = connect_url(server_url)
    rid = rid or runner_id()
    hostname = hostname or socket.gethostname()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    delay = 1
    while True:
        started = time.monotonic()
        try:
            with connect(url, additional_headers=headers, open_timeout=15, max_size=16 * 2**20) as ws:
                serve_connection(ws, runner, rid, hostname, log)
            reason = "connection closed"
        except ConnectionClosed as exc:
            if exc.rcvd and exc.rcvd.code == 4401:
                raise AuthError("The server rejected the runner token (set runner_token in client.yaml or JARVIS_RUNNER_TOKEN)") from None
            reason = f"connection closed ({exc.rcvd.code if exc.rcvd else 'no close frame'})"
        except InvalidStatus as exc:
            if exc.response.status_code in (401, 403):
                raise AuthError(f"The server refused the runner (HTTP {exc.response.status_code})") from None
            reason = f"HTTP {exc.response.status_code}"
        except (OSError, TimeoutError, ConnectionError, ValueError) as exc:
            reason = str(exc) or exc.__class__.__name__
        if time.monotonic() - started > 60:
            delay = 1  # it was a healthy connection; retry quickly
        log(f"Runner disconnected from {server_url}: {reason}; retrying in {delay}s")
        time.sleep(delay)
        delay = min(delay * 2, 60)
