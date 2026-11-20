"""Agent tool implementations shared by the server executor (core.py) and the runner
(runner.py). Nothing here touches the database or the server: the runner executes
these on the user's own machine with only this module and its local settings.

Every filesystem path goes through a ``check(path) -> Path`` callable supplied by the
caller, which must resolve symlinks and refuse anything outside the grants."""
import json
import os
import subprocess
from pathlib import Path

NAMES = ("list_files", "read_file", "write_file", "run_command")
LIMITS = {"max_file_bytes": 100000, "command_timeout_seconds": 30}


def grantable(path):
    """Resolve a user-chosen file or folder for access grants."""
    try:
        p = Path(path).expanduser().resolve(strict=True)
    except (OSError, RuntimeError):
        raise ValueError(f"Path does not exist: {path}") from None
    if not (p.is_dir() or p.is_file()):
        raise ValueError("Only regular files and folders can be attached")
    if p == Path(p.anchor):
        raise ValueError("Attaching the whole filesystem is not allowed")
    return p


def within(path, roots):
    """Resolve ``path`` (following symlinks) and require it to be inside one of ``roots``."""
    if not isinstance(path, str) or not path:
        raise ValueError("A path is required")
    p = Path(path).expanduser().resolve()
    if not any(p.is_relative_to(Path(root)) for root in roots):
        raise ValueError("Path is outside granted folders")
    return p


def execute(name, args, check, limits=LIMITS):
    max_bytes = int(limits.get("max_file_bytes", LIMITS["max_file_bytes"]))
    if not isinstance(args, dict):
        raise ValueError("Tool arguments must be an object")
    if name == "list_files":
        p = check(args.get("path"))
        if not p.is_dir():
            raise ValueError("Not a directory")
        return json.dumps([{"name": x.name, "directory": x.is_dir()} for x in list(p.iterdir())[:200]])
    if name == "read_file":
        p = check(args.get("path"))
        if not p.is_file():
            raise ValueError("Not a file")
        if p.stat().st_size > max_bytes:
            raise ValueError("File too large")
        return p.read_text(errors="replace")
    if name == "write_file":
        p = check(args.get("path"))
        content = args.get("content")
        if not isinstance(content, str):
            raise ValueError("content must be a string")
        if not p.parent.is_dir():
            raise ValueError("Parent directory does not exist")
        if len(content.encode()) > max_bytes:
            raise ValueError("Content too large")
        p.write_text(content)
        return f"Wrote {p}"
    if name == "run_command":
        cwd = check(args.get("cwd"))
        if not cwd.is_dir():
            raise ValueError("Working directory must exist")
        # Shell is deliberately disabled. Arguments are passed as an argv array.
        argv = args.get("argv")
        if not isinstance(argv, list) or not argv or not all(isinstance(x, str) for x in argv):
            raise ValueError("argv must be a nonempty string array")
        if Path(argv[0]).name in {"sudo", "su", "pkexec", "doas"}:
            raise ValueError("Privilege escalation is disabled")
        result = subprocess.run(argv, cwd=cwd, text=True, capture_output=True,
                                timeout=float(limits.get("command_timeout_seconds", LIMITS["command_timeout_seconds"])),
                                stdin=subprocess.DEVNULL, env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})
        return json.dumps({"exit_code": result.returncode, "stdout": result.stdout[:12000], "stderr": result.stderr[:12000]})
    raise ValueError("Unknown tool")
