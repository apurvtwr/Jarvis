"""Jarvis HTTP API. The only interface clients (desktop app, web UI, CLI) use.

Designed to run either on the user's machine (default, 127.0.0.1) or on a remote
host behind TLS. See docs/ARCHITECTURE.md."""
import asyncio
import hmac
import ipaddress
import json
import os
import re
import socket
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from . import core

API_VERSION = 2
PUBLIC_API = {"/api/version", "/api/info"}


@asynccontextmanager
async def lifespan(_app):
    core.reset_interrupted_runs()
    yield


app = FastAPI(title="Jarvis", docs_url="/api/docs", lifespan=lifespan)
ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
ASSETS = {"theme.css": "text/css", "icons.js": "text/javascript", "app.js": "text/javascript"}

class ChatRequest(BaseModel):
    message: str
    model: str = "auto"
class FolderRequest(BaseModel):
    path: str
    runner_id: str | None = None   # with tools.executor: runner, the machine the path is on
class PickRequest(BaseModel):
    kind: str = "folder"
    multiple: bool = True
class ApprovalRequest(BaseModel):
    accept: bool
class NewRequest(BaseModel):
    title: str = "New chat"
    model: str | None = None
    project_id: str | None = None
class ModelRequest(BaseModel):
    model: str | None = None

class RenameRequest(BaseModel):
    title: str
class MoveRequest(BaseModel):
    project_id: str | None = None
class ProjectRequest(BaseModel):
    name: str
    context: str = ""
class ProjectUpdate(BaseModel):
    name: str | None = None
    context: str | None = None

def auth_token():
    return os.environ.get("JARVIS_AUTH_TOKEN") or core.config()["server"].get("auth_token") or None


def runner_token():
    """Runners may use their own token (narrower than the API token); either is accepted."""
    return os.environ.get("JARVIS_RUNNER_TOKEN") or core.config()["server"].get("runner_token") or None


def runner_authorized(header, client_host):
    tokens = [t for t in (runner_token(), auth_token()) if t]
    if not tokens:
        return is_loopback(client_host)  # no tokens configured: the server is loopback-only anyway
    return any(hmac.compare_digest(header.encode(), f"Bearer {t}".encode()) for t in tokens)


def is_loopback(host):
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host == "localhost"


@app.middleware("http")
async def require_token(request: Request, call_next):
    """Bearer-token auth for /api/* when a token is configured (mandatory off-localhost)."""
    token = auth_token()
    path = request.url.path
    if token and path.startswith("/api/") and path not in PUBLIC_API:
        supplied = request.headers.get("authorization", "")
        if not hmac.compare_digest(supplied.encode(), f"Bearer {token}".encode()):
            return JSONResponse({"detail": "Authentication required"}, status_code=401,
                                headers={"WWW-Authenticate": "Bearer"})
    return await call_next(request)


def call(fn, *args):
    try:
        return fn(*args)
    except (ValueError, OSError) as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(502, f"Backend error: {exc}") from exc

@app.get("/")
def index():
    return FileResponse(WEB / "index.html", headers={"Cache-Control": "no-cache"})

@app.get("/jarvis.svg")
def icon():
    return FileResponse(ROOT / "packaging" / "jarvis.svg", media_type="image/svg+xml")

@app.get("/web/{name}")
def asset(name: str):
    if name not in ASSETS:
        raise HTTPException(404, "Not Found")
    return FileResponse(WEB / name, media_type=ASSETS[name], headers={"Cache-Control": "no-cache"})

@app.get("/api/config")
def settings():
    cfg = core.config()
    profile = Path.home() / ".config" / "jarvis" / "profile.json"
    try:
        display_name = json.loads(profile.read_text()).get("display_name", "You")
    except (OSError, ValueError):
        display_name = "You"
    status = core.provider_status()
    return {"display_name": display_name, "models": cfg["models"], "available_models": core.available_models(status),
            "providers": status, "routing": cfg["routing"]}

@app.get("/api/chats")
def chats():
    return core.conversations()

@app.post("/api/chats")
def create(req: NewRequest):
    return {"id": call(core.create_conversation, req.title, req.model, req.project_id)}

@app.get("/api/chats/{cid}")
def detail(cid: str):
    return {"conversation": call(core.get_conversation, cid), "messages": call(core.history, cid), "folders": core.folders(cid), "pending": core.pending(cid)}

@app.post("/api/chats/{cid}/folders")
def folder(cid: str, req: FolderRequest):
    return {"path": call(core.add_folder, cid, req.path, req.runner_id)}

def start_turn(cid, fn, *args):
    """Turns can take minutes (LLM + tools). Run them in the background and return
    202 immediately so the request survives proxy/load-balancer timeouts; clients
    poll GET /api/chats/{cid} until conversation.run_status is no longer "running"."""
    call(core.begin_run, cid)

    def worker():
        try:
            core.run_turn(cid, fn, *args)
        except Exception:
            pass  # recorded on the conversation by run_turn
    threading.Thread(target=worker, name=f"turn-{cid[:8]}", daemon=True).start()
    return JSONResponse({"status": "running"}, status_code=202)


@app.post("/api/chats/{cid}/messages")
def message(cid: str, req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(400, "Message is empty")
    call(core.get_conversation, cid)
    if core.pending(cid):
        raise HTTPException(400, "Resolve the pending tool approval first")
    if req.model and req.model != "auto":
        call(core.normalize_model, req.model)
    return start_turn(cid, core.respond, req.message, req.model)

@app.post("/api/chats/{cid}/approvals/{aid}")
def approval(cid: str, aid: str, req: ApprovalRequest):
    pending = call(core.pending, cid)
    if not pending or pending["id"] != aid:
        raise HTTPException(400, "Unknown or resolved approval")
    return start_turn(cid, core.approve, aid, req.accept)

@app.put("/api/chats/{cid}/model")
def select_model(cid: str, req: ModelRequest):
    return {"model": call(core.set_conversation_model, cid, req.model)}

FEATURES = ["projects", "project-folders", "chat-renaming", "chat-delete", "project-delete", "providers",
            "themed-dialogs", "file-picker", "file-grants", "auth", "async-turns", "runners"]


@app.get("/api/version")
def version():
    return {"app": "jarvis", "api": API_VERSION, "features": FEATURES}


@app.get("/api/info")
def info():
    """Public capability document clients read before doing anything location-dependent."""
    ex = core.executor()
    runners = core.RUNNERS.list()
    if ex.location == "runner":  # tools run on the runner's machine, if exactly one is connected
        host = runners[0]["hostname"] if len(runners) == 1 else None
    else:
        host = socket.gethostname()
    return {"app": "jarvis", "api": API_VERSION, "auth_required": bool(auth_token()),
            "tools": {"executor": ex.location, "host": host,
                      "runners": [{"id": r["id"], "hostname": r["hostname"]} for r in runners]}}


@app.get("/api/runners")
def runners():
    return core.RUNNERS.list()


RUNNER_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


@app.websocket("/api/runners/connect")
async def runner_connect(ws: WebSocket):
    """Outbound connection from `jarvis runner` on the user's machine. Protocol (JSON):
    runner → {"type": "hello", "runner_id", "hostname"}; server → {"type": "welcome"};
    server → {"type": "call", "call_id", "tool", "args", "grants"};
    runner → {"type": "result", "call_id", "ok", "result" | "error"}."""
    await ws.accept()
    if not runner_authorized(ws.headers.get("authorization", ""), ws.client.host if ws.client else ""):
        await ws.close(code=4401, reason="Authentication required")
        return
    try:
        hello = await asyncio.wait_for(ws.receive_json(), timeout=15)
    except (asyncio.TimeoutError, WebSocketDisconnect, ValueError):
        await ws.close(code=4400, reason="Expected hello")
        return
    rid, hostname = hello.get("runner_id"), hello.get("hostname")
    if hello.get("type") != "hello" or not isinstance(rid, str) or not RUNNER_ID.match(rid) \
            or not isinstance(hostname, str) or not 0 < len(hostname) <= 255:
        await ws.close(code=4400, reason="Invalid hello")
        return
    loop = asyncio.get_running_loop()

    def send(msg):  # called from agent worker threads
        asyncio.run_coroutine_threadsafe(ws.send_json(msg), loop).result(timeout=15)

    def close():
        asyncio.run_coroutine_threadsafe(ws.close(code=4409, reason="Replaced by a newer connection"), loop)

    conn = core.RUNNERS.register(rid, hostname, send, close)
    try:
        await ws.send_json({"type": "welcome", "api": API_VERSION})
        while True:
            msg = await ws.receive_json()
            if isinstance(msg, dict) and msg.get("type") == "result":
                core.RUNNERS.deliver(conn, msg)
    except (WebSocketDisconnect, ValueError, RuntimeError):
        pass
    finally:
        core.RUNNERS.unregister(conn)

@app.get("/api/projects")
def list_projects():
    return core.projects()

@app.post("/api/projects")
def create_project(req: ProjectRequest):
    return {"id": call(core.create_project, req.name, req.context)}

@app.patch("/api/projects/{pid}")
def edit_project(pid: str, req: ProjectUpdate):
    call(core.update_project, pid, req.name, req.context)
    return core.project_detail(pid)

@app.post("/api/projects/{pid}/folders")
def project_folder(pid: str, req: FolderRequest):
    return {"path": call(core.add_project_folder, pid, req.path, req.runner_id)}

@app.patch("/api/chats/{cid}")
def rename_chat(cid: str, req: RenameRequest):
    call(core.rename_conversation, cid, req.title)
    return core.get_conversation(cid)

@app.put("/api/chats/{cid}/project")
def move_chat(cid: str, req: MoveRequest):
    call(core.move_conversation, cid, req.project_id)
    return core.get_conversation(cid)

@app.delete("/api/chats/{cid}")
def delete_chat(cid: str):
    call(core.delete_conversation, cid)
    return {"deleted": cid}

@app.delete("/api/projects/{pid}")
def delete_project(pid: str, delete_chats: bool = False):
    return {"deleted": pid, "chats": call(core.delete_project, pid, delete_chats)}

@app.get("/api/providers")
def providers():
    return core.provider_status()

@app.post("/api/pick")
def pick(req: PickRequest, request: Request):
    """Fallback native picker for browser mode (the desktop app uses its own GTK dialog).
    Only for a browser on the server's own machine: a remote server's display is not
    the user's. JSON-only body, so browsers block cross-site calls via CORS preflight."""
    import shutil
    import subprocess
    if not is_loopback(request.client.host if request.client else ""):
        raise HTTPException(403, "The file picker is only available when Jarvis runs on this computer")
    if not shutil.which("zenity") or not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        raise HTTPException(501, "No graphical file picker is available to the Jarvis service")
    args = ["zenity", "--file-selection", "--separator=\n", "--filename=" + str(Path.home()) + "/",
            "--title=" + ("Attach folder" if req.kind == "folder" else "Attach files")]
    if req.kind == "folder":
        args.append("--directory")
    if req.multiple:
        args.append("--multiple")
    result = subprocess.run(args, capture_output=True, text=True)
    return {"paths": [line for line in result.stdout.splitlines() if line.strip()] if result.returncode == 0 else []}
