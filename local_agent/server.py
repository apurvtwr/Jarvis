from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
import json
from pydantic import BaseModel
from . import core

app = FastAPI(title="Jarvis", docs_url="/api/docs")
ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
ASSETS = {"theme.css": "text/css", "icons.js": "text/javascript", "app.js": "text/javascript"}

class ChatRequest(BaseModel):
    message: str
    model: str = "auto"
class FolderRequest(BaseModel):
    path: str
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

def call(fn, *args):
    try:
        return fn(*args)
    except (ValueError, OSError) as exc:
        raise HTTPException(400, str(exc)) from exc
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
    return {"path": call(core.add_folder, cid, req.path)}

@app.post("/api/chats/{cid}/messages")
def message(cid: str, req: ChatRequest):
    return call(core.respond, cid, req.message, req.model)

@app.post("/api/chats/{cid}/approvals/{aid}")
def approval(cid: str, aid: str, req: ApprovalRequest):
    return call(core.approve, cid, aid, req.accept)

@app.put("/api/chats/{cid}/model")
def select_model(cid: str, req: ModelRequest):
    return {"model": call(core.set_conversation_model, cid, req.model)}

@app.get("/api/version")
def version():
    return {"app": "jarvis", "features": ["projects", "project-folders", "chat-renaming", "chat-delete", "project-delete", "providers", "themed-dialogs"]}

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
    return {"path": call(core.add_project_folder, pid, req.path)}

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
