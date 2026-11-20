from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from . import core

app = FastAPI(title="Jarvis", docs_url="/api/docs")
WEB = Path(__file__).resolve().parent.parent / "web" / "index.html"

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
class ModelRequest(BaseModel):
    model: str | None = None

def call(fn, *args):
    try:
        return fn(*args)
    except (ValueError, OSError) as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(502, f"Backend error: {exc}") from exc

@app.get("/")
def index():
    return FileResponse(WEB)

@app.get("/api/config")
def settings():
    cfg = core.config()
    return {"models": cfg["models"], "available_models": core.available_models(), "routing": cfg["routing"]}

@app.get("/api/chats")
def chats():
    with core.db() as con:
        return [dict(r) for r in con.execute("SELECT * FROM conversations ORDER BY created DESC")]

@app.post("/api/chats")
def create(req: NewRequest):
    return {"id": call(core.create_conversation, req.title, req.model)}

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
