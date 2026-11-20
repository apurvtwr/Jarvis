import json
import subprocess
import sys
import webbrowser
from pathlib import Path

import httpx
import typer
from rich.console import Console
from rich.table import Table
from . import core

app = typer.Typer(help="Jarvis AI agent; conversations shared with desktop UI")
console = Console()

@app.command()
def serve():
    """Start local desktop backend."""
    import uvicorn
    cfg = core.config()["server"]
    uvicorn.run("local_agent.server:app", host=cfg["host"], port=cfg["port"])

@app.command()
def desktop():
    """Open desktop interface (start `jarvis serve` first)."""
    cfg = core.config()["server"]
    url = f"http://{cfg['host']}:{cfg['port']}"
    for browser in ("chromium", "google-chrome", "chromium-browser"):
        from shutil import which
        if which(browser):
            subprocess.Popen([browser, f"--app={url}"])
            return
    webbrowser.open(url)

@app.command()
def new(title: str = "New chat", folder: list[Path] = typer.Option(None, "--folder", "-f"), model: str = typer.Option("auto", "--model")):
    """Create a conversation and grant folders."""
    cid = core.create_conversation(title, model)
    for path in folder or []:
        core.add_folder(cid, str(path))
    console.print(cid)

@app.command()
def chats():
    """List conversations."""
    table = Table("ID", "Title", "Created")
    with core.db() as con:
        for row in con.execute("SELECT * FROM conversations ORDER BY created DESC"):
            table.add_row(row["id"], row["title"], row["created"][:19])
    console.print(table)

@app.command()
def grant(chat_id: str, folder: Path):
    """Grant an existing conversation access to a folder."""
    console.print(core.add_folder(chat_id, str(folder)))

@app.command()
def model(chat_id: str, name: str = "auto"):
    """Pin a model to a conversation, or select auto routing."""
    selected = core.set_conversation_model(chat_id, name)
    console.print(selected or "Auto router enabled")

@app.command()
def models():
    """List models available through configured providers."""
    for name in core.available_models():
        console.print(name)

@app.command()
def chat(chat_id: str, model: str = "auto"):
    """Chat in terminal; type /quit to exit."""
    core.history(chat_id)
    console.print(f"[bold]Chat {chat_id}[/bold] | pinned model: {core.get_conversation(chat_id)['model_override'] or 'auto'}")
    while True:
        try:
            query = console.input("[cyan]You> [/cyan]").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if query in {"/quit", "/exit"}:
            break
        if not query:
            continue
        try:
            result = core.respond(chat_id, query, model)
            while result["status"] == "approval":
                console.print(f"[yellow]Approval requested: {result['tool']}[/yellow]")
                console.print_json(json.dumps(result["arguments"]))
                accepted = console.input("Approve? [y/N] ").strip().lower() == "y"
                result = core.approve(chat_id, result["id"], accepted)
            console.print(f"[green]{result['model']} >[/green] {result['content']}")
        except Exception as exc:
            console.print(f"[red]Error: {exc}[/red]")

@app.command()
def ask(prompt: str, chat_id: str = typer.Option(None, "--chat-id"), model: str = "auto"):
    """Ask once (will not automatically approve write/command tools)."""
    cid = chat_id or core.create_conversation(prompt[:60])
    result = core.respond(cid, prompt, model)
    console.print_json(json.dumps({"chat_id": cid, **result}))

if __name__ == "__main__":
    app()
