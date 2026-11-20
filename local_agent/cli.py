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
    # Prefer an isolated browser app window with its own desktop identity.
    import os
    import time
    from urllib.request import urlopen
    from shutil import which
    try:
        with urlopen(url, timeout=1) as response:
            if response.status >= 400:
                raise RuntimeError(f"Backend returned {response.status}")
    except Exception:
        subprocess.run(["systemctl", "--user", "start", "jarvis.service"], check=False)
        for _ in range(30):
            try:
                with urlopen(url, timeout=1):
                    break
            except Exception:
                time.sleep(0.3)
        else:
            typer.echo("Jarvis backend is unavailable; inspect journalctl --user -u jarvis", err=True)
            raise typer.Exit(code=1)
    for browser in ("chromium", "google-chrome", "chromium-browser", "google-chrome-stable", "brave-browser"):

        from shutil import which
        if which(browser):
            subprocess.Popen([browser, f"--app={url}", "--ozone-platform=x11", "--class=jarvis-app", "--name=jarvis-app", "--user-data-dir=" + str(Path.home() / ".local/share/jarvis/browser-profile")])
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
    table = Table("ID", "Title", "Project", "Created")
    names = {p["id"]: p["name"] for p in core.projects()}
    for row in core.conversations():
        table.add_row(row["id"], row["title"], names.get(row["project_id"], ""), row["created"][:19])
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
def rename(chat_id: str, title: str):
    """Rename a conversation."""
    core.rename_conversation(chat_id, title)
    console.print(title)

@app.command()
def delete(chat_id: str, yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation")):
    """Permanently delete a conversation and its history."""
    title = core.get_conversation(chat_id)["title"]
    if not yes and not typer.confirm(f"Delete '{title}' permanently?"):
        raise typer.Exit()
    core.delete_conversation(chat_id)
    console.print(f"Deleted {title}")

@app.command()
def move(chat_id: str, project_id: str = typer.Argument(None, help="Target project id; omit to remove from its project")):
    """Move a conversation into a project (or out of one)."""
    core.move_conversation(chat_id, project_id)
    console.print(project_id or "No project")

@app.command()
def projects():
    """List projects."""
    table = Table("ID", "Name", "Chats")
    chats = core.conversations()
    for p in core.projects():
        table.add_row(p["id"], p["name"], str(sum(c["project_id"] == p["id"] for c in chats)))
    console.print(table)

@app.command()
def project(name: str, context: str = typer.Option("", "--context", help="Instructions shared by the project's chats")):
    """Create a project."""
    console.print(core.create_project(name, context))

@app.command()
def providers():
    """Show configured LLM backends and whether they are reachable."""
    table = Table("Provider", "Type", "Status", "Models")
    for item in core.provider_status():
        status = "[green]ok[/green]" if item["ok"] else f"[red]{item['error'][:60]}[/red]"
        table.add_row(item["name"], item["type"] or "?", status, str(len(item["models"])))
    console.print(table)

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
