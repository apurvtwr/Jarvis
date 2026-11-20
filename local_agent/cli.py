"""Jarvis CLI. Every command except `serve` talks to the backend over HTTP
(local or remote; see local_agent/client.py for how the target is chosen)."""
import ipaddress
import json
import os
import subprocess
import time
import webbrowser
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .client import ApiError, JarvisClient, client_settings, is_local_url

app = typer.Typer(help="Jarvis AI agent. Talks to a local or remote Jarvis server.")
console = Console()

SYSTEM_PYTHON = "/usr/bin/python3"
DESKTOP_SHELL = Path(__file__).resolve().with_name("desktop.py")


def api():
    return JarvisClient()


def fail(exc):
    console.print(f"[red]Error: {exc}[/red]")
    raise typer.Exit(code=1)


def grant_path(client, path):
    # A local server shares this machine's filesystem, so send an absolute path.
    # A remote server resolves paths on its own host (see docs/ARCHITECTURE.md).
    return str(Path(path).expanduser().resolve()) if client.local else str(path)


@app.command()
def serve():
    """Run the Jarvis server (what the jarvis.service user unit runs)."""
    import uvicorn
    from . import core
    from .server import auth_token
    cfg = core.config()["server"]
    try:
        loopback = cfg["host"] == "localhost" or ipaddress.ip_address(cfg["host"]).is_loopback
    except ValueError:
        loopback = False
    if not loopback and not auth_token():
        console.print("[red]Refusing to listen on a non-loopback address without an auth token.[/red]\n"
                      "Set JARVIS_AUTH_TOKEN (or server.auth_token) and put Jarvis behind TLS. See docs/ARCHITECTURE.md.")
        raise typer.Exit(code=2)
    uvicorn.run("local_agent.server:app", host=cfg["host"], port=cfg["port"],
                proxy_headers=not loopback, forwarded_allow_ips="*" if not loopback else None)


def desktop_env():
    # The native shell uses the system PyGObject; keep ROS/venv paths out of it.
    return {k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"}}


@app.command()
def desktop(browser: bool = typer.Option(False, "--browser", help="Open in a browser window instead of the native app")):
    """Open the Jarvis desktop app (starts the local backend service if needed)."""
    url = client_settings()["url"]
    if not browser and Path(SYSTEM_PYTHON).exists():
        check = subprocess.run([SYSTEM_PYTHON, str(DESKTOP_SHELL), "--check"], env=desktop_env())
        if check.returncode == 0:
            # Exec the script itself (not python3 script) so its AppArmor profile attaches.
            os.execve(str(DESKTOP_SHELL), [str(DESKTOP_SHELL), "--url", url], desktop_env())
        typer.echo("Native desktop libraries missing; falling back to a browser window.\n"
                   "Install them with: sudo apt install python3-gi gir1.2-gtk-4.0 gir1.2-adw-1 gir1.2-webkit-6.0", err=True)
    open_browser_window(url)


def open_browser_window(url):
    from shutil import which
    if is_local_url(url):
        client = JarvisClient(url=url)
        try:
            client.get("/version")
        except ApiError:
            subprocess.run(["systemctl", "--user", "start", "jarvis.service"], check=False)
            for _ in range(30):
                try:
                    client.get("/version")
                    break
                except ApiError:
                    time.sleep(0.3)
            else:
                typer.echo("Jarvis backend is unavailable; inspect journalctl --user -u jarvis", err=True)
                raise typer.Exit(code=1)
    for name in ("chromium", "google-chrome", "chromium-browser", "google-chrome-stable", "brave-browser"):
        if which(name):
            subprocess.Popen([name, f"--app={url}", "--class=jarvis-app", "--name=jarvis-app",
                              "--user-data-dir=" + str(Path.home() / ".local/share/jarvis/browser-profile")])
            return
    webbrowser.open(url)


@app.command()
def status():
    """Show which Jarvis server this CLI talks to and what it can do."""
    client = api()
    try:
        info = client.get("/info")
        client.get("/chats")  # verifies the token
    except ApiError as exc:
        console.print(f"Server: {client.url}")
        fail(exc)
    console.print(f"Server: {client.url} ({'local' if client.local else 'remote'})")
    console.print(f"Auth: {'token' if info['auth_required'] else 'none (local only)'}")
    console.print(f"Tools run on: {info['tools']['executor']} ({info['tools']['host']})")


@app.command()
def new(title: str = "New chat", folder: list[Path] = typer.Option(None, "--folder", "-f"), model: str = typer.Option("auto", "--model")):
    """Create a conversation and grant folders."""
    client = api()
    try:
        cid = client.post("/chats", {"title": title, "model": None if model == "auto" else model})["id"]
        for path in folder or []:
            client.post(f"/chats/{cid}/folders", {"path": grant_path(client, path)})
    except ApiError as exc:
        fail(exc)
    console.print(cid)


@app.command()
def chats():
    """List conversations."""
    client = api()
    try:
        names = {p["id"]: p["name"] for p in client.get("/projects")}
        rows = client.get("/chats")
    except ApiError as exc:
        fail(exc)
    table = Table("ID", "Title", "Project", "Created")
    for row in rows:
        table.add_row(row["id"], row["title"], names.get(row["project_id"], ""), row["created"][:19])
    console.print(table)


@app.command()
def grant(chat_id: str, folder: Path):
    """Grant an existing conversation access to a file or folder."""
    client = api()
    try:
        console.print(client.post(f"/chats/{chat_id}/folders", {"path": grant_path(client, folder)})["path"])
    except ApiError as exc:
        fail(exc)


@app.command()
def model(chat_id: str, name: str = "auto"):
    """Pin a model to a conversation, or select auto routing."""
    try:
        selected = api().put(f"/chats/{chat_id}/model", {"model": name})["model"]
    except ApiError as exc:
        fail(exc)
    console.print(selected or "Auto router enabled")


@app.command()
def models():
    """List models available through configured providers."""
    try:
        for name in api().get("/config")["available_models"]:
            console.print(name)
    except ApiError as exc:
        fail(exc)


@app.command()
def rename(chat_id: str, title: str):
    """Rename a conversation."""
    try:
        api().patch(f"/chats/{chat_id}", {"title": title})
    except ApiError as exc:
        fail(exc)
    console.print(title)


@app.command()
def delete(chat_id: str, yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation")):
    """Permanently delete a conversation and its history."""
    client = api()
    try:
        title = client.get(f"/chats/{chat_id}")["conversation"]["title"]
        if not yes and not typer.confirm(f"Delete '{title}' permanently?"):
            raise typer.Exit()
        client.delete(f"/chats/{chat_id}")
    except ApiError as exc:
        fail(exc)
    console.print(f"Deleted {title}")


@app.command()
def move(chat_id: str, project_id: str = typer.Argument(None, help="Target project id; omit to remove from its project")):
    """Move a conversation into a project (or out of one)."""
    try:
        api().put(f"/chats/{chat_id}/project", {"project_id": project_id})
    except ApiError as exc:
        fail(exc)
    console.print(project_id or "No project")


@app.command()
def projects():
    """List projects."""
    client = api()
    try:
        rows, chats_ = client.get("/projects"), client.get("/chats")
    except ApiError as exc:
        fail(exc)
    table = Table("ID", "Name", "Chats")
    for p in rows:
        table.add_row(p["id"], p["name"], str(sum(c["project_id"] == p["id"] for c in chats_)))
    console.print(table)


@app.command()
def project(name: str, context: str = typer.Option("", "--context", help="Instructions shared by the project's chats")):
    """Create a project."""
    try:
        console.print(api().post("/projects", {"name": name, "context": context})["id"])
    except ApiError as exc:
        fail(exc)


@app.command()
def providers():
    """Show configured LLM backends and whether they are reachable."""
    try:
        rows = api().get("/providers")
    except ApiError as exc:
        fail(exc)
    table = Table("Provider", "Type", "Status", "Models")
    for item in rows:
        state = "[green]ok[/green]" if item["ok"] else f"[red]{item.get('error', '')[:60]}[/red]"
        table.add_row(item["name"], item["type"] or "?", state, str(len(item["models"])))
    console.print(table)


def run_turn(client, cid, send):
    """Send a message (or approval), then handle approvals until the turn completes."""
    with console.status("Jarvis is working…"):
        detail = send()
    while detail.get("pending"):
        pending = detail["pending"]
        console.print(f"[yellow]Approval requested: {pending['tool']}[/yellow]")
        console.print_json(json.dumps(json.loads(pending["arguments"])["args"]))
        accepted = console.input("Approve? [y/N] ").strip().lower() == "y"
        with console.status("Jarvis is working…"):
            detail = client.approve(cid, pending["id"], accepted)
    reply = next((m for m in reversed(detail["messages"]) if m["role"] == "assistant"), None)
    return detail, reply


@app.command()
def chat(chat_id: str, model: str = "auto"):
    """Chat in terminal; type /quit to exit."""
    client = api()
    try:
        conv = client.get(f"/chats/{chat_id}")["conversation"]
    except ApiError as exc:
        fail(exc)
    console.print(f"[bold]{conv['title']}[/bold] | pinned model: {conv['model_override'] or 'auto'} | {client.url}")
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
            _, reply = run_turn(client, chat_id, lambda: client.send(chat_id, query, model))
            if reply:
                console.print(f"[green]{reply['model']} >[/green] {reply['content']}")
        except ApiError as exc:
            console.print(f"[red]Error: {exc}[/red]")


@app.command()
def ask(prompt: str, chat_id: str = typer.Option(None, "--chat-id"), model: str = "auto"):
    """Ask once and print the reply as JSON."""
    client = api()
    try:
        cid = chat_id or client.post("/chats", {"title": prompt[:60]})["id"]
        detail, reply = run_turn(client, cid, lambda: client.send(cid, prompt, model))
    except ApiError as exc:
        fail(exc)
    console.print_json(json.dumps({"chat_id": cid, "model": reply and reply["model"], "content": reply and reply["content"]}))


if __name__ == "__main__":
    app()
