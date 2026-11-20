# Jarvis — instructions for AI assistants

This file is the single source of truth for every AI model or coding agent
working on this repository — Claude, GPT/Codex, Ollama-hosted models, or Jarvis
itself (Jarvis injects `AGENTS.md` from attached folders into the system prompt
of whichever backend is in use). `CLAUDE.md` only imports this file. Edit
rules **here**, never in a model-specific copy.

Do not guess. If something below does not answer a question, read the code
named here before changing anything; do not invent APIs, files or commands.

## What Jarvis is

A personal local-first agent for Ubuntu 24.04:

| Part | Path | Runs with |
|---|---|---|
| Backend API (FastAPI) | `local_agent/server.py` | project venv (`.venv`), systemd user service `jarvis.service`, `127.0.0.1:8765` |
| Core: config, SQLite, projects/chats, tools, agent loop | `local_agent/core.py` | venv |
| LLM backends | `local_agent/adapters.py` | venv |
| CLI (`jarvis …`, Typer) | `local_agent/cli.py` | venv |
| Native desktop app (GTK4 + libadwaita + WebKitGTK 6.0) | `local_agent/desktop.py` | **system** `/usr/bin/python3` with `python3-gi`; stdlib + `gi` only, never import the package or venv deps there |
| Web UI loaded by the desktop app | `web/index.html`, `web/theme.css`, `web/icons.js`, `web/app.js` | plain JS, no build step, no frameworks |
| Packaging | `packaging/` (`dev.jarvis.Jarvis.desktop.in`, `jarvis.service.in`, icons), `Makefile`, `scripts/` | |

Data: `~/.local/share/jarvis/agent.sqlite3`. Config: `~/.config/jarvis/config.yaml`
(see `config.example.yaml`). Secrets: `~/.config/jarvis/env` (service `EnvironmentFile`).

## Non-negotiable: the Tron theme

Before touching anything visual, read **[docs/THEME.md](docs/THEME.md)** and
follow it. In short: dark only, cyan = primary energy, orange = danger only,
thin glowing 1px lines, tight radii, icon-only buttons with labels/tooltips,
icons only from `web/icons.js`, styles only in `web/theme.css` via tokens,
dialogs only via `openDialog()` — never `alert/confirm/prompt`, never emoji.
No UI frameworks, component libraries or light theme. The GTK window chrome in
`desktop.py` (`WINDOW_CSS`) uses the same palette.

## Key mechanisms

* **Backends** — `adapters.py`. Register with `@register("type")`; implement
  `chat(messages, model, tools)` returning the internal assistant message
  (format documented at the top of the file) and optionally `list_models()`.
  Built-ins: `ollama`, `openai` (+ any OpenAI-compatible `base_url`), `anthropic`.
  Model ids are `provider/model`, providers are keys under `providers:` in config.
* **Agent loop** — `core.run_agent`. `list_files`/`read_file` run freely inside
  granted paths; `write_file`/`run_command` pause for user approval and resume
  via `core.approve`, threading `tool_call_id`s.
* **Grants** — folders *or files* attached to a chat (`folders` table) or a
  project (`project_folders`). `core.allowed_path` enforces them after resolving
  symlinks. Never widen this.
* **Projects** — chats in a project share its instructions, grants and recent
  sibling turns (`core.project_context`). Chats can be renamed, moved, deleted.
* **Workspace instructions** — `core.workspace_instructions` loads `AGENTS.md`
  / `CLAUDE.md` from attached folder roots into every backend's system prompt.
* **Desktop bridge** — `desktop.py` injects `window.jarvisDesktop.pick(kind,
  multiple)` → native GTK file dialog. In browser mode `web/app.js` falls back
  to `POST /api/pick` (zenity) and then to a typed path.

## Working rules

* Tests: `make test` (wraps `env -u PYTHONPATH` because ROS on this machine
  injects broken pytest plugins). Keep tests offline: mock `model_chat`/HTTP.
* Run backend: `make serve`. Desktop: `jarvis desktop` (`--browser` fallback).
* Keep `desktop.py` importable without GTK (it must print a helpful message).
* Do not add dependencies without updating `pyproject.toml` and `Makefile`
  provisioning; system packages go in the `provision` apt line.
* Security: the API is unauthenticated and bound to localhost; keep JSON bodies
  on state-changing endpoints (CORS preflight protection) and never bind to 0.0.0.0.
