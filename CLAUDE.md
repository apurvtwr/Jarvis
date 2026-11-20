# Jarvis — notes for AI assistants

Jarvis is a personal local-first agent: FastAPI backend + CLI (`local_agent/`),
a single-page desktop UI (`web/`), SQLite storage in `~/.local/share/jarvis`.

## Non-negotiable: the Tron theme

Before touching anything visual, read **[docs/THEME.md](docs/THEME.md)** and
follow it. In short: dark only, cyan = primary energy, orange = danger only,
thin glowing 1px lines, tight radii, icon-only buttons with labels/tooltips,
icons only from `web/icons.js`, styles only in `web/theme.css` via tokens,
dialogs only via `openDialog()` — never `alert/confirm/prompt`, never emoji.
Do not add UI frameworks, component libraries or a light theme.

## Architecture

* `local_agent/adapters.py` — LLM backends. Register a class with
  `@register("type")`; implement `chat(messages, model, tools)` (+ `list_models`).
  Internal message format is documented at the top of that file. Built-ins:
  `ollama`, `openai` (and OpenAI-compatible), `anthropic`.
* `local_agent/core.py` — config, DB, projects/chats, tools, routing, agent loop
  (`run_agent`, approvals for `write_file`/`run_command`).
* Model ids are `provider/model`; providers are configured in
  `~/.config/jarvis/config.yaml` (see `config.example.yaml`).
* Chats in a project share the project's instructions, folders and recent turns
  (`project_context`).

## Dev

`make test` (or `.venv/bin/pytest -q`), `make serve`. Keep tests offline — mock
`model_chat` / HTTP.
