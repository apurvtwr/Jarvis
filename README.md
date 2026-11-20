# Jarvis

A local-first Ollama agent with a desktop UI, terminal CLI, model routing, per-conversation folder grants, tool approval, and shared SQLite history.

## Ubuntu 24.04 quick start

Requirements: Python 3.12 + `python3-venv`, Make, running Ollama, and locally installed models.

```bash
sudo apt update && sudo apt install -y python3-venv make
ollama pull qwen3:8b
ollama pull qwen2.5-coder:7b
make install
jarvis desktop
```

If `~/.local/bin` isn't on your PATH, open a new terminal or use `~/.local/bin/jarvis`.

`make install` creates a user-level systemd service and a desktop launcher. It does not require sudo except for optional apt prerequisites. The backend listens only on `127.0.0.1:8765` by default.

## CLI

```bash
jarvis new "Review my ROS code" --folder ~/Documents/M31
jarvis chats
jarvis chat <conversation-id>
jarvis ask "Explain PID control"
jarvis grant <conversation-id> ~/other-project
jarvis desktop
```

Desktop and CLI read/write the same SQLite database (`~/.local/share/jarvis/agent.sqlite3`). Desktop chats appear in the terminal and vice versa. Launch multiple desktop windows with `jarvis desktop`.

## Configuration

Edit `~/.config/jarvis/config.yaml`:

```yaml
ollama_url: http://127.0.0.1:11434
models:
  general: qwen3:8b
  code: qwen2.5-coder:7b
  router: qwen3:8b
routing:
  mode: heuristic # or llm
```

`heuristic` is a fast keyword router. `llm` asks the router model to classify requests as code/general. Choose a model per message in the desktop dropdown or `jarvis chat <id> --model code`.

Restart the service after changing settings: `systemctl --user restart jarvis.service`.

## Tool permissions and security

- Chats have **no file access** until a folder is granted.
- Read/list tools run automatically inside granted folders.
- Write and command tools require approval **once per invocation** in the active UI.
- Paths are canonicalized, including symlinks, and checked against grants.
- Shell commands run as your normal user without a shell, with a timeout.
- **Important:** approved commands are *not sandboxed*. An approved executable can access anything your Linux account can access, regardless of granted folders. Inspect every proposed command before approving it. Do not expose the backend to the network; it has no authentication.
- There is no streaming, cancellation, or per-conversation concurrent-turn coordination in this MVP. Avoid sending simultaneous requests to the same chat from different interfaces.
- If the backend is interrupted while an LLM call is running, the last request may need to be retried. Chat messages persist; intermediate read-tool calls are not yet saved as separate audit events.
- Attach files or folders with the native GNOME picker (paperclip icon). Grants can be whole folders or individual files.

## Development

```bash
make provision
make serve
make test
```

The desktop is a lightweight HTML frontend served by FastAPI. CLI and desktop both call the same Python core, with SQLite in WAL mode. No cloud account is needed.

## Uninstall

`make uninstall` removes the desktop launcher, CLI symlink, and service but preserves config and conversations.

## LLM backends (Ollama, OpenAI, Claude, …)

Model identifiers use `provider/model-name`, e.g. `ollama/qwen3:8b`, `openai/gpt-5`,
`anthropic/claude-sonnet-5-5`. Each provider in `~/.config/jarvis/config.yaml` names an
adapter `type`:

| type | backend |
|---|---|
| `ollama` | Local or remote Ollama |
| `openai` | OpenAI, or any OpenAI-compatible server via `base_url` (LM Studio, vLLM, llama.cpp, OpenRouter…) |
| `anthropic` | Anthropic Messages API (Claude) |

See `config.example.yaml` for ready-to-uncomment entries. API keys come from environment
variables (`api_key_env`, default `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`). For the desktop
service put them in `~/.config/jarvis/env` (`chmod 600`), then re-run `make install` once so
the service picks up the `EnvironmentFile`, and `systemctl --user restart jarvis.service`.
`jarvis providers` shows which backends are reachable; the sidebar status light turns
orange when one is offline.

To add a new protocol, subclass `Adapter` in `local_agent/adapters.py`, implement
`chat(messages, model, tools=None)` (and optionally `list_models()`), and decorate it with
`@register("my-type")`. The internal message format is documented at the top of that file.

## Projects and chats

Chats can be renamed, deleted, and moved between projects (header icons, sidebar hover
icons, or drag a chat onto a project). Chats inside a project share the project's
instructions, folder grants, and recent turns from sibling chats (tunable under
`projects:` in the config). CLI: `jarvis rename`, `jarvis delete`, `jarvis move`,
`jarvis projects`, `jarvis project NAME`.

## UI theme

The UI follows a binding Tron theme documented in `docs/THEME.md`; styles live only in
`web/theme.css` and icons only in `web/icons.js`.

Use `jarvis models` to list locally discoverable models, `jarvis new "My task" --model ollama/qwen2.5-coder:7b` to pin a new conversation, and `jarvis model CHAT_ID auto` to restore automatic routing. In the desktop UI choose the model in the chat header. The selection is stored in SQLite and shared by the CLI and desktop. When pinned, the router is never called for that conversation.

Existing installations: rename the old `~/.config/local-agent/config.yaml` to `~/.config/jarvis/config.yaml` and `~/.local/share/local-agent` to `~/.local/share/jarvis` to retain settings and history; adapt old `ollama_url` to the new `providers` format. Stop and uninstall the old local-agent user service before installing Jarvis to avoid port conflicts.

### Complete Ubuntu provisioning

`make provision` installs Ubuntu Python prerequisites, creates the Python environment,
installs Jarvis dependencies, creates the user configuration (if absent), installs
Ollama from its official installer if missing, starts the Ollama service when needed,
and pulls missing models referenced in `models` in `~/.config/jarvis/config.yaml`.
`make install` invokes `make provision` before creating desktop and CLI launchers.
`make update-models` pulls newly configured models; `make check` checks availability
without installing anything. Provisioning can require sudo and several gigabytes of
model downloads. The official Ollama installer is a third-party script; review it
before provisioning on a managed machine. Non-Ollama provider types are not
provisioned automatically. Remote Ollama servers are never installed or modified.

### Native desktop app (Ubuntu 24.04)

Jarvis ships a real GNOME app: a GTK4 + libadwaita window hosting the UI in
WebKitGTK (`local_agent/desktop.py`), not a browser tab. It runs with the system
Python and these packages, which `make provision` installs:

```bash
sudo apt install python3-gi gir1.2-gtk-4.0 gir1.2-adw-1 gir1.2-webkit-6.0 zenity
```

`make install` registers it as `dev.jarvis.Jarvis` in the application launcher.
Open Activities, search **Jarvis**, launch it, then right-click its dock icon and choose
**Pin to Dash**. It is single-instance (launching again raises the window), starts the
backend service if needed, remembers its window size, opens external links in your
browser, and uses the native file chooser for attaching files and folders.
Shortcuts: `Ctrl+N` new chat, `Ctrl+B` sidebar, `Ctrl+R` reload, `F11` fullscreen, `Ctrl+Q` quit.

`jarvis desktop` launches the same app from a terminal; `jarvis desktop --browser`
(also in the launcher's right-click menu) opens the old browser window instead. On
NVIDIA GPUs the app sets `WEBKIT_DISABLE_DMABUF_RENDERER=1` to avoid a blank window.

### Running the server elsewhere (cloud-ready)

Jarvis is client/server; the desktop app and CLI only use the HTTP API. To use a
server on another machine, create `~/.config/jarvis/client.yaml` on your computer
(see `packaging/client.example.yaml`):

```yaml
url: https://jarvis.example.com
token_env: JARVIS_TOKEN
```

On the server, set `JARVIS_AUTH_TOKEN`, put a TLS reverse proxy in front, and set
`server.host`. `jarvis serve` refuses non-local binds without a token. `jarvis status`
shows which server the CLI is talking to. Agent tools run on the server's machine by
default. To keep files on your computer with a cloud server, set `tools.executor: runner`
and `JARVIS_RUNNER_TOKEN` on the server, add `runner_token` and `runner.allow` to
`client.yaml` (see `packaging/client.example.yaml`), and run `jarvis runner` (or
`systemctl --user enable --now jarvis-runner.service`). Details in `docs/ARCHITECTURE.md`.

### Instructions for AI models

`AGENTS.md` holds the rules for any AI working on this repo (Claude, GPT/Codex,
Ollama models); `CLAUDE.md` just imports it. Jarvis itself loads `AGENTS.md` /
`CLAUDE.md` from the root of attached folders into the system prompt for whichever
backend is active, so every model gets the same project instructions.

### Jarvis identity and desktop icon

`make install` prompts for a chat display name on first setup and stores it in
`~/.config/jarvis/profile.json`. You can edit `display_name` later. The UI
labels assistant messages **JARVIS** regardless of the selected underlying
model. The cyan atom icon is used in the app, favicon, desktop launcher and
loading animation. The desktop launcher uses a dedicated Chromium application
window and XWayland WM_CLASS for more reliable GNOME dock grouping. If an old
Jarvis dock favorite still displays the previous icon, remove it from Favorites,
launch Jarvis again from Applications, then pin the new window.

Messages are displayed immediately while Jarvis processes them; a rotating
atom indicates that the request is still in flight. The animation respects
reduced-motion preferences.
