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
- Desktop folder selection currently uses an absolute-path prompt, not a graphical folder picker.

## Development

```bash
make provision
make serve
make test
```

The desktop is a lightweight HTML frontend served by FastAPI. CLI and desktop both call the same Python core, with SQLite in WAL mode. No cloud account is needed.

## Uninstall

`make uninstall` removes the desktop launcher, CLI symlink, and service but preserves config and conversations.

## Provider adapters and per-conversation models

Model identifiers use `provider/model-name`, for example `ollama/qwen3:8b`. The provider adapter owns HTTP transport and message formatting; the model name is independent of the adapter implementation. To support a new provider protocol, implement an adapter with `chat(messages, model, tools=None)` and register it in `ADAPTERS` in `local_agent/core.py`. Additional Ollama-compatible servers can be added under `providers` with `type: ollama` and a distinct provider key.

Use `jarvis models` to list locally discoverable models, `jarvis new "My task" --model ollama/qwen2.5-coder:7b` to pin a new conversation, and `jarvis model CHAT_ID auto` to restore automatic routing. In the desktop UI choose the model in the chat header. The selection is stored in SQLite and shared by the CLI and desktop. When pinned, the router is never called for that conversation.

Existing installations: rename the old `~/.config/local-agent/config.yaml` to `~/.config/jarvis/config.yaml` and `~/.local/share/local-agent` to `~/.local/share/jarvis` to retain settings and history; adapt old `ollama_url` to the new `providers` format. Stop and uninstall the old local-agent user service before installing Jarvis to avoid port conflicts.
