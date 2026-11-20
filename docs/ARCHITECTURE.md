# Jarvis architecture: local today, cloud-ready

Jarvis is a **client/server** system. The server can run on the user's own
computer (the default) or on a remote host. Nothing in a client may assume the
server shares its machine, filesystem, display or process manager.

```
 ┌──────────── user's computer ────────────┐          ┌──────── server (local or cloud) ────────┐
 │ Desktop app (GTK/WebKit, desktop.py)    │  HTTPS   │ FastAPI  (server.py)                     │
 │ Browser                                 │ ───────► │  auth · chats · projects · run state     │
 │ CLI (cli.py → client.py)                │  Bearer  │ Core     (core.py)                       │
 │                                         │  token   │  agent loop · routing · SQLite           │
 │ Runner (jarvis runner → runner.py):     │ ◄─────── │ Adapters (adapters.py) → Ollama/OpenAI/  │
 │   executes tools on this machine's files│  (WSS)   │                          Anthropic       │
 └─────────────────────────────────────────┘          └──────────────────────────────────────────┘
```

## Rules

1. **One interface.** Desktop app, web UI and CLI use only the HTTP API.
   Only `jarvis serve` imports `core`. Never add a client feature that reads the
   SQLite file or calls `core` directly.
2. **Client config is separate from server config.** Where to connect lives in
   `~/.config/jarvis/client.yaml` (`url`, `token` / `token_env`) or `JARVIS_URL` /
   `JARVIS_TOKEN`. With nothing set, clients use the local server from `config.yaml`.
3. **Local-only conveniences are gated on a loopback URL.** Starting
   `jarvis.service`, `journalctl` hints and the server-side zenity picker apply only
   when the server is on `127.0.0.1`/`localhost`/`::1`. `/api/pick` refuses non-loopback clients.
4. **Auth.** If `JARVIS_AUTH_TOKEN` (or `server.auth_token`) is set, every `/api/*`
   route except `/api/version` and `/api/info` requires `Authorization: Bearer <token>`.
   `jarvis serve` refuses to bind a non-loopback address without a token. Off-machine
   traffic must use TLS (reverse proxy such as Caddy/nginx in front of uvicorn;
   `serve` enables proxy headers when not on loopback).
5. **Turns are asynchronous.** `POST /chats/{id}/messages` and
   `POST /chats/{id}/approvals/{aid}` return `202 {"status": "running"}` immediately.
   Clients poll `GET /chats/{id}` until `conversation.run_status` is `idle` or `error`
   (`run_error` holds the message). Only one turn per chat runs at a time (409 otherwise).
   Runs interrupted by a server restart are marked `error` on startup. This keeps
   minutes-long LLM/tool turns working behind proxies with 60–100 s request timeouts.
6. **Capabilities are discovered, not assumed.** `GET /api/info` (public) returns
   `auth_required` and `tools: {executor, host, runners: [{id, hostname}]}`. Clients
   decide from it whether a local file picker makes sense.

## Where tools run (the important cloud question)

Agent tools (`list_files`, `read_file`, `write_file`, `run_command`) touch files.
`core.EXECUTORS` maps `tools.executor` in the server config to the class that runs them:

| executor | status | runs tools on | right when |
|---|---|---|---|
| `server` | implemented (`ServerExecutor`) | the machine running the backend | the backend runs on the user's computer, or the user wants a cloud workspace |
| `runner` | implemented (`RunnerExecutor` + `jarvis runner`) | the user's computer, via a small runner process | the backend runs in the cloud but files stay local |

Grants (`folders`, `project_folders`) are paths **on the machine where the executor
runs**. With `server`, the desktop app compares `/api/info` `tools.host` to its own
hostname: native browsing is offered only when they match. With `runner`, browsing is
offered when a runner with this computer's hostname is connected, and the grant is sent
with that `runner_id`. Otherwise the user types a path that exists on that host.

Tool implementations live in `local_agent/tools.py` (no DB, no server imports), shared
by `ServerExecutor` and the runner.

### Runner (`tools.executor: runner`)

* **Connection.** `jarvis runner` (`cli.py` → `runner.py`) opens an **outbound**
  WebSocket to `wss://server/api/runners/connect` (works through NAT/firewalls; nothing
  listens locally). It sends `Authorization: Bearer <runner token>`; the server accepts
  `JARVIS_RUNNER_TOKEN` / `server.runner_token` (a narrower secret: it can only connect
  a runner) or the API token. With no token configured at all, only loopback runners are
  accepted. Plain `ws://` is refused unless the server URL is loopback. A rejected token
  closes with code 4401 and the runner exits with status 2 (the unit does not restart).
  Other failures reconnect with backoff (1 s → 60 s).
* **Identity.** A stable `runner_id` (`~/.config/jarvis/runner-id`, created on first
  run) plus the hostname, sent in `hello`. A reconnect with the same id replaces the
  old socket. The registry (`core.RUNNERS`) is in memory; runners re-register after a
  server restart.
* **Protocol** (JSON text frames): runner `{"type":"hello","runner_id","hostname"}` →
  server `{"type":"welcome"}`; server `{"type":"call","call_id","tool","args","grants"}` →
  runner `{"type":"result","call_id","ok","result"|"error"}`. `tool` is one of the agent
  tools or `resolve` (validate a path before it is stored as a grant). Calls run
  concurrently on the runner (thread pool). Only the connection a call was sent to may
  answer it.
* **Grants.** `folders` / `project_folders` have a `runner_id` column (NULL = server
  grant). With `executor: runner`, attaching a path asks the runner to `resolve` it
  (body `runner_id`, or the only connected runner) and stores the resolved path with
  that runner. Grants made before switching to the runner must be attached again.
* **Routing.** `RunnerExecutor.execute` requires an absolute path, matches it lexically
  against the chat's grants (the server cannot see the runner's filesystem), picks the
  runner of the longest matching grant, and sends that runner's grant paths with the call.
  Workspace instructions (`AGENTS.md`/`CLAUDE.md`) are read through the executor too.
* **Local enforcement (do not trust the server).** The runner resolves every path
  (symlinks followed, same as `core.allowed_path`) and requires it to be inside the
  grants sent with the call **and** inside its own `runner.allow` roots (client.yaml,
  default `~`) **and** outside `runner.deny` (default: `~/.ssh`, `~/.gnupg`,
  `~/.config/jarvis`, keyrings, cloud credentials). A compromised server can lie about
  grants but cannot leave `allow`/`deny`. `run_command` is code execution inside an
  allowed folder; set `runner.allow_commands: false` to disable it locally, and narrow
  `runner.allow` to specific project folders for the strongest boundary.
* **Timeouts.** Each call waits `tools.runner_timeout_seconds` (default 60; keep it above
  `command_timeout_seconds`). Timeout, disconnect or a missing runner becomes a
  `Tool error: …` the model sees; the turn continues.
* **Approvals** stay on the server's run state; the runner never decides on its own.
* **Service.** `make install` installs `jarvis-runner.service` (user unit, not enabled):
  `systemctl --user enable --now jarvis-runner.service`. It reads client.yaml and
  optionally `~/.config/jarvis/runner.env`.
* The desktop app may host the runner in-process later, so a single launch does both.

## Deployment checklist (cloud)

- Set `server.host: 0.0.0.0` and `JARVIS_AUTH_TOKEN` (long random string) in the
  service environment; put a TLS reverse proxy in front.
- Configure providers on the server (API keys via env). Ollama must be reachable
  from the server (`base_url`), not from the client.
- On each client: `client.yaml` with `url: https://…` and the token (`chmod 600`).
- Choose `tools.executor`: `server` for a cloud workspace; `runner` for local files
  (set `JARVIS_RUNNER_TOKEN` on the server, `runner_token` + `runner.allow` in each
  client's client.yaml, then enable `jarvis-runner.service` there).
- SQLite is fine for one user; multi-user would need per-user scoping and a
  server database — out of scope until needed.
