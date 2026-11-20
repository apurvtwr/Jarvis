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
 │ [future] Runner: executes tools on      │ ◄─────── │ Adapters (adapters.py) → Ollama/OpenAI/  │
 │          this machine's files           │  (WSS)   │                          Anthropic       │
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
   `auth_required` and `tools: {executor, host}`. Clients decide from it whether a
   local file picker makes sense.

## Where tools run (the important cloud question)

Agent tools (`list_files`, `read_file`, `write_file`, `run_command`) touch files.
`core.EXECUTORS` maps `tools.executor` in the server config to the class that runs them:

| executor | status | runs tools on | right when |
|---|---|---|---|
| `server` | implemented (`ServerExecutor`) | the machine running the backend | the backend runs on the user's computer, or the user wants a cloud workspace |
| `runner` | **design only** | the user's computer, via a small runner process | the backend runs in the cloud but files stay local |

Grants (`folders`, `project_folders`) are paths **on the machine where the executor
runs**. The desktop app compares `/api/info` `tools.host` to its own hostname: native
browsing is offered only when they match; otherwise the user types a path that exists
on that host.

### Planned runner design

* `jarvis runner` runs on the user's computer, opens an **outbound** WebSocket to
  `wss://server/api/runners/connect` (works through NAT/firewalls; nothing listens
  locally), authenticated with its own runner token, and registers a stable
  `runner_id` and hostname.
* Grants gain a `runner_id`. `RunnerExecutor.execute` sends
  `{call_id, tool, args}` to that runner and awaits the result (timeout →
  tool error). The runner enforces the same `allowed_path` checks locally, so a
  compromised server still cannot escape the user's grants.
* Approvals stay on the server's run state; the runner never decides on its own.
* The desktop app may host the runner in-process later, so a single launch does both.

## Deployment checklist (cloud)

- Set `server.host: 0.0.0.0` and `JARVIS_AUTH_TOKEN` (long random string) in the
  service environment; put a TLS reverse proxy in front.
- Configure providers on the server (API keys via env). Ollama must be reachable
  from the server (`base_url`), not from the client.
- On each client: `client.yaml` with `url: https://…` and the token (`chmod 600`).
- Choose `tools.executor`: `server` for a cloud workspace; `runner` once implemented
  for local files.
- SQLite is fine for one user; multi-user would need per-user scoping and a
  server database — out of scope until needed.
