"""Runner executor: boundary checks on the runner, the server registry/timeouts and the
WebSocket endpoint. All offline: runners are in-process fakes."""
import json
import sqlite3
import threading

import pytest

from local_agent import core, runner as jr


def runner_cfg(timeout=5):
    cfg = json.loads(json.dumps(core.DEFAULT))
    cfg["tools"] = {"executor": "runner", "runner_timeout_seconds": timeout}
    return cfg


class FakeLink:
    """Stands in for the WebSocket: delivers each call to a local Runner on a thread."""

    def __init__(self, worker, answer=True):
        self.worker, self.answer, self.sent = worker, answer, []

    def attach(self, registry, rid="r1", hostname="laptop"):
        self.conn = registry.register(rid, hostname, self.send)
        return self.conn

    def send(self, msg):
        self.sent.append(msg)
        if self.answer:
            threading.Thread(target=lambda: core.RUNNERS.deliver(self.conn, self.worker.handle(msg))).start()


@pytest.fixture
def runner_env(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "HOME", tmp_path / "data")
    monkeypatch.setattr(core, "config", lambda: runner_cfg())
    monkeypatch.setattr(core, "RUNNERS", core.RunnerRegistry())
    exposed = tmp_path / "exposed"
    (exposed / "proj").mkdir(parents=True)
    (exposed / "proj" / "notes.txt").write_text("local notes")
    (tmp_path / "secret.txt").write_text("secret")
    worker = jr.Runner(allow=[str(exposed)], deny=[str(exposed / "private")])
    link = FakeLink(worker)
    link.attach(core.RUNNERS)
    return tmp_path, exposed, worker, link


# ---------- runner-side boundary checks (do not trust the server) ----------

def test_runner_enforces_grants_and_symlinks(runner_env):
    tmp, exposed, worker, _ = runner_env
    proj = str(exposed / "proj")
    ok = worker.handle({"call_id": "1", "tool": "read_file", "args": {"path": proj + "/notes.txt"}, "grants": [proj]})
    assert ok == {"type": "result", "call_id": "1", "ok": True, "result": "local notes"}
    # Not in the grants the server sent.
    other = exposed / "other.txt"
    other.write_text("x")
    r = worker.handle({"call_id": "2", "tool": "read_file", "args": {"path": str(other)}, "grants": [proj]})
    assert not r["ok"] and "outside granted" in r["error"]
    # Symlink inside the grant that points outside it is resolved and refused.
    (exposed / "proj" / "link").symlink_to(tmp / "secret.txt")
    r = worker.handle({"call_id": "3", "tool": "read_file", "args": {"path": proj + "/link"}, "grants": [proj]})
    assert not r["ok"] and "outside granted" in r["error"]
    # Relative paths and ../ tricks resolve before the check.
    r = worker.handle({"call_id": "4", "tool": "read_file", "args": {"path": proj + "/../../secret.txt"}, "grants": [proj]})
    assert not r["ok"]


def test_compromised_server_cannot_widen_grants(runner_env):
    tmp, exposed, worker, _ = runner_env
    # A malicious server claims "/" is granted: the local allow list still applies.
    r = worker.handle({"call_id": "1", "tool": "read_file", "args": {"path": str(tmp / "secret.txt")}, "grants": ["/"]})
    assert not r["ok"] and "runner.allow" in r["error"]
    (exposed / "private").mkdir()
    (exposed / "private" / "key").write_text("k")
    r = worker.handle({"call_id": "2", "tool": "read_file", "args": {"path": str(exposed / "private" / "key")}, "grants": [str(exposed)]})
    assert not r["ok"] and "runner.deny" in r["error"]
    r = worker.handle({"call_id": "3", "tool": "resolve", "args": {"path": str(tmp / "secret.txt")}})
    assert not r["ok"]
    r = worker.handle({"call_id": "4", "tool": "resolve", "args": {"path": "/"}})
    assert not r["ok"]
    r = worker.handle({"call_id": "5", "tool": "rm_rf", "args": {}, "grants": ["/"]})
    assert not r["ok"] and r["error"] == "Unknown tool"


def test_runner_can_disable_commands(tmp_path):
    worker = jr.Runner(allow=[str(tmp_path)], allow_commands=False)
    r = worker.handle({"call_id": "1", "tool": "run_command", "args": {"cwd": str(tmp_path), "argv": ["true"]}, "grants": [str(tmp_path)]})
    assert not r["ok"] and "disabled" in r["error"]


def test_default_deny_protects_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / ".ssh").mkdir()
    (tmp_path / ".ssh" / "id_ed25519").write_text("key")
    worker = jr.Runner()  # default: allow ~, deny credential stores
    r = worker.handle({"call_id": "1", "tool": "read_file", "args": {"path": str(tmp_path / ".ssh/id_ed25519")}, "grants": [str(tmp_path)]})
    assert not r["ok"] and "runner.deny" in r["error"]


def test_connect_url_requires_tls_off_loopback():
    assert jr.connect_url("http://127.0.0.1:8765") == "ws://127.0.0.1:8765/api/runners/connect"
    assert jr.connect_url("https://j.example.com/base/") == "wss://j.example.com/base/api/runners/connect"
    with pytest.raises(ValueError):
        jr.connect_url("http://j.example.com")


def test_runner_id_is_stable(tmp_path):
    path = tmp_path / "rid"
    assert jr.runner_id(path) == jr.runner_id(path)


# ---------- server side: grants, routing, approvals, timeouts ----------

def test_chat_lists_reads_and_writes_through_runner_with_approval(runner_env, monkeypatch):
    from local_agent import adapters
    tmp, exposed, worker, link = runner_env
    proj = exposed / "proj"
    (proj / "AGENTS.md").write_text("Runner-side rules.")
    cid = core.create_conversation("x", model="ollama/m")
    assert core.add_folder(cid, str(proj)) == str(proj)            # resolved on the runner
    assert core.grants(cid) == [{"path": str(proj), "runner_id": "r1"}]
    assert "Runner-side rules." in core.system_prompt(cid)        # read through the runner
    script = iter([
        [{"id": "a", "function": {"name": "list_files", "arguments": {"path": str(proj)}}},
         {"id": "b", "function": {"name": "read_file", "arguments": {"path": str(proj / "notes.txt")}}}],
        [{"id": "c", "function": {"name": "write_file", "arguments": {"path": str(proj / "out.txt"), "content": "hi"}}}],
        [],
    ])
    seen = []

    def fake_chat(messages, model, tools=None):
        seen.append(messages[-2:])
        calls = next(script)
        return {"role": "assistant", "content": "" if calls else "done", "tool_calls": calls}

    monkeypatch.setattr(core, "model_chat", fake_chat)
    result = core.respond(cid, "do it")
    assert result["status"] == "approval" and result["tool"] == "write_file"
    assert not (proj / "out.txt").exists()                        # nothing written before approval
    tool_results = [m["content"] for m in seen[1] if m["role"] == "tool"]
    assert "notes.txt" in tool_results[0] and tool_results[1] == "local notes"
    assert core.approve(cid, result["id"], True)["content"] == "done"
    assert (proj / "out.txt").read_text() == "hi"
    assert all(m["grants"] == [str(proj)] for m in link.sent if m["tool"] != "resolve")
    assert adapters  # imported for parity with other tests


def test_server_refuses_paths_outside_grants_before_calling_runner(runner_env):
    tmp, exposed, worker, link = runner_env
    cid = core.create_conversation()
    core.add_folder(cid, str(exposed / "proj"))
    sent = len(link.sent)
    for path in (str(tmp / "secret.txt"), str(exposed / "projX"), "relative/path"):
        with pytest.raises(ValueError):
            core.executor().execute(cid, "read_file", {"path": path})
    assert len(link.sent) == sent
    # Lexically inside but escaping via symlink: the runner refuses it.
    (exposed / "proj" / "esc").symlink_to(tmp / "secret.txt")
    with pytest.raises(ValueError, match="outside granted"):
        core.executor().execute(cid, "read_file", {"path": str(exposed / "proj" / "esc")})


def test_runner_timeout_becomes_tool_error(runner_env, monkeypatch):
    tmp, exposed, worker, link = runner_env
    cid = core.create_conversation(model="ollama/m")
    core.add_folder(cid, str(exposed / "proj"))
    link.answer = False
    monkeypatch.setattr(core, "config", lambda: runner_cfg(timeout=0.2))
    with pytest.raises(TimeoutError):
        core.executor().execute(cid, "read_file", {"path": str(exposed / "proj" / "notes.txt")})
    assert core.RUNNERS.pending == {}
    replies = iter([[{"id": "a", "function": {"name": "read_file", "arguments": {"path": str(exposed / "proj" / "notes.txt")}}}], []])
    seen = []

    def fake_chat(messages, model, tools=None):
        seen.append(messages[-1])
        calls = next(replies)
        return {"role": "assistant", "content": "" if calls else "ok", "tool_calls": calls}

    monkeypatch.setattr(core, "model_chat", fake_chat)
    assert core.respond(cid, "read")["content"] == "ok"
    assert seen[-1]["role"] == "tool" and "did not answer read_file" in seen[-1]["content"]


def test_disconnect_fails_pending_and_unknown_runner(runner_env):
    tmp, exposed, worker, link = runner_env
    cid = core.create_conversation()
    core.add_folder(cid, str(exposed / "proj"))
    link.answer = False
    errors = []

    def call():
        try:
            core.executor().execute(cid, "read_file", {"path": str(exposed / "proj" / "notes.txt")})
        except Exception as exc:
            errors.append(exc)
    t = threading.Thread(target=call)
    t.start()
    for _ in range(100):
        if core.RUNNERS.pending:
            break
        threading.Event().wait(0.01)
    core.RUNNERS.unregister(link.conn)
    t.join(5)
    assert errors and "disconnected" in str(errors[0])
    with pytest.raises(ValueError, match="not connected"):
        core.executor().execute(cid, "read_file", {"path": str(exposed / "proj" / "notes.txt")})
    # An answer from a different connection is ignored.
    other = core.RUNNERS.register("r2", "intruder", lambda m: None)
    core.RUNNERS.deliver(other, {"call_id": "nope", "ok": True, "result": "x"})


def test_grant_requires_a_runner(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "HOME", tmp_path / "data")
    monkeypatch.setattr(core, "config", lambda: runner_cfg())
    monkeypatch.setattr(core, "RUNNERS", core.RunnerRegistry())
    cid = core.create_conversation()
    with pytest.raises(ValueError, match="No runner connected"):
        core.add_folder(cid, str(tmp_path))
    core.RUNNERS.register("a", "one", lambda m: None)
    core.RUNNERS.register("b", "two", lambda m: None)
    with pytest.raises(ValueError, match="Several runners"):
        core.add_folder(cid, str(tmp_path))


def test_grants_table_migrates_runner_id(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "HOME", tmp_path / "data")
    (tmp_path / "data").mkdir()
    con = sqlite3.connect(tmp_path / "data" / "agent.sqlite3")
    con.executescript("CREATE TABLE folders(conversation_id TEXT NOT NULL, path TEXT NOT NULL, PRIMARY KEY(conversation_id,path));"
                      "CREATE TABLE project_folders(project_id TEXT NOT NULL, path TEXT NOT NULL, PRIMARY KEY(project_id,path));"
                      "INSERT INTO folders VALUES ('c','/old');")
    con.commit()
    con.close()
    with core.db() as con:
        for table in ("folders", "project_folders"):
            assert "runner_id" in {r[1] for r in con.execute(f"PRAGMA table_info({table})")}
        assert con.execute("SELECT runner_id FROM folders").fetchone()[0] is None


# ---------- WebSocket endpoint ----------

def _ws_client(tmp_path, monkeypatch, token=None, runner_token=None):
    from fastapi.testclient import TestClient
    from local_agent import server
    monkeypatch.setattr(core, "HOME", tmp_path / "data")
    monkeypatch.setattr(core, "config", lambda: runner_cfg())
    monkeypatch.setattr(core, "RUNNERS", core.RunnerRegistry())
    for name, value in (("JARVIS_AUTH_TOKEN", token), ("JARVIS_RUNNER_TOKEN", runner_token)):
        if value:
            monkeypatch.setenv(name, value)
        else:
            monkeypatch.delenv(name, raising=False)
    return TestClient(server.app)


def test_websocket_runner_end_to_end(tmp_path, monkeypatch):
    from starlette.websockets import WebSocketDisconnect
    client = _ws_client(tmp_path, monkeypatch, token="api", runner_token="run")
    with pytest.raises(WebSocketDisconnect) as bad:
        with client.websocket_connect("/api/runners/connect", headers={"Authorization": "Bearer wrong"}) as ws:
            ws.receive_json()
    assert bad.value.code == 4401
    worker = jr.Runner(allow=[str(tmp_path)])
    (tmp_path / "w").mkdir()
    (tmp_path / "w" / "a.txt").write_text("over the wire")
    with client.websocket_connect("/api/runners/connect", headers={"Authorization": "Bearer run"}) as ws:
        ws.send_json({"type": "hello", "runner_id": "laptop-1", "hostname": "laptop"})
        assert ws.receive_json()["type"] == "welcome"
        info = client.get("/api/info").json()["tools"]
        assert info == {"executor": "runner", "host": "laptop", "runners": [{"id": "laptop-1", "hostname": "laptop"}]}
        headers = {"Authorization": "Bearer api"}
        cid = client.post("/api/chats", json={"title": "t"}, headers=headers).json()["id"]
        out = {}

        def grant():
            out["grant"] = client.post(f"/api/chats/{cid}/folders", json={"path": str(tmp_path / "w"), "runner_id": "laptop-1"}, headers=headers)
        t = threading.Thread(target=grant)
        t.start()
        msg = ws.receive_json()                       # the server asks the runner to resolve the grant
        assert msg["tool"] == "resolve"
        ws.send_json(worker.handle(msg))
        t.join(5)
        assert out["grant"].status_code == 200 and out["grant"].json()["path"] == str(tmp_path / "w")

        def read():
            out["read"] = core.executor().execute(cid, "read_file", {"path": str(tmp_path / "w" / "a.txt")})
        t = threading.Thread(target=read)
        t.start()
        ws.send_json(worker.handle(ws.receive_json()))
        t.join(5)
        assert out["read"] == "over the wire"
    for _ in range(100):  # unregistered once the socket closes
        if not core.RUNNERS.list():
            break
        threading.Event().wait(0.01)
    assert client.get("/api/info").json()["tools"]["runners"] == []


def test_websocket_rejects_bad_hello(tmp_path, monkeypatch):
    from starlette.websockets import WebSocketDisconnect
    client = _ws_client(tmp_path, monkeypatch)   # no tokens: loopback-only, TestClient host is "testclient"
    from local_agent import server
    monkeypatch.setattr(server, "is_loopback", lambda host: True)
    with pytest.raises(WebSocketDisconnect) as bad:
        with client.websocket_connect("/api/runners/connect") as ws:
            ws.send_json({"type": "hello", "runner_id": "../../etc", "hostname": "x"})
            ws.receive_json()
    assert bad.value.code == 4400
    monkeypatch.setattr(server, "is_loopback", lambda host: False)
    with pytest.raises(WebSocketDisconnect) as bad:
        with client.websocket_connect("/api/runners/connect") as ws:
            ws.receive_json()
    assert bad.value.code == 4401


def test_serve_connection_protocol(tmp_path):
    """The runner loop over a fake sync socket: hello, welcome, call → result."""
    (tmp_path / "f.txt").write_text("data")

    class FakeSocket:
        def __init__(self):
            self.out = []
            self.incoming = [json.dumps({"type": "call", "call_id": "x", "tool": "read_file",
                                         "args": {"path": str(tmp_path / "f.txt")}, "grants": [str(tmp_path)]})]

        def send(self, raw):
            self.out.append(json.loads(raw))

        def recv(self, timeout=None):
            return json.dumps({"type": "welcome"})

        def __iter__(self):
            return iter(self.incoming)

    sock = FakeSocket()
    jr.serve_connection(sock, jr.Runner(allow=[str(tmp_path)]), "rid", "host", log=lambda m: None)
    assert sock.out[0] == {"type": "hello", "runner_id": "rid", "hostname": "host"}
    assert sock.out[1] == {"type": "result", "call_id": "x", "ok": True, "result": "data"}
