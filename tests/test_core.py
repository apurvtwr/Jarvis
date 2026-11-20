import pytest
from local_agent import core

def test_route_code(monkeypatch):
    monkeypatch.setattr(core, "config", lambda: core.DEFAULT)
    assert core.route("debug this C++ function") == "ollama/qwen2.5-coder:7b"
    assert core.route("write a poem") == "ollama/qwen3:8b"
    assert core.route("debug this function", "general") == "ollama/qwen3:8b"

def test_folder_boundaries(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "HOME", tmp_path / "db")
    folder = tmp_path / "project"
    folder.mkdir()
    other = tmp_path / "outside.txt"
    other.write_text("secret")
    cid = core.create_conversation()
    core.add_folder(cid, str(folder))
    assert core.allowed_path(cid, str(folder / "file.txt")) == folder / "file.txt"
    with pytest.raises(ValueError):
        core.allowed_path(cid, str(other))
    (folder / "link").symlink_to(other)
    with pytest.raises(ValueError):
        core.tool_execute(cid, "read_file", {"path": str(folder / "link")})

def test_read_write(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "HOME", tmp_path / "db")
    folder = tmp_path / "project"
    folder.mkdir()
    cid = core.create_conversation()
    core.add_folder(cid, str(folder))
    p = str(folder / "a.txt")
    core.tool_execute(cid, "write_file", {"path": p, "content": "hello"})
    assert core.tool_execute(cid, "read_file", {"path": p}) == "hello"

def test_pinned_model_bypasses_router(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "HOME", tmp_path / "db")
    monkeypatch.setattr(core, "config", lambda: core.DEFAULT)
    cid = core.create_conversation("Coding", "ollama/qwen2.5-coder:7b")
    def no_router(*args, **kwargs):
        raise AssertionError("Router should not be invoked")
    monkeypatch.setattr(core, "route", no_router)
    monkeypatch.setattr(core, "advance", lambda chat, model: {"status": "complete", "model": model})
    assert core.respond(cid, "write a poem")["model"] == "ollama/qwen2.5-coder:7b"
    assert core.get_conversation(cid)["model_override"] == "ollama/qwen2.5-coder:7b"
    assert core.set_conversation_model(cid, "auto") is None

def test_model_persistence(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "HOME", tmp_path / "db")
    monkeypatch.setattr(core, "config", lambda: core.DEFAULT)
    cid = core.create_conversation()
    assert core.set_conversation_model(cid, "qwen3:8b") == "ollama/qwen3:8b"
    assert core.get_conversation(cid)["model_override"] == "ollama/qwen3:8b"


def test_projects_share_folders_and_context(tmp_path, monkeypatch):
    from local_agent import core
    monkeypatch.setattr(core, 'HOME', tmp_path / 'data')
    folder = tmp_path / 'workspace'
    folder.mkdir()
    pid = core.create_project('Robotics', 'Use ROS2 Jazzy')
    core.add_project_folder(pid, str(folder))
    a = core.create_conversation('Planning', project_id=pid)
    b = core.create_conversation('Testing', project_id=pid)
    assert str(folder) in core.folders(a)
    assert str(folder) in core.folders(b)
    assert 'ROS2 Jazzy' in core.project_context(b)
    core.save_message(a, 'user', 'Remember the controller interface')
    assert 'controller interface' in core.project_context(b)
    core.rename_conversation(b, 'Controller tests')
    assert core.get_conversation(b)['title'] == 'Controller tests'
    core.move_conversation(b, None)
    assert str(folder) not in core.folders(b)


def test_delete_conversation_and_project(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "HOME", tmp_path / "data")
    folder = tmp_path / "ws"
    folder.mkdir()
    pid = core.create_project("Robot")
    core.add_project_folder(pid, str(folder))
    a = core.create_conversation("A", project_id=pid)
    b = core.create_conversation("B", project_id=pid)
    core.save_message(a, "user", "hello")
    core.delete_conversation(a)
    assert a not in {c["id"] for c in core.conversations()}
    with pytest.raises(ValueError):
        core.history(a)
    core.delete_project(pid)
    assert core.get_conversation(b)["project_id"] is None
    assert core.folders(b) == []
    assert core.projects() == []


def test_move_between_projects_changes_shared_context(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "HOME", tmp_path / "data")
    p1, p2 = core.create_project("One", "alpha rules"), core.create_project("Two", "beta rules")
    c = core.create_conversation("Chat", project_id=p1)
    assert "alpha rules" in core.project_context(c)
    core.move_conversation(c, p2)
    assert "beta rules" in core.project_context(c) and "alpha" not in core.project_context(c)
    assert "beta rules" in core.system_prompt(c)


def test_openai_wire_format():
    from local_agent.adapters import OpenAIAdapter
    wire = OpenAIAdapter({"type": "openai"}).to_wire([
        {"role": "system", "content": "sys"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "function": {"name": "read_file", "arguments": {"path": "/x"}}}]},
        {"role": "tool", "tool_call_id": "c1", "tool_name": "read_file", "content": "data"},
    ])
    assert wire[1]["tool_calls"][0]["function"]["arguments"] == '{"path": "/x"}'
    assert wire[1]["content"] is None
    assert wire[2] == {"role": "tool", "tool_call_id": "c1", "content": "data"}


def test_anthropic_wire_format():
    from local_agent.adapters import AnthropicAdapter
    system, wire = AnthropicAdapter({"type": "anthropic"}).to_wire([
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "let me look", "tool_calls": [
            {"id": "t1", "function": {"name": "list_files", "arguments": {"path": "/a"}}},
            {"id": "t2", "function": {"name": "list_files", "arguments": {"path": "/b"}}}]},
        {"role": "tool", "tool_call_id": "t1", "content": "[]"},
        {"role": "tool", "tool_call_id": "t2", "content": "[]"},
    ])
    assert system == "sys"
    assert [m["role"] for m in wire] == ["user", "assistant", "user"]
    assert [b["type"] for b in wire[1]["content"]] == ["text", "tool_use", "tool_use"]
    assert [b["tool_use_id"] for b in wire[2]["content"]] == ["t1", "t2"]


def test_unknown_adapter_type():
    from local_agent.adapters import get_adapter
    with pytest.raises(ValueError):
        get_adapter({"type": "nope"})


def test_register_custom_adapter_and_approval_flow(tmp_path, monkeypatch):
    from local_agent import adapters
    monkeypatch.setattr(core, "HOME", tmp_path / "data")
    folder = tmp_path / "ws"
    folder.mkdir()
    cfg = {**core.DEFAULT, "providers": {**core.DEFAULT["providers"], "fake": {"type": "fake"}}}
    monkeypatch.setattr(core, "config", lambda: cfg)
    seen = []

    @adapters.register("fake")
    class Fake(adapters.Adapter):
        def chat(self, messages, model, tools=None):
            seen.append(messages[-1])
            if messages[-1]["role"] == "user":
                return {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "w1", "function": {"name": "write_file", "arguments": {"path": str(folder / "o.txt"), "content": "hi"}}}]}
            return {"role": "assistant", "content": "done", "tool_calls": []}

    try:
        cid = core.create_conversation("x", model="fake/m1")
        core.add_folder(cid, str(folder))
        result = core.respond(cid, "write it")
        assert result["status"] == "approval" and result["model"] == "fake/m1"
        final = core.approve(cid, result["id"], True)
        assert final["content"] == "done"
        assert (folder / "o.txt").read_text() == "hi"
        assert seen[-1]["tool_call_id"] == "w1"
    finally:
        adapters.ADAPTERS.pop("fake", None)
