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
