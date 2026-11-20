"""LLM backend adapters.

Jarvis talks to every backend through one internal message format (the
Ollama/OpenAI-style chat format) and lets each adapter translate to and from
its provider's wire protocol.

Internal format
---------------
messages: list of dicts with ``role`` in {system, user, assistant, tool}
  * assistant messages may carry ``tool_calls``:
        [{"id": str, "function": {"name": str, "arguments": dict}}]
  * tool messages carry ``tool_call_id`` and ``tool_name`` plus ``content``
tools: OpenAI-style function schemas
        [{"type": "function", "function": {"name", "description", "parameters"}}]

``Adapter.chat`` returns one assistant message in the same internal format.

Adding a backend
----------------
Subclass ``Adapter``, implement ``chat`` (and optionally ``list_models``), and
decorate it with ``@register("your-type")``. Then reference it from
``~/.config/jarvis/config.yaml``::

    providers:
      mybackend:
        type: your-type
        base_url: https://...
        api_key_env: MYBACKEND_API_KEY
"""
import json
import os
import uuid

import httpx

ADAPTERS = {}


def register(type_name):
    def wrap(cls):
        cls.type_name = type_name
        ADAPTERS[type_name] = cls
        return cls
    return wrap


def get_adapter(settings):
    adapter_type = settings.get("type")
    if adapter_type not in ADAPTERS:
        raise ValueError(f"Unsupported adapter type: {adapter_type!r}. Known types: {', '.join(sorted(ADAPTERS))}")
    return ADAPTERS[adapter_type](settings)


def new_call_id():
    return "call_" + uuid.uuid4().hex[:24]


def parse_arguments(raw):
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


class Adapter:
    """Base class for LLM backends."""

    type_name = "base"
    default_base_url = ""
    default_api_key_env = None

    def __init__(self, settings):
        self.settings = settings
        self.base_url = (settings.get("base_url") or self.default_base_url).rstrip("/")
        self.timeout = float(settings.get("timeout_seconds", 180))
        self.temperature = settings.get("temperature", 0.2)

    @property
    def api_key(self):
        if self.settings.get("api_key"):
            return self.settings["api_key"]
        env = self.settings.get("api_key_env", self.default_api_key_env)
        return os.environ.get(env) if env else None

    def require_key(self):
        key = self.api_key
        if not key:
            env = self.settings.get("api_key_env", self.default_api_key_env)
            raise ValueError(f"{self.type_name} provider has no API key; set the {env} environment variable "
                             "(for the desktop service, put it in ~/.config/jarvis/env)")
        return key

    def configured_models(self):
        return list(self.settings.get("models") or [])

    def chat(self, messages, model, tools=None):
        raise NotImplementedError

    def list_models(self):
        """Models offered by this backend. Defaults to the ``models`` list in config."""
        return self.configured_models()

    def status(self):
        """Short human-readable health description, never raises."""
        try:
            models = self.list_models()
            return f"ok ({len(models)} models)"
        except Exception as exc:  # pragma: no cover - diagnostics only
            return f"unavailable: {exc}"


@register("ollama")
class OllamaAdapter(Adapter):
    default_base_url = "http://127.0.0.1:11434"

    def to_wire(self, messages):
        out = []
        for m in messages:
            msg = {"role": m["role"], "content": m.get("content") or ""}
            if m.get("tool_calls"):
                msg["tool_calls"] = [{"function": {"name": c["function"]["name"],
                                                   "arguments": parse_arguments(c["function"].get("arguments"))}}
                                     for c in m["tool_calls"]]
            if m["role"] == "tool" and m.get("tool_name"):
                msg["tool_name"] = m["tool_name"]
            out.append(msg)
        return out

    def chat(self, messages, model, tools=None):
        payload = {"model": model, "messages": self.to_wire(messages), "stream": False,
                   "options": {"temperature": self.temperature}}
        if tools:
            payload["tools"] = tools
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(self.base_url + "/api/chat", json=payload)
            response.raise_for_status()
            msg = response.json()["message"]
        calls = [{"id": c.get("id") or new_call_id(),
                  "function": {"name": c["function"]["name"], "arguments": parse_arguments(c["function"].get("arguments"))}}
                 for c in msg.get("tool_calls") or []]
        return {"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls}

    def list_models(self):
        if self.configured_models():
            return self.configured_models()
        with httpx.Client(timeout=5) as client:
            response = client.get(self.base_url + "/api/tags")
            response.raise_for_status()
            return [item["name"] for item in response.json().get("models", [])]


@register("openai")
class OpenAIAdapter(Adapter):
    """OpenAI Chat Completions. Also works for OpenAI-compatible servers
    (LM Studio, vLLM, llama.cpp server, OpenRouter, Groq...) via ``base_url``."""

    default_base_url = "https://api.openai.com/v1"
    default_api_key_env = "OPENAI_API_KEY"

    def headers(self):
        key = self.api_key
        if not key and "api.openai.com" in self.base_url:
            key = self.require_key()
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        if self.settings.get("organization"):
            headers["OpenAI-Organization"] = self.settings["organization"]
        return headers

    def to_wire(self, messages):
        out = []
        for m in messages:
            if m["role"] == "tool":
                out.append({"role": "tool", "tool_call_id": m.get("tool_call_id") or new_call_id(),
                            "content": m.get("content") or ""})
                continue
            msg = {"role": m["role"], "content": m.get("content") or ""}
            if m.get("tool_calls"):
                msg["tool_calls"] = [{"id": c.get("id") or new_call_id(), "type": "function",
                                      "function": {"name": c["function"]["name"],
                                                   "arguments": json.dumps(parse_arguments(c["function"].get("arguments")))}}
                                     for c in m["tool_calls"]]
                if not msg["content"]:
                    msg["content"] = None
            out.append(msg)
        return out

    def chat(self, messages, model, tools=None):
        payload = {"model": model, "messages": self.to_wire(messages)}
        if self.temperature is not None and not self.settings.get("omit_temperature"):
            payload["temperature"] = self.temperature
        if tools:
            payload["tools"] = tools
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(self.base_url + "/chat/completions", json=payload, headers=self.headers())
            if response.status_code >= 400:
                raise ValueError(f"OpenAI-compatible backend returned {response.status_code}: {response.text[:400]}")
            msg = response.json()["choices"][0]["message"]
        calls = [{"id": c.get("id") or new_call_id(),
                  "function": {"name": c["function"]["name"], "arguments": parse_arguments(c["function"].get("arguments"))}}
                 for c in msg.get("tool_calls") or []]
        return {"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls}

    def list_models(self):
        if self.configured_models():
            return self.configured_models()
        if "api.openai.com" in self.base_url and not self.api_key:
            return []
        with httpx.Client(timeout=5) as client:
            response = client.get(self.base_url + "/models", headers=self.headers())
            response.raise_for_status()
            return sorted(item["id"] for item in response.json().get("data", []))


@register("anthropic")
class AnthropicAdapter(Adapter):
    """Anthropic Messages API (Claude)."""

    default_base_url = "https://api.anthropic.com/v1"
    default_api_key_env = "ANTHROPIC_API_KEY"
    api_version = "2023-06-01"

    def headers(self):
        return {"x-api-key": self.require_key(), "anthropic-version": self.settings.get("api_version", self.api_version),
                "content-type": "application/json"}

    def to_wire(self, messages):
        system, out = [], []

        def append(role, blocks):
            # Anthropic requires alternating roles; merge consecutive same-role turns.
            if out and out[-1]["role"] == role:
                out[-1]["content"].extend(blocks)
            else:
                out.append({"role": role, "content": list(blocks)})

        for m in messages:
            text = m.get("content") or ""
            if m["role"] == "system":
                if text:
                    system.append(text)
            elif m["role"] == "tool":
                append("user", [{"type": "tool_result", "tool_use_id": m.get("tool_call_id") or new_call_id(),
                                 "content": text or "(empty)"}])
            elif m["role"] == "assistant":
                blocks = [{"type": "text", "text": text}] if text else []
                for c in m.get("tool_calls") or []:
                    blocks.append({"type": "tool_use", "id": c.get("id") or new_call_id(),
                                   "name": c["function"]["name"], "input": parse_arguments(c["function"].get("arguments"))})
                if blocks:
                    append("assistant", blocks)
            else:
                append("user", [{"type": "text", "text": text or "(empty)"}])
        return "\n\n".join(system), out

    def chat(self, messages, model, tools=None):
        system, wire = self.to_wire(messages)
        payload = {"model": model, "max_tokens": int(self.settings.get("max_tokens", 4096)), "messages": wire}
        if system:
            payload["system"] = system
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if tools:
            payload["tools"] = [{"name": t["function"]["name"], "description": t["function"].get("description", ""),
                                 "input_schema": t["function"]["parameters"]} for t in tools]
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(self.base_url + "/messages", json=payload, headers=self.headers())
            if response.status_code >= 400:
                raise ValueError(f"Anthropic backend returned {response.status_code}: {response.text[:400]}")
            data = response.json()
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        calls = [{"id": b["id"], "function": {"name": b["name"], "arguments": b.get("input") or {}}}
                 for b in data.get("content", []) if b.get("type") == "tool_use"]
        return {"role": "assistant", "content": text, "tool_calls": calls}

    def list_models(self):
        if self.configured_models():
            return self.configured_models()
        if not self.api_key:
            return []
        with httpx.Client(timeout=5) as client:
            response = client.get(self.base_url + "/models", headers=self.headers(), params={"limit": 100})
            response.raise_for_status()
            return sorted(item["id"] for item in response.json().get("data", []))
