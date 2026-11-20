"""HTTP client for the Jarvis API, used by the CLI.

Every client (CLI, desktop app, web UI) talks to the backend only through the
HTTP API, so the backend can run locally or in the cloud. Where to connect is
*client* configuration, separate from the server's config.yaml:

    ~/.config/jarvis/client.yaml      (chmod 600 if it holds a token)
        url: https://jarvis.example.com
        token: <bearer token>          # or token_env: JARVIS_TOKEN

Environment overrides: JARVIS_URL, JARVIS_TOKEN. With nothing configured the
client uses the local backend from config.yaml (http://127.0.0.1:8765).
"""
import ipaddress
import os
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx
import yaml

CLIENT_CONFIG = Path(os.environ.get("JARVIS_CLIENT_CONFIG", Path.home() / ".config/jarvis/client.yaml"))


def client_settings():
    data = {}
    if CLIENT_CONFIG.exists():
        data = yaml.safe_load(CLIENT_CONFIG.read_text()) or {}
    url = os.environ.get("JARVIS_URL") or data.get("url")
    if not url:
        from . import core
        server = core.config()["server"]
        url = f"http://{server['host']}:{server['port']}"
    token = os.environ.get("JARVIS_TOKEN") or data.get("token") or (os.environ.get(data["token_env"]) if data.get("token_env") else None)
    return {"url": url.rstrip("/"), "token": token}


def is_local_url(url):
    host = urlparse(url).hostname or ""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class ApiError(Exception):
    def __init__(self, status, detail):
        super().__init__(detail)
        self.status = status


class JarvisClient:
    def __init__(self, url=None, token=None, timeout=30.0):
        settings = client_settings()
        self.url = (url or settings["url"]).rstrip("/")
        self.token = token if token is not None else settings["token"]
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        self.http = httpx.Client(base_url=self.url + "/api", headers=headers, timeout=timeout)

    @property
    def local(self):
        return is_local_url(self.url)

    def request(self, method, path, json=None, params=None):
        try:
            response = self.http.request(method, path, json=json, params=params)
        except httpx.HTTPError as exc:
            raise ApiError(0, f"Cannot reach Jarvis at {self.url}: {exc}") from exc
        if response.status_code == 401:
            raise ApiError(401, "Authentication required: set JARVIS_TOKEN or `token:` in " + str(CLIENT_CONFIG))
        try:
            body = response.json() if response.content else {}
        except ValueError:
            body = {}
        if response.status_code >= 400:
            detail = body.get("detail") if isinstance(body, dict) else None
            raise ApiError(response.status_code, detail if isinstance(detail, str) else f"HTTP {response.status_code}")
        return body

    get = lambda self, path, **kw: self.request("GET", path, **kw)  # noqa: E731
    post = lambda self, path, data=None, **kw: self.request("POST", path, json=data if data is not None else {}, **kw)  # noqa: E731
    put = lambda self, path, data=None: self.request("PUT", path, json=data or {})  # noqa: E731
    patch = lambda self, path, data=None: self.request("PATCH", path, json=data or {})  # noqa: E731
    delete = lambda self, path, **kw: self.request("DELETE", path, **kw)  # noqa: E731

    def wait_for_turn(self, cid, poll=1.0, on_tick=None):
        """Turns run in the background on the server; poll until the chat is idle."""
        while True:
            detail = self.get(f"/chats/{cid}")
            status = detail["conversation"].get("run_status")
            if status != "running":
                if status == "error":
                    raise ApiError(502, detail["conversation"].get("run_error") or "Turn failed")
                return detail
            if on_tick:
                on_tick()
            time.sleep(poll)

    def send(self, cid, message, model="auto"):
        self.post(f"/chats/{cid}/messages", {"message": message, "model": model})
        return self.wait_for_turn(cid)

    def approve(self, cid, aid, accept):
        self.post(f"/chats/{cid}/approvals/{aid}", {"accept": accept})
        return self.wait_for_turn(cid)
