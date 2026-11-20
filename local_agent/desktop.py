#!/usr/bin/python3
"""Jarvis native desktop shell for Ubuntu / GNOME.

A GTK4 + libadwaita window hosting the Jarvis web UI in WebKitGTK 6.0, with a
small JS bridge (``window.jarvisDesktop``) for native features such as the
GNOME file/folder picker.

This file is deliberately standalone: it uses only the standard library and
the *system* PyGObject (``python3-gi``), so it runs with ``/usr/bin/python3``
rather than the project virtualenv. Ubuntu 24.04 packages:

    sudo apt install python3-gi gir1.2-gtk-4.0 gir1.2-adw-1 gir1.2-webkit-6.0
"""
import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.request import urlopen

APP_ID = "dev.jarvis.Jarvis"
CONFIG_DIR = Path(os.environ.get("LOCAL_AGENT_CONFIG", Path.home() / ".config/jarvis/config.yaml")).parent
STATE_FILE = CONFIG_DIR / "window.json"
REQUIRED = (("Gtk", "4.0"), ("Adw", "1"), ("WebKit", "6.0"), ("JavaScriptCore", "6.0"))

APPARMOR_PROFILE = "jarvis-desktop"

# WebKitGTK's DMA-BUF renderer shows a blank window on many NVIDIA drivers.
if Path("/proc/driver/nvidia").exists():
    os.environ.setdefault("WEBKIT_DISABLE_DMABUF_RENDERER", "1")


def userns_blocked():
    """True when Ubuntu's AppArmor policy will stop WebKit's bubblewrap sandbox:
    unprivileged user namespaces are restricted and we are not running under
    the jarvis-desktop profile (installed by `make install`)."""
    try:
        restricted = Path("/proc/sys/kernel/apparmor_restrict_unprivileged_userns").read_text().strip() == "1"
    except OSError:
        return False
    if not restricted:
        return False
    try:
        label = Path("/proc/self/attr/current").read_text().strip()
    except OSError:
        label = "unconfined"
    # Any non-default profile may allow userns (e.g. the jarvis-desktop one); only plain
    # "unconfined" is known to be blocked.
    return label == "unconfined"


def configure_sandbox():
    if userns_blocked() and "WEBKIT_DISABLE_SANDBOX_THIS_IS_DANGEROUS" not in os.environ:
        # Without this WebKit's web process cannot start and the window dies immediately.
        # Jarvis only renders its own localhost UI and opens other sites in the browser,
        # but the sandbox should be restored by installing the AppArmor profile.
        print("jarvis-desktop: AppArmor blocks user namespaces and the jarvis-desktop profile is not "
              "active; running WebKit without its sandbox. Run `make install` to install the profile.",
              file=sys.stderr)
        os.environ["WEBKIT_DISABLE_SANDBOX_THIS_IS_DANGEROUS"] = "1"


def dependencies_ok():
    try:
        import gi
        for namespace, version in REQUIRED:
            gi.require_version(namespace, version)
        from gi.repository import Adw, Gtk, JavaScriptCore, WebKit  # noqa: F401
        return True
    except (ImportError, ValueError):
        return False


def read_yaml(path):
    try:
        import yaml  # python3-yaml ships with Ubuntu; optional
        return yaml.safe_load(path.read_text()) or {}
    except Exception:
        return {}


def client_settings():
    """Where to connect: JARVIS_URL/JARVIS_TOKEN, else ~/.config/jarvis/client.yaml,
    else the local server from config.yaml. Mirrors local_agent/client.py (which
    this file cannot import: it runs under the system Python)."""
    data = read_yaml(CONFIG_DIR / "client.yaml")
    url = os.environ.get("JARVIS_URL") or data.get("url")
    if not url:
        server = read_yaml(CONFIG_DIR / "config.yaml").get("server", {})
        url = f"http://{server.get('host', '127.0.0.1')}:{server.get('port', 8765)}"
    token = os.environ.get("JARVIS_TOKEN") or data.get("token") or (
        os.environ.get(data["token_env"]) if data.get("token_env") else None)
    return url.rstrip("/"), token


def is_local_url(url):
    import ipaddress
    from urllib.parse import urlparse
    host = urlparse(url).hostname or ""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def backend_up(url):
    try:
        with urlopen(url + "/api/version", timeout=3) as response:
            return response.status < 400
    except Exception:
        return False


def ensure_backend(url, wait=12.0):
    if backend_up(url):
        return True
    if not is_local_url(url):
        return False  # a remote server is not ours to start
    subprocess.run(["systemctl", "--user", "start", "jarvis.service"], check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        if backend_up(url):
            return True
        time.sleep(0.3)
    return False


def bridge_js(token):
    # Token and hostname let app.js authenticate and decide whether the native picker's
    # paths mean anything to the server (only if tools run on this same machine).
    return BRIDGE_JS.replace("__CONFIG__", json.dumps({"token": token, "hostname": socket.gethostname()}))


# Injected into every page before it loads. Exposes the native bridge to app.js.
BRIDGE_JS = """
window.jarvisDesktop = {
  native: true,
  ...__CONFIG__,
  pick(kind = 'folder', multiple = true) {
    return window.webkit.messageHandlers.jarvis.postMessage({cmd: 'pick', kind, multiple});
  },
};
"""

# Tron-styled window chrome (see docs/THEME.md).
WINDOW_CSS = """
window.jarvis, window.jarvis headerbar { background: #050b11; color: #dcecf3; }
window.jarvis headerbar { box-shadow: inset 0 -1px #17333f; min-height: 38px; }
window.jarvis headerbar .title { font-family: 'Space Grotesk', sans-serif; font-weight: 700;
  letter-spacing: 0.3em; color: #45e4f4; }
window.jarvis headerbar windowcontrols button { color: #75c9d7; }
window.jarvis headerbar windowcontrols button:hover { color: #45e4f4; background: #12323d; }
window.jarvis .offline { color: #8ba6b3; }
window.jarvis .offline-title { color: #ff9e3d; font-weight: 700; letter-spacing: 0.15em; }
window.jarvis button.retry { color: #050b11; background: #45e4f4; border-radius: 6px; }
"""

OFFLINE_HINT = "journalctl --user -u jarvis.service -n 50"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Jarvis desktop app")
    parser.add_argument("--url", default=None, help="Backend URL (default: from config)")
    parser.add_argument("--check", action="store_true", help="Exit 0 if desktop dependencies are installed")
    args = parser.parse_args(argv)
    if args.check:
        return 0 if dependencies_ok() else 3
    if not dependencies_ok():
        print("Jarvis desktop needs: sudo apt install python3-gi gir1.2-gtk-4.0 gir1.2-adw-1 gir1.2-webkit-6.0",
              file=sys.stderr)
        return 3

    configure_sandbox()
    import gi
    for namespace, version in REQUIRED:
        gi.require_version(namespace, version)
    from gi.repository import Adw, Gdk, Gio, GLib, Gtk, JavaScriptCore, WebKit

    configured_url, token = client_settings()
    url = (args.url or configured_url).rstrip("/")
    local = is_local_url(url)
    if not local and url.startswith("http://"):
        print(f"jarvis-desktop: warning: {url} is not encrypted; use https:// for a remote server", file=sys.stderr)
    GLib.set_prgname(APP_ID)
    GLib.set_application_name("Jarvis")

    class JarvisWindow(Adw.ApplicationWindow):
        def __init__(self, app):
            super().__init__(application=app, title="Jarvis")
            self.add_css_class("jarvis")
            self.set_icon_name("jarvis")
            state = self.load_state()
            self.set_default_size(state.get("width", 1320), state.get("height", 860))
            if state.get("maximized"):
                self.maximize()

            header = Adw.HeaderBar()
            header.set_title_widget(Gtk.Label(label="JARVIS", css_classes=["title"]))

            manager = WebKit.UserContentManager()
            manager.add_script(WebKit.UserScript.new(
                bridge_js(token), WebKit.UserContentInjectedFrames.TOP_FRAME,
                WebKit.UserScriptInjectionTime.START, None, None))
            manager.register_script_message_handler_with_reply("jarvis", None)
            manager.connect("script-message-with-reply-received::jarvis", self.on_bridge_message)

            self.web = WebKit.WebView(user_content_manager=manager)
            bg = Gdk.RGBA()
            bg.parse("#050b11")
            self.web.set_background_color(bg)
            settings = self.web.get_settings()
            settings.set_enable_developer_extras("--devtools" in sys.argv or bool(os.environ.get("JARVIS_DEVTOOLS")))
            settings.set_javascript_can_access_clipboard(True)
            self.web.connect("decide-policy", self.on_decide_policy)
            self.web.connect("context-menu", self.on_context_menu)
            self.web.connect("load-failed", self.on_load_failed)

            self.stack = Gtk.Stack(transition_type=Gtk.StackTransitionType.CROSSFADE)
            self.stack.add_named(self.web, "web")
            self.stack.add_named(self.build_status_page(), "status")

            box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
            box.append(header)
            self.stack.set_vexpand(True)
            box.append(self.stack)
            self.set_content(box)
            self.connect("close-request", self.on_close)
            self.connect_backend()

        # ---------- backend lifecycle ----------
        def build_status_page(self):
            page = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=14, valign=Gtk.Align.CENTER,
                           halign=Gtk.Align.CENTER, css_classes=["offline"])
            self.status_title = Gtk.Label(label="CONNECTING TO JARVIS…", css_classes=["offline-title"])
            self.status_detail = Gtk.Label(label="", wrap=True, max_width_chars=60, justify=Gtk.Justification.CENTER)
            self.status_spinner = Gtk.Spinner(spinning=True)
            self.retry_button = Gtk.Button(icon_name="view-refresh-symbolic", tooltip_text="Retry",
                                           halign=Gtk.Align.CENTER, css_classes=["retry", "circular"], visible=False)
            self.retry_button.connect("clicked", lambda *_: self.connect_backend())
            for widget in (self.status_spinner, self.status_title, self.status_detail, self.retry_button):
                page.append(widget)
            return page

        def connect_backend(self):
            self.stack.set_visible_child_name("status")
            self.status_title.set_label("CONNECTING TO JARVIS…")
            self.status_detail.set_label(url)
            self.status_spinner.set_spinning(True)
            self.retry_button.set_visible(False)

            def worker():
                ok = ensure_backend(url)
                GLib.idle_add(self.backend_ready if ok else self.backend_failed)
            threading.Thread(target=worker, daemon=True).start()

        def backend_ready(self):
            self.web.load_uri(url + "/")
            self.stack.set_visible_child_name("web")
            return False

        def backend_failed(self):
            self.status_spinner.set_spinning(False)
            self.status_title.set_label("JARVIS BACKEND OFFLINE")
            hint = f"Check the service with: {OFFLINE_HINT}" if local else \
                "Check your network connection and the server URL in ~/.config/jarvis/client.yaml."
            self.status_detail.set_label(f"Could not reach {url}.\n{hint}")
            self.retry_button.set_visible(True)
            return False

        def on_load_failed(self, _web, _event, failing_uri, _error):
            if failing_uri.startswith(url):
                self.backend_failed()
                self.stack.set_visible_child_name("status")
                return True
            return False

        # ---------- browser behaviour ----------
        def on_decide_policy(self, _web, decision, decision_type):
            if decision_type in (WebKit.PolicyDecisionType.NAVIGATION_ACTION, WebKit.PolicyDecisionType.NEW_WINDOW_ACTION):
                target = decision.get_navigation_action().get_request().get_uri()
                if not target.startswith(url) and not target.startswith("about:"):
                    # External links open in the user's browser, never inside Jarvis.
                    Gtk.UriLauncher.new(target).launch(self, None, None, None)
                    decision.ignore()
                    return True
            return False

        def on_context_menu(self, _web, _menu, hit):
            # Keep copy/paste menus for text; hide "Back/Reload/Inspect" chrome elsewhere.
            return not (hit.context_is_editable() or hit.context_is_selection() or hit.context_is_link())

        # ---------- native bridge ----------
        def on_bridge_message(self, _manager, value, reply):
            try:
                message = json.loads(value.to_json(0))
            except (TypeError, ValueError):
                message = {}
            context = value.get_context()

            def respond(payload):
                reply.return_value(JavaScriptCore.Value.new_from_json(context, json.dumps(payload)))

            if message.get("cmd") == "pick":
                self.pick(message.get("kind", "folder"), bool(message.get("multiple", True)), respond)
            else:
                reply.return_error_message("Unknown command")
            return True  # reply is sent asynchronously

        def pick(self, kind, multiple, respond):
            dialog = Gtk.FileDialog(title="Attach folder" if kind == "folder" else "Attach files", modal=True)
            home = Gio.File.new_for_path(str(Path.home()))
            dialog.set_initial_folder(home)

            def done(dlg, result):
                paths = []
                try:
                    if kind == "folder":
                        files = dlg.select_multiple_folders_finish(result) if multiple else [dlg.select_folder_finish(result)]
                    else:
                        files = dlg.open_multiple_finish(result) if multiple else [dlg.open_finish(result)]
                    for item in files:
                        if item is not None and item.get_path():
                            paths.append(item.get_path())
                except GLib.Error:
                    paths = []  # cancelled
                respond({"paths": paths})

            if kind == "folder":
                (dialog.select_multiple_folders if multiple else dialog.select_folder)(self, None, done)
            else:
                (dialog.open_multiple if multiple else dialog.open)(self, None, done)

        # ---------- window state ----------
        def load_state(self):
            try:
                return json.loads(STATE_FILE.read_text())
            except (OSError, ValueError):
                return {}

        def on_close(self, *_):
            try:
                state = self.load_state()
                if not self.is_maximized() and not self.is_fullscreen():
                    state.update(width=self.get_width(), height=self.get_height())
                state["maximized"] = self.is_maximized()
                STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
                STATE_FILE.write_text(json.dumps(state))
            except OSError:
                pass
            return False

    class JarvisApp(Adw.Application):
        def __init__(self):
            super().__init__(application_id=APP_ID, flags=Gio.ApplicationFlags.DEFAULT_FLAGS)
            self.connect("activate", self.on_activate)
            self.connect("startup", self.on_startup)

        def on_startup(self, *_):
            # On SIGTERM/SIGINT/SIGHUP close windows through GTK so the compositor sees a normal
            # unmap. Dying mid-frame has crashed GNOME Shell 46 (tiling-assistant + mutter
            # meta_window_get_workspaces assertion), taking the whole session down.
            for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
                GLib.unix_signal_add(GLib.PRIORITY_HIGH, sig, self.graceful_quit)
            Adw.StyleManager.get_default().set_color_scheme(Adw.ColorScheme.FORCE_DARK)
            Gtk.Window.set_default_icon_name("jarvis")
            css = Gtk.CssProvider()
            css.load_from_string(WINDOW_CSS)
            Gtk.StyleContext.add_provider_for_display(Gdk.Display.get_default(), css, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)
            actions = {
                "quit": (lambda *_: self.quit(), ["<Control>q"]),
                "reload": (lambda *_: self.window().web.reload(), ["<Control>r", "F5"]),
                "new-chat": (lambda *_: self.run_js("newChat()"), ["<Control>n"]),
                "toggle-sidebar": (lambda *_: self.run_js("toggleSidebar()"), ["<Control>b"]),
                "fullscreen": (lambda *_: self.toggle_fullscreen(), ["F11"]),
            }
            for name, (handler, accels) in actions.items():
                action = Gio.SimpleAction.new(name, None)
                action.connect("activate", handler)
                self.add_action(action)
                self.set_accels_for_action("app." + name, accels)

        def window(self):
            return self.get_active_window()

        def graceful_quit(self, *_):
            for win in list(self.get_windows()):
                win.close()  # runs close-request (saves state) and unmaps cleanly
            # Let the compositor process the unmap before the process exits.
            GLib.timeout_add(250, lambda: self.quit() or False)
            return GLib.SOURCE_REMOVE

        def run_js(self, code):
            win = self.window()
            if win:
                win.web.evaluate_javascript(code, -1, None, None, None, None)

        def toggle_fullscreen(self):
            win = self.window()
            if win:
                win.unfullscreen() if win.is_fullscreen() else win.fullscreen()

        def on_activate(self, *_):
            # Single instance: launching again just raises the existing window.
            (self.window() or JarvisWindow(self)).present()

    return JarvisApp().run([sys.argv[0]])


if __name__ == "__main__":
    sys.exit(main())
