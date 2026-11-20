"""Verify the running service is the installed Jarvis version."""
import json
import time
from urllib.request import urlopen

for attempt in range(20):
    try:
        with urlopen("http://127.0.0.1:8765/api/version", timeout=2) as response:
            info = json.load(response)
        if "projects" not in info.get("features", []):
            raise RuntimeError("Running Jarvis backend lacks projects support")
        print("Jarvis service ready; projects API available")
        break
    except Exception:
        if attempt == 19:
            raise SystemExit("Jarvis service did not become ready. Check: journalctl --user -u jarvis.service -n 50")
        time.sleep(0.5)
