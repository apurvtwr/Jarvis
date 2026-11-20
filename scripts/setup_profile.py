import json
import os
import sys
from pathlib import Path

profile = Path.home() / ".config/jarvis/profile.json"
profile.parent.mkdir(parents=True, exist_ok=True)
if not profile.exists():
    default = os.environ.get("JARVIS_USER_NAME", "").strip()
    if not default and sys.stdin.isatty():
        default = input("Name to display in Jarvis chats: ").strip()
    if not default:
        default = os.environ.get("USER", "You")
    profile.write_text(json.dumps({"display_name": default}, indent=2) + "\n")
    print(f"Jarvis chat display name: {default}")
