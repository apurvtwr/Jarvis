#!/usr/bin/env python3
"""Provision the locally configured Ollama provider and required models."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlopen

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = Path(os.environ.get('LOCAL_AGENT_CONFIG', Path.home() / '.config/jarvis/config.yaml'))


def run(*args):
    subprocess.run(args, check=True)


def api(base, path):
    with urlopen(base.rstrip('/') + path, timeout=4) as response:
        return json.load(response)


def ready(base):
    try:
        api(base, '/api/tags')
        return True
    except (URLError, TimeoutError, OSError, ValueError):
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    cfg = yaml.safe_load(CONFIG.read_text()) or {}
    providers = cfg.get('providers', {})
    required = {}
    for value in cfg.get('models', {}).values():
        if not isinstance(value, str) or '/' not in value:
            raise SystemExit(f'Invalid model reference: {value!r}; expected provider/model')
        provider, model = value.split('/', 1)
        if provider not in providers:
            raise SystemExit(f'Undefined provider {provider!r}')
        required.setdefault(provider, set()).add(model)
    for provider, settings in providers.items():
        if settings.get('type') != 'ollama':
            print(f'{provider}: external provider; provisioning not supported, skipping')
            continue
        base = settings.get('base_url', 'http://127.0.0.1:11434')
        parsed = urlparse(base)
        local = parsed.hostname in ('127.0.0.1', 'localhost', '::1')
        if not ready(base):
            if args.check:
                raise SystemExit(f'{provider}: Ollama is not reachable at {base}')
            if not local:
                raise SystemExit(f'{provider}: remote Ollama at {base} is unreachable; refusing local installation')
            if not shutil.which('ollama'):
                print('Ollama missing. Installing via official installer (requires sudo)...', flush=True)
                # Download first; do not pipe a remote script directly to a shell.
                import tempfile
                from urllib.request import urlretrieve
                with tempfile.TemporaryDirectory() as tmp:
                    installer = Path(tmp) / 'install.sh'
                    urlretrieve('https://ollama.com/install.sh', installer)
                    run('sh', str(installer))
            if shutil.which('systemctl'):
                subprocess.run(['sudo', 'systemctl', 'enable', '--now', 'ollama'], check=False)
            for _ in range(20):
                if ready(base):
                    break
                time.sleep(1)
            if not ready(base):
                raise SystemExit(f'Ollama is installed but not reachable at {base}; check `systemctl status ollama`')
        installed = {item['name'] for item in api(base, '/api/tags').get('models', [])}
        for model in sorted(required.get(provider, set())):
            aliases = {model, model + ':latest' if ':' not in model else model}
            if installed.intersection(aliases):
                print(f'{provider}/{model}: installed')
            elif args.check:
                raise SystemExit(f'{provider}/{model}: missing')
            else:
                if not local:
                    raise SystemExit(f'{provider}/{model}: missing on remote provider; pull on remote host')
                print(f'Pulling {model} (may take several GB)...', flush=True)
                run('ollama', 'pull', model)
    print('Provider and model checks passed.')


if __name__ == '__main__':
    main()
