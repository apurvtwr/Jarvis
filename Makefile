PYTHON ?= python3
VENV := .venv
BIN := $(VENV)/bin
.PHONY: help provision apparmor install update-models check serve desktop test uninstall clean
help:
	@echo 'make provision   Install Ubuntu prerequisites, Python packages, Ollama and models'
	@echo 'make install     Provision and install CLI and desktop launcher in ~/.local'
	@echo 'make update-models Pull missing configured Ollama models'
	@echo 'make check       Verify configured Ollama models are available'
	@echo 'make serve       Run backend in foreground'
	@echo 'make desktop     Open desktop UI (backend must be running)'
	@echo 'make test        Run tests'
	@echo 'make uninstall   Remove launchers and user service (keep chats)'
provision:
	sudo apt-get update
	sudo apt-get install -y python3-venv python3-pip curl ca-certificates \
		python3-gi gir1.2-gtk-4.0 gir1.2-adw-1 gir1.2-webkit-6.0 zenity
	env -u PYTHONPATH $(PYTHON) -m venv $(VENV)
	env -u PYTHONPATH $(BIN)/python -m pip install --upgrade pip
	env -u PYTHONPATH $(BIN)/python -m pip install -e .
	env -u PYTHONPATH $(BIN)/python -m pip check
	env -u PYTHONPATH $(BIN)/python -c "import yaml, httpx, fastapi, uvicorn, websockets"
	mkdir -p $(HOME)/.config/jarvis
	@test -f $(HOME)/.config/jarvis/config.yaml || cp config.example.yaml $(HOME)/.config/jarvis/config.yaml
	$(BIN)/python scripts/setup_profile.py
	$(BIN)/python scripts/provision.py
update-models:
	@test -x $(BIN)/python || (echo "Run make provision first"; exit 1)
	$(BIN)/python scripts/provision.py
check:
	@test -x $(BIN)/python || (echo "Run make provision first"; exit 1)
	$(BIN)/python scripts/provision.py --check
DESKTOP_SHELL := $(abspath local_agent/desktop.py)
APPARMOR_PROFILE := /etc/apparmor.d/jarvis-desktop
apparmor:
	@# Ubuntu 24.04 restricts user namespaces; WebKitGTK's sandbox needs them (see packaging/jarvis-desktop.apparmor.in).
	chmod +x $(DESKTOP_SHELL)
	@if [ -d /etc/apparmor.d ] && command -v apparmor_parser >/dev/null; then \
		sed 's|@@DESKTOP@@|$(DESKTOP_SHELL)|g' packaging/jarvis-desktop.apparmor.in | sudo tee $(APPARMOR_PROFILE) >/dev/null && \
		sudo apparmor_parser -r $(APPARMOR_PROFILE) && echo 'AppArmor profile jarvis-desktop loaded'; \
	else echo 'AppArmor not present; skipping profile'; fi
install: provision apparmor
	mkdir -p $(HOME)/.local/bin $(HOME)/.local/share/applications $(HOME)/.config/systemd/user $(HOME)/.local/share/icons/hicolor/scalable/apps
	cp packaging/jarvis.svg $(HOME)/.local/share/icons/hicolor/scalable/apps/jarvis.svg
	@for size in 64 128 256; do mkdir -p $(HOME)/.local/share/icons/hicolor/$${size}x$${size}/apps; cp packaging/jarvis-$$size.png $(HOME)/.local/share/icons/hicolor/$${size}x$${size}/apps/jarvis.png; done
	ln -sfn $(abspath $(BIN)/jarvis) $(HOME)/.local/bin/jarvis
	@sed 's|@@EXEC@@|$(abspath $(BIN)/jarvis)|g' packaging/jarvis.service.in > $(HOME)/.config/systemd/user/jarvis.service
	@sed -e 's|@@EXEC@@|$(abspath $(BIN)/jarvis)|g' -e 's|@@DOCS@@|$(abspath docs/ARCHITECTURE.md)|g' packaging/jarvis-runner.service.in > $(HOME)/.config/systemd/user/jarvis-runner.service
	@sed -e 's|@@EXEC@@|$(abspath $(BIN)/jarvis)|g' -e 's|@@DESKTOP@@|$(DESKTOP_SHELL)|g' packaging/dev.jarvis.Jarvis.desktop.in > $(HOME)/.local/share/applications/dev.jarvis.Jarvis.desktop
	rm -f $(HOME)/.local/share/applications/jarvis.desktop
	-update-desktop-database $(HOME)/.local/share/applications 2>/dev/null
	-gtk-update-icon-cache -f --ignore-theme-index $(HOME)/.local/share/icons/hicolor >/dev/null 2>&1
	systemctl --user daemon-reload
	systemctl --user enable jarvis.service
	systemctl --user restart jarvis.service
	@$(BIN)/python scripts/check_service.py
	@echo 'Installed. Run jarvis desktop or launch Jarvis from applications.'
	@echo 'Remote server with tools.executor: runner? Enable the runner: systemctl --user enable --now jarvis-runner.service'
serve:
	$(BIN)/jarvis serve
desktop:
	$(BIN)/jarvis desktop
test:
	env -u PYTHONPATH $(BIN)/pip install -q pytest
	env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $(BIN)/pytest -q
uninstall:
	-[ -f $(APPARMOR_PROFILE) ] && sudo apparmor_parser -R $(APPARMOR_PROFILE) && sudo rm -f $(APPARMOR_PROFILE)
	-systemctl --user disable --now jarvis.service
	-systemctl --user disable --now jarvis-runner.service
	rm -f $(HOME)/.local/bin/jarvis $(HOME)/.config/systemd/user/jarvis.service $(HOME)/.config/systemd/user/jarvis-runner.service $(HOME)/.local/share/applications/jarvis.desktop $(HOME)/.local/share/applications/dev.jarvis.Jarvis.desktop $(HOME)/.local/share/icons/hicolor/scalable/apps/jarvis.svg
	@for size in 64 128 256; do rm -f $(HOME)/.local/share/icons/hicolor/$${size}x$${size}/apps/jarvis.png; done
	systemctl --user daemon-reload
clean:
	rm -rf $(VENV) .pytest_cache
