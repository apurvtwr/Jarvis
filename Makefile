PYTHON ?= python3
VENV := .venv
BIN := $(VENV)/bin
.PHONY: help provision install update-models check serve desktop test uninstall clean
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
	sudo apt-get install -y python3-venv python3-pip curl ca-certificates
	env -u PYTHONPATH $(PYTHON) -m venv $(VENV)
	env -u PYTHONPATH $(BIN)/python -m pip install --upgrade pip
	env -u PYTHONPATH $(BIN)/python -m pip install -e .
	env -u PYTHONPATH $(BIN)/python -m pip check
	env -u PYTHONPATH $(BIN)/python -c "import yaml, httpx, fastapi, uvicorn"
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
install: provision
	mkdir -p $(HOME)/.local/bin $(HOME)/.local/share/applications $(HOME)/.config/systemd/user $(HOME)/.local/share/icons/hicolor/scalable/apps
	cp packaging/jarvis.svg $(HOME)/.local/share/icons/hicolor/scalable/apps/jarvis.svg
	@for size in 64 128 256; do mkdir -p $(HOME)/.local/share/icons/hicolor/$${size}x$${size}/apps; cp packaging/jarvis-$$size.png $(HOME)/.local/share/icons/hicolor/$${size}x$${size}/apps/jarvis.png; done
	ln -sfn $(abspath $(BIN)/jarvis) $(HOME)/.local/bin/jarvis
	@sed 's|@@EXEC@@|$(abspath $(BIN)/jarvis)|g' packaging/jarvis.service.in > $(HOME)/.config/systemd/user/jarvis.service
	@sed 's|@@EXEC@@|$(abspath $(BIN)/jarvis)|g' packaging/jarvis.desktop.in > $(HOME)/.local/share/applications/jarvis.desktop
	-gtk-update-icon-cache -f $(HOME)/.local/share/icons/hicolor 2>/dev/null
	systemctl --user daemon-reload
	systemctl --user enable jarvis.service
	systemctl --user restart jarvis.service
	@$(BIN)/python scripts/check_service.py
	@echo 'Installed. Run jarvis desktop or launch Jarvis from applications.'
serve:
	$(BIN)/jarvis serve
desktop:
	$(BIN)/jarvis desktop
test:
	env -u PYTHONPATH $(BIN)/pip install -q pytest
	env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $(BIN)/pytest -q
uninstall:
	-systemctl --user disable --now jarvis.service
	rm -f $(HOME)/.local/bin/jarvis $(HOME)/.config/systemd/user/jarvis.service $(HOME)/.local/share/applications/jarvis.desktop $(HOME)/.local/share/icons/hicolor/scalable/apps/jarvis.svg
	@for size in 64 128 256; do rm -f $(HOME)/.local/share/icons/hicolor/$${size}x$${size}/apps/jarvis.png; done
	systemctl --user daemon-reload
clean:
	rm -rf $(VENV) .pytest_cache
