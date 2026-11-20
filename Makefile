PYTHON ?= python3
VENV := .venv
BIN := $(VENV)/bin
.PHONY: help provision install serve desktop test uninstall clean
help:
	@echo 'make provision   Create virtual environment and install dependencies'
	@echo 'make install     Install CLI and desktop launcher in ~/.local'
	@echo 'make serve       Run backend in foreground'
	@echo 'make desktop     Open desktop UI (backend must be running)'
	@echo 'make test        Run tests'
	@echo 'make uninstall   Remove launchers and user service (keep chats)'
provision:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -e .
	mkdir -p $(HOME)/.config/jarvis
	@test -f $(HOME)/.config/jarvis/config.yaml || cp config.example.yaml $(HOME)/.config/jarvis/config.yaml
install: provision
	mkdir -p $(HOME)/.local/bin $(HOME)/.local/share/applications $(HOME)/.config/systemd/user
	ln -sfn $(abspath $(BIN)/jarvis) $(HOME)/.local/bin/jarvis
	@sed 's|@@EXEC@@|$(abspath $(BIN)/jarvis)|g' packaging/jarvis.service.in > $(HOME)/.config/systemd/user/jarvis.service
	@sed 's|@@EXEC@@|$(abspath $(BIN)/jarvis)|g' packaging/jarvis.desktop.in > $(HOME)/.local/share/applications/jarvis.desktop
	systemctl --user daemon-reload
	systemctl --user enable --now jarvis.service
	@echo 'Installed. Run jarvis desktop or launch Jarvis from applications.'
serve:
	$(BIN)/jarvis serve
desktop:
	$(BIN)/jarvis desktop
test:
	$(BIN)/pip install pytest
	$(BIN)/pytest -q
uninstall:
	-systemctl --user disable --now jarvis.service
	rm -f $(HOME)/.local/bin/jarvis $(HOME)/.config/systemd/user/jarvis.service $(HOME)/.local/share/applications/jarvis.desktop
	systemctl --user daemon-reload
clean:
	rm -rf $(VENV) .pytest_cache
