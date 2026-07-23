# Convenience targets. Requires python3 on PATH for `make setup`.
VENV := .venv
PY   := $(VENV)/bin/python

.PHONY: setup inspect run clean help

help:
	@echo "make setup                  - create .venv and install requirements"
	@echo "make inspect FITS=path      - print FITS header + stats"
	@echo "make run FITS=path [V=label]- run full pipeline -> output/<name>_<label>"
	@echo "make clean                  - remove work/ intermediates"

setup:
	python3 -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt
	@# Keep the venv out of Dropbox sync (macOS; harmless if Dropbox absent).
	-xattr -w com.dropbox.ignored 1 $(VENV) 2>/dev/null || true
	@echo "Environment ready in $(VENV)"

inspect:
	$(PY) scripts/inspect_fits.py "$(FITS)"

run:
	scripts/run_pipeline.sh "$(FITS)" $(V)

clean:
	rm -rf work
