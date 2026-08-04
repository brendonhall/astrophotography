# Convenience targets. Requires uv on PATH for `make setup`.
export UV_PROJECT_ENVIRONMENT := $(HOME)/.venvs/astrophotography
PY   := uv run python

.PHONY: setup inspect run run-starless clean help test

test:
	uv run pytest -q

help:
	@echo "make setup                  - create env (~/.venvs/astrophotography) and install deps via uv"
	@echo "make inspect FITS=path      - print FITS header + stats"
	@echo "make run FITS=path [V=label]- run full pipeline -> output/<name>_<label>"
	@echo "make run-starless FITS=path [V=label] - full pipeline w/ StarNet2 starless finish"
	@echo "make clean                  - remove work/ intermediates"

setup:
	uv sync

inspect:
	$(PY) scripts/inspect_fits.py "$(FITS)"

run:
	scripts/run_pipeline.sh "$(FITS)" $(V)

run-starless:
	scripts/run_pipeline.sh "$(FITS)" $(V) --starless

clean:
	rm -rf work
