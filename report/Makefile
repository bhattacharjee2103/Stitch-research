PYTHON ?= $(abspath ../.venv/bin/python)
.PHONY: all thesis clean
all: thesis
thesis:
	cd .. && $(PYTHON) scripts/generate_artifacts.py
	$(PYTHON) ../scripts/generate_extended_evidence.py
	$(PYTHON) ../scripts/prepare_iitr_assets.py
	latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
	cp build/main.pdf thesis.pdf
	$(PYTHON) ../scripts/verify_thesis.py
clean:
	latexmk -C -outdir=build main.tex
