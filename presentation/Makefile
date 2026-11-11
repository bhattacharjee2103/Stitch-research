PYTHON ?= $(abspath ../.venv/bin/python)
.PHONY: slides clean
slides:
	latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build Stitch_IITR_Presentation.tex
	cp build/Stitch_IITR_Presentation.pdf Stitch_IITR_Presentation.pdf
	$(PYTHON) ../scripts/export_presentation.py
clean:
	latexmk -C -outdir=build Stitch_IITR_Presentation.tex
