PYTHON ?= .venv/bin/python
SUBMAKE_PYTHON := $(if $(findstring /,$(PYTHON)),$(abspath $(PYTHON)),$(PYTHON))
.PHONY: setup data baseline evaluate train export demo profile test thesis presentation documents clean package check-documents
setup:
	bash scripts/setup.sh
data:
	$(PYTHON) scripts/run_logged.py $(PYTHON) scripts/download_data.py
	$(PYTHON) scripts/run_logged.py $(PYTHON) scripts/prepare_data.py
baseline:
	$(PYTHON) scripts/run_logged.py $(PYTHON) scripts/run_originals.py --limit 90
evaluate:
	$(PYTHON) scripts/run_logged.py $(PYTHON) scripts/evaluate.py --learned-exhaustive
train:
	$(PYTHON) scripts/run_logged.py $(PYTHON) scripts/train.py --steps 120
export:
	$(PYTHON) scripts/run_logged.py $(PYTHON) scripts/export_models.py
demo:
	$(PYTHON) scripts/run_logged.py $(PYTHON) scripts/demo.py
profile:
	$(PYTHON) scripts/run_logged.py $(PYTHON) scripts/benchmark_runtime.py
test:
	$(PYTHON) scripts/run_logged.py $(PYTHON) -m pytest -q
thesis:
	$(PYTHON) scripts/run_logged.py $(MAKE) -C report PYTHON="$(SUBMAKE_PYTHON)" thesis
presentation:
	$(PYTHON) scripts/run_logged.py $(PYTHON) scripts/prepare_iitr_assets.py
	$(PYTHON) scripts/run_logged.py $(PYTHON) scripts/generate_presentation_data.py
	$(PYTHON) scripts/run_logged.py $(MAKE) -C presentation PYTHON="$(SUBMAKE_PYTHON)" slides
documents: thesis
	$(MAKE) presentation
package: documents
	$(PYTHON) scripts/run_logged.py $(PYTHON) scripts/verify_iitr_documents.py
	$(PYTHON) scripts/run_logged.py $(PYTHON) scripts/package_iitr_documents.py
check-documents:
	$(PYTHON) scripts/run_logged.py $(PYTHON) scripts/verify_research.py
	$(PYTHON) scripts/run_logged.py $(PYTHON) scripts/verify_model_artifacts.py
	$(PYTHON) scripts/run_logged.py $(PYTHON) scripts/verify_thesis.py
	$(PYTHON) scripts/run_logged.py $(PYTHON) scripts/verify_iitr_documents.py
	$(PYTHON) scripts/run_logged.py $(PYTHON) scripts/check_document_packages.py
clean:
	$(MAKE) -C report clean
	$(MAKE) -C presentation clean
