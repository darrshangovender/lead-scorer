PYTHON ?= python

.PHONY: install data train eval test notebook clean

install:
	$(PYTHON) -m pip install -e ".[dev]"

data:
	$(PYTHON) -m lead_scorer.sample_generator

train:
	$(PYTHON) -m lead_scorer.pipeline --leads data/leads.csv --out artifacts

eval: train
	@echo "--- metrics ---"
	@cat artifacts/metrics.json

test:
	$(PYTHON) -m pytest

notebook:
	jupyter nbconvert --to notebook --execute notebooks/exploratory.ipynb \
	  --output exploratory.ipynb

clean:
	rm -rf artifacts/ __pycache__/ .pytest_cache/ */__pycache__/ */*/__pycache__/
