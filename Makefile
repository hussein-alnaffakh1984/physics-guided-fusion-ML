# =====================================================================
# HIF-ML-Reliability -- convenience Makefile
# =====================================================================

.PHONY: help install test test-quick all clean lint

help:
	@echo "Common targets:"
	@echo "  make install     -- install Python dependencies"
	@echo "  make test        -- run the full test suite"
	@echo "  make test-quick  -- run only the fast (non-ML) tests"
	@echo "  make all         -- run all paper experiments end-to-end"
	@echo "  make lint        -- run black and ruff"
	@echo "  make clean       -- remove generated caches and figures"

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v

test-quick:
	pytest tests/test_physics.py tests/test_data.py -v

all:
	@echo ">>> Step 1: random-split benchmark"
	python -m src.evaluation.run_random_splits
	@echo ">>> Step 2: extrapolation benchmark"
	python -m src.evaluation.run_extrapolation
	@echo ">>> Step 3: leave-one-system-out"
	python -m src.evaluation.run_loso
	@echo ">>> Step 4: uncertainty quantification"
	python -m src.evaluation.run_uncertainty

lint:
	black --check src/ tests/
	ruff check src/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	rm -rf .coverage htmlcov/ build/ dist/ *.egg-info
