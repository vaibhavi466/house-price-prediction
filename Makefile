# Makefile for Bangalore House Price Predictor
# Note: Designed to support Windows PowerShell/CMD and Unix-like environments

ifeq ($(OS),Windows_NT)
    VENV_BIN = venv/Scripts
    RM_VENV = if exist venv rmdir /s /q venv
    RM_CACHE = del /s /q /f *.pyc 2>nul & for /d /r . %%d in (__pycache__ .pytest_cache .coverage htmlcov) do @if exist "%%d" rmdir /s /q "%%d" 2>nul || exit 0
else
    VENV_BIN = venv/bin
    RM_VENV = rm -rf venv
    RM_CACHE = find . -type d -name "__pycache__" -exec rm -rf {} + && rm -rf .pytest_cache .coverage htmlcov
endif

PYTHON = $(VENV_BIN)/python
PIP = $(VENV_BIN)/pip
PYTEST = $(VENV_BIN)/pytest
BLACK = $(VENV_BIN)/black
FLAKE8 = $(VENV_BIN)/flake8
ISORT = $(VENV_BIN)/isort
UVICORN = $(VENV_BIN)/uvicorn

.PHONY: setup lint test clean train serve all

setup:
	python -m venv venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

lint:
	$(BLACK) --check .
	$(FLAKE8) .
	$(ISORT) --check-only .

test:
	$(PYTEST) --cov=src --cov=server --cov-report=term-missing tests/

clean:
	$(RM_VENV)
	$(RM_CACHE)

train:
	$(PYTHON) src/train.py

serve:
	$(UVICORN) server.main:app --reload

all: setup lint test train
