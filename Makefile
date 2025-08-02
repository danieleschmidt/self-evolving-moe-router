# Self-Evolving MoE-Router Development Makefile
# Usage: make <target>

.PHONY: help install dev-install test lint format clean build docs serve-docs
.DEFAULT_GOAL := help

# Variables
PYTHON := python3
PIP := pip
PYTEST := pytest
BLACK := black
ISORT := isort
RUFF := ruff
MYPY := mypy
BANDIT := bandit
SAFETY := safety
PRE_COMMIT := pre-commit

# Source directories
SRC_DIR := src
TEST_DIR := tests
DOCS_DIR := docs

# Help target
help: ## Show this help message
	@echo "Self-Evolving MoE-Router Development Makefile"
	@echo ""
	@echo "Available targets:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Installation targets
install: ## Install package in production mode
	$(PIP) install .

dev-install: ## Install package in development mode with all dependencies
	$(PIP) install -e ".[dev,viz,distributed,benchmark]"
	$(PRE_COMMIT) install

install-hooks: ## Install pre-commit hooks
	$(PRE_COMMIT) install
	$(PRE_COMMIT) install --hook-type commit-msg

# Development targets
dev: dev-install ## Set up development environment
	@echo "Development environment setup complete!"
	@echo "Run 'make test' to run tests"
	@echo "Run 'make lint' to run linting"
	@echo "Run 'make format' to format code"

# Testing targets
test: ## Run all tests
	$(PYTEST) $(TEST_DIR) -v

test-cov: ## Run tests with coverage report
	$(PYTEST) $(TEST_DIR) --cov=$(SRC_DIR) --cov-report=html --cov-report=term-missing

test-fast: ## Run fast tests only (excluding slow integration tests)
	$(PYTEST) $(TEST_DIR) -v -m "not slow"

test-slow: ## Run slow integration tests only
	$(PYTEST) $(TEST_DIR) -v -m "slow"

test-parallel: ## Run tests in parallel
	$(PYTEST) $(TEST_DIR) -n auto

# Code quality targets
lint: ## Run all linting tools
	@echo "Running ruff..."
	$(RUFF) check $(SRC_DIR) $(TEST_DIR)
	@echo "Running mypy..."
	$(MYPY) $(SRC_DIR)
	@echo "Running bandit..."
	$(BANDIT) -r $(SRC_DIR)
	@echo "Running safety..."
	$(SAFETY) check

lint-fix: ## Run linting with auto-fix where possible
	$(RUFF) check --fix $(SRC_DIR) $(TEST_DIR)
	$(ISORT) $(SRC_DIR) $(TEST_DIR)
	$(BLACK) $(SRC_DIR) $(TEST_DIR)

format: ## Format code with black and isort
	$(BLACK) $(SRC_DIR) $(TEST_DIR)
	$(ISORT) $(SRC_DIR) $(TEST_DIR)

format-check: ## Check code formatting without making changes
	$(BLACK) --check $(SRC_DIR) $(TEST_DIR)
	$(ISORT) --check-only $(SRC_DIR) $(TEST_DIR)

# Security targets
security: ## Run security checks
	$(BANDIT) -r $(SRC_DIR) -f json -o security-report.json
	$(SAFETY) check --json --output security-deps.json

vulnerability-scan: ## Scan for vulnerabilities in dependencies
	$(SAFETY) check --full-report

# Pre-commit targets
pre-commit: ## Run pre-commit on all files
	$(PRE_COMMIT) run --all-files

pre-commit-update: ## Update pre-commit hooks
	$(PRE_COMMIT) autoupdate

# Build targets
build: clean ## Build distribution packages
	$(PYTHON) -m build

build-wheel: ## Build wheel package only
	$(PYTHON) -m build --wheel

build-sdist: ## Build source distribution only
	$(PYTHON) -m build --sdist

# Documentation targets
docs: ## Build documentation
	cd $(DOCS_DIR) && make html

docs-serve: ## Serve documentation locally
	cd $(DOCS_DIR) && make html && python -m http.server 8000 -d _build/html

docs-clean: ## Clean documentation build
	cd $(DOCS_DIR) && make clean

# Docker targets
docker-build: ## Build Docker image
	docker build -t self-evolving-moe-router .

docker-dev: ## Build development Docker image
	docker build --target development -t self-evolving-moe-router:dev .

docker-run: ## Run Docker container
	docker run -it --rm --gpus all self-evolving-moe-router

docker-shell: ## Open shell in Docker container
	docker run -it --rm --gpus all -v $(PWD):/workspace self-evolving-moe-router:dev bash

# Environment targets
env-check: ## Check Python environment and dependencies
	@echo "Python version:"
	@$(PYTHON) --version
	@echo ""
	@echo "Pip version:"
	@$(PIP) --version
	@echo ""
	@echo "Installed packages:"
	@$(PIP) list | grep -E "(torch|numpy|scipy|matplotlib|pytest|black|ruff|mypy)"

env-freeze: ## Freeze current environment to requirements.txt
	$(PIP) freeze > requirements-frozen.txt

# Data and model management
download-data: ## Download example datasets
	mkdir -p data/datasets
	# Add dataset download commands here

download-models: ## Download pre-trained models
	mkdir -p data/models
	# Add model download commands here

# Cleaning targets
clean: ## Clean build artifacts and cache files
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

clean-logs: ## Clean log files
	rm -rf logs/
	rm -rf wandb/
	rm -rf tensorboard_logs/

clean-all: clean clean-logs ## Clean everything
	rm -rf data/experiments/
	rm -rf results/
	rm -rf .cache/

# Benchmarking and profiling
benchmark: ## Run performance benchmarks
	$(PYTHON) -m self_evolving_moe.benchmarks.run_all

profile: ## Run profiling
	$(PYTHON) -m cProfile -o profile.prof -m self_evolving_moe.cli --help
	$(PYTHON) -c "import pstats; pstats.Stats('profile.prof').sort_stats('cumulative').print_stats(20)"

# Development server targets
jupyter: ## Start Jupyter Lab server
	jupyter lab --ip=0.0.0.0 --no-browser --allow-root

tensorboard: ## Start TensorBoard server
	tensorboard --logdir=logs --host=0.0.0.0

dashboard: ## Start evolution dashboard
	$(PYTHON) -m self_evolving_moe.dashboard

# CI/CD targets
ci-test: ## Run tests in CI environment
	$(PYTEST) $(TEST_DIR) --cov=$(SRC_DIR) --cov-report=xml --cov-report=term

ci-lint: ## Run linting in CI environment
	$(RUFF) check $(SRC_DIR) $(TEST_DIR) --output-format=github
	$(MYPY) $(SRC_DIR) --junit-xml=mypy-report.xml
	$(BANDIT) -r $(SRC_DIR) -f json -o bandit-report.json

ci-build: ## Build for CI environment
	$(PYTHON) -m build
	twine check dist/*

# Release targets
version-patch: ## Bump patch version
	bump2version patch

version-minor: ## Bump minor version
	bump2version minor

version-major: ## Bump major version
	bump2version major

release-test: ## Release to Test PyPI
	twine upload --repository testpypi dist/*

release: ## Release to PyPI
	twine upload dist/*

# Git hooks and workflow
git-hooks: install-hooks ## Install git hooks

commit: ## Interactive commit with pre-commit checks
	$(PRE_COMMIT) run --all-files && git add -A && git commit

# Quick development workflow
quick-check: format lint test-fast ## Quick development check (format, lint, fast tests)

full-check: format lint test security ## Full development check (format, lint, all tests, security)

# Environment info
info: ## Show development environment information
	@echo "=== Development Environment Information ==="
	@echo "Python: $$($(PYTHON) --version)"
	@echo "Pip: $$($(PIP) --version)"
	@echo "Git: $$(git --version)"
	@echo "Current branch: $$(git branch --show-current)"
	@echo "Working directory: $$(pwd)"
	@echo "=== Package Information ==="
	@$(PIP) show self-evolving-moe-router 2>/dev/null || echo "Package not installed"
	@echo "=== GPU Information ==="
	@$(PYTHON) -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')" 2>/dev/null || echo "PyTorch not installed"