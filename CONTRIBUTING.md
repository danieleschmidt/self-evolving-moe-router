# Contributing to Self-Evolving MoE-Router

Thank you for your interest in contributing! This document outlines how to contribute to the project.

## Development Setup

1. Fork and clone the repository:
```bash
git clone https://github.com/yourusername/self-evolving-moe-router.git
cd self-evolving-moe-router
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install development dependencies:
```bash
pip install -e ".[dev]"
```

4. Set up pre-commit hooks:
```bash
pre-commit install
```

## Code Style

- Follow PEP 8 guidelines
- Use Black for formatting (88 character line length)
- Use isort for import sorting
- Type hints are required for all public functions
- Docstrings required for all public classes and functions

## Testing

Run tests with:
```bash
pytest
```

For coverage report:
```bash
pytest --cov=self_evolving_moe --cov-report=html
```

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes
3. Add tests for new functionality
4. Ensure all tests pass
5. Update documentation if needed
6. Submit a pull request

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).

## Questions?

Open an issue or start a discussion on GitHub.