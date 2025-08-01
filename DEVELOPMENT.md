# Development Guide

## Project Structure

```
self-evolving-moe-router/
├── src/self_evolving_moe/           # Main package
│   ├── evolution/                   # Evolution algorithms
│   ├── routing/                     # Routing implementations
│   ├── experts/                     # Expert architectures
│   ├── training/                    # Training utilities
│   └── visualization/               # Visualization tools
├── tests/                          # Test suite
├── examples/                       # Usage examples
├── benchmarks/                     # Performance benchmarks
├── docs/                          # Documentation
└── configs/                       # Configuration files
```

## Development Environment

### Prerequisites
- Python 3.8+
- PyTorch 1.12+
- CUDA (optional, for GPU acceleration)

### Setup
```bash
# Clone repository
git clone https://github.com/terragon-labs/self-evolving-moe-router.git
cd self-evolving-moe-router

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install in development mode
pip install -e ".[dev,viz,distributed]"

# Setup pre-commit hooks
pre-commit install
```

## Testing

### Running Tests
```bash
# All tests
pytest

# Specific test file
pytest tests/test_evolution.py

# With coverage
pytest --cov=self_evolving_moe

# Integration tests only
pytest -m integration
```

### Test Categories
- Unit tests: Individual component testing
- Integration tests: End-to-end workflows
- Performance tests: Benchmarking evolution speed
- Hardware tests: GPU/CPU compatibility

## Code Quality

### Linting and Formatting
```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint
flake8 src/ tests/

# Type checking
mypy src/
```

### Pre-commit Hooks
Automatically run on commit:
- Black formatting
- isort import sorting
- flake8 linting
- mypy type checking
- Security scanning

## Building Documentation

```bash
cd docs/
make html
```

## Performance Profiling

```bash
# Profile evolution performance
python -m cProfile -o profile_output scripts/profile_evolution.py

# Analyze profile
python -c "import pstats; pstats.Stats('profile_output').sort_stats('cumulative').print_stats(20)"
```

## Debugging Evolution

### Common Issues
1. **Slow convergence**: Check fitness function, increase mutation rate
2. **Poor diversity**: Increase population size, adjust selection pressure
3. **Memory issues**: Reduce batch size, use gradient checkpointing
4. **Unstable training**: Lower learning rate, add regularization

### Debug Tools
```python
from self_evolving_moe.debug import EvolutionDebugger

debugger = EvolutionDebugger()
debugger.attach_to_evolver(evolver)
debugger.monitor_diversity()
debugger.log_fitness_landscape()
```

## Contributing New Features

### Evolution Algorithms
1. Implement base class from `evolution.base.EvolutionStrategy`
2. Add unit tests in `tests/evolution/`
3. Add benchmark comparison
4. Update documentation

### Expert Architectures
1. Inherit from `experts.base.BaseExpert`
2. Implement slimmable interface if applicable
3. Add architecture-specific tests
4. Include usage examples

### Routing Strategies
1. Extend `routing.base.BaseRouter`
2. Implement hardware-aware features
3. Add routing efficiency benchmarks
4. Document routing patterns

## Release Process

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Create release branch
4. Run full test suite
5. Build and test package
6. Tag release
7. Deploy to PyPI

## Troubleshooting

### Environment Issues
```bash
# Reset environment
rm -rf venv/
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

### CUDA Issues
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Install CUDA-specific PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Memory Issues
- Reduce batch size in evolution configs
- Use gradient checkpointing
- Enable mixed precision training
- Monitor GPU memory usage

## Getting Help

- GitHub Issues: Bug reports and feature requests
- Discussions: General questions and ideas
- Discord: Real-time community chat
- Email: Direct maintainer contact