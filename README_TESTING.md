# Testing Guide for RAG-LLM Project

This document provides comprehensive information about the testing framework for the RAG-LLM project.

## Overview

The project uses a comprehensive testing strategy with multiple layers:

- **Unit Tests**: Test individual components in isolation with mocked dependencies
- **Integration Tests**: Test API endpoints and component interactions
- **End-to-End Tests**: Test complete workflows including CLI and monitoring tools
- **Quality Gates**: Automated linting, formatting, and coverage checks

## Quick Start

### Install Dependencies

```bash
# Install production dependencies only
pip install -r requirements.txt

# Install all dependencies including testing tools
pip install -r requirements-dev.txt

# Or use just
just install-dev
```

### Run Tests

```bash
# Run all tests
just test

# Run only fast tests (excludes slow/GPU tests)
just test-fast

# Run specific test categories
just test-unit
just test-integration
just test-e2e

# Run with coverage
just coverage
```

## Test Structure

```
tests/
├── conftest.py                 # Shared fixtures and utilities
├── unit/                       # Unit tests
│   ├── test_rag_system.py     # OptimizedRAGSystem tests
│   ├── test_api.py            # API endpoint tests
│   ├── test_client_example.py # Client helper tests
│   ├── test_log_monitor.py    # Log monitor tests
│   └── test_ui.py             # Streamlit UI tests
├── integration/                # Integration tests
│   └── test_api_integration.py # Full API workflow tests
└── e2e/                        # End-to-end tests
    ├── test_client_cli.py      # CLI workflow tests
    ├── test_log_monitor.py     # Log monitor E2E tests
    └── test_streamlit_smoke.py # Streamlit UI smoke tests
```

## Test Categories

### Unit Tests (`tests/unit/`)

**Purpose**: Test individual components in isolation with all external dependencies mocked.

**Key Features**:
- Mock PyMuPDFLoader, HuggingFaceEmbeddings, FAISS, and MLXPipeline
- Test cache-key stability and document chunking
- Verify error handling and edge cases
- Test timing statistics and performance tracking

**Run**: `pytest tests/unit/ -m "unit"`

### Integration Tests (`tests/integration/`)

**Purpose**: Test API endpoints and component interactions with stubbed RAG systems.

**Key Features**:
- Use FastAPI TestClient and httpx.AsyncClient
- Test complete upload → query workflows
- Verify concurrent session handling
- Test file caching and cache corruption fallback

**Run**: `pytest tests/integration/ -m "integration"`

### End-to-End Tests (`tests/e2e/`)

**Purpose**: Test complete system workflows including CLI tools and monitoring.

**Key Features**:
- Launch FastAPI in background threads
- Test client_example.py CLI functionality
- Test log_monitor.py with dynamic session data
- Optional Streamlit UI smoke tests

**Run**: `pytest tests/e2e/ -m "e2e"`

## Test Markers

The project uses pytest markers to categorize tests:

- `@pytest.mark.unit` - Unit tests (fast, no external dependencies)
- `@pytest.mark.integration` - Integration tests (may require external services)
- `@pytest.mark.e2e` - End-to-end tests (full system testing)
- `@pytest.mark.slow` - Tests that take longer to run
- `@pytest.mark.gpu` - Tests requiring GPU/MLX hardware
- `@pytest.mark.network` - Tests requiring network access

## Running Tests

### Basic Commands

```bash
# Run all tests
pytest

# Run specific test categories
pytest -m "unit"           # Unit tests only
pytest -m "integration"    # Integration tests only
pytest -m "e2e"           # End-to-end tests only

# Run fast tests (exclude slow/GPU)
pytest -m "not slow and not gpu"

# Run with coverage
pytest --cov=. --cov-report=term-missing --cov-fail-under=80
```

### Using Just

```bash
just test              # Run all tests
just test-fast         # Run fast tests only
just test-unit         # Unit tests only
just test-integration  # Integration tests only
just test-e2e         # End-to-end tests only
just coverage         # Run with coverage
just quality-gate     # Full quality check
```

### Using Tox

```bash
# Run all environments
tox

# Run specific environments
tox -e py313          # Python 3.13 tests
tox -e lint           # Linting only
tox -e format         # Format check
tox -e coverage       # Coverage report
```

## Coverage Requirements

The project enforces a minimum coverage threshold of 80%:

- **Current Target**: 80% line coverage
- **Coverage Report**: Generated in `htmlcov/index.html`
- **CI Integration**: Coverage reports uploaded to Codecov

### View Coverage

```bash
# Generate HTML coverage report
just coverage-html

# Open in browser
open htmlcov/index.html
```

## Quality Gates

### Automated Checks

1. **Linting**: Ruff for code quality and import sorting
2. **Formatting**: Ruff for consistent code style
3. **Testing**: Comprehensive test suite with coverage
4. **Type Checking**: Optional mypy integration

### Pre-commit Hooks

```bash
# Install pre-commit (optional)
pip install pre-commit
pre-commit install

# Run pre-commit checks manually
just pre-commit
```

### CI/CD Pipeline

The GitHub Actions workflow runs:

1. **Lint Check**: Code quality and formatting
2. **Unit Tests**: Fast tests with coverage
3. **Integration Tests**: API workflow tests
4. **Quality Gate**: Full quality assessment

## Test Fixtures and Utilities

### Shared Fixtures (`tests/conftest.py`)

- `temp_dir`: Temporary directory for test files
- `sample_pdf_path`: Dummy PDF file for testing
- `sample_documents`: Mock Document objects
- `mock_pymupdf_loader`: Mocked PDF loader
- `mock_huggingface_embeddings`: Mocked embeddings
- `mock_faiss`: Mocked vector store
- `mock_mlx_pipeline`: Mocked MLX pipeline
- `stubbed_rag_system`: Complete RAG system with mocked dependencies

### Utility Functions

- `create_test_pdf()`: Create minimal PDF files for testing
- `assert_timing_stats_present()`: Verify timing statistics
- `create_mock_upload_file()`: Mock file uploads for API tests

## Mocking Strategy

### External Dependencies

All external dependencies are mocked to ensure tests run without:
- GPU/MLX hardware requirements
- Network access
- Large model downloads
- File system dependencies

### Key Mocks

- **PyMuPDFLoader**: Returns predefined document chunks
- **HuggingFaceEmbeddings**: Mocked embedding model
- **FAISS**: Mocked vector store with retriever
- **MLXPipeline**: Mocked LLM with predictable responses
- **HTTP Requests**: Mocked API calls using responses library

## Performance Testing

### Timing Tests

Tests verify that timing statistics are properly recorded:

```python
def test_timing_stats_present(rag_system):
    expected_operations = [
        "Document Processing",
        "MLX Model Loading",
        "Query Processing"
    ]
    assert_timing_stats_present(rag_system, expected_operations)
```

### Cache Performance

Tests validate caching behavior:

- Cache key stability across instances
- Cache hit/miss scenarios
- Corrupt cache fallback behavior

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Mock Failures**: Check that mocks are properly configured
3. **Timeout Errors**: Increase timeout for slow tests
4. **Coverage Failures**: Add tests for uncovered code paths

### Debug Mode

```bash
# Run tests with verbose output
pytest -v -s

# Run specific test with debugging
pytest tests/unit/test_rag_system.py::TestOptimizedRAGSystem::test_init -v -s

# Run with pdb on failures
pytest --pdb
```

### Test Data

Test data is generated dynamically to avoid committing large files:

- PDF files: Minimal valid PDF structures
- Documents: Synthetic text content
- Responses: Mocked API responses

## Contributing

### Adding New Tests

1. **Unit Tests**: Add to appropriate `tests/unit/test_*.py` file
2. **Integration Tests**: Add to `tests/integration/test_*.py`
3. **E2E Tests**: Add to `tests/e2e/test_*.py`

### Test Guidelines

- Use descriptive test names
- Follow AAA pattern (Arrange, Act, Assert)
- Mock external dependencies
- Test both success and failure cases
- Include edge cases and error conditions

### Coverage Requirements

- New code must maintain 80% coverage
- Add tests for any new functionality
- Update existing tests when modifying behavior

## CI/CD Integration

### GitHub Actions

The CI pipeline runs on every push and PR:

1. **Lint Check**: Code quality validation
2. **Format Check**: Code style validation
3. **Unit Tests**: Fast test suite with coverage
4. **Integration Tests**: API workflow validation
5. **Quality Gate**: Comprehensive quality assessment

### Local CI Simulation

```bash
# Simulate CI pipeline locally
just ci-test

# Run full quality gate
just quality-gate
```

## Advanced Usage

### Custom Test Configurations

Create custom pytest configurations:

```ini
# pytest.ini
[tool:pytest]
addopts = --strict-markers --tb=short
markers =
    custom: Custom test marker
```

### Parallel Testing

```bash
# Install pytest-xdist for parallel testing
pip install pytest-xdist

# Run tests in parallel
pytest -n auto
```

### Test Profiling

```bash
# Profile test execution
pytest --durations=10
```

## Resources

- [pytest Documentation](https://docs.pytest.org/)
- [pytest-asyncio](https://pytest-asyncio.readthedocs.io/)
- [pytest-cov](https://pytest-cov.readthedocs.io/)
- [responses](https://github.com/getsentry/responses)
- [ruff](https://docs.astral.sh/ruff/)
