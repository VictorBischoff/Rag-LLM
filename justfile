# Justfile for RAG-LLM project testing and quality gates

# Default target
default:
    @echo "RAG-LLM Project Just Commands"
    @echo "============================="
    @echo ""
    @echo "Testing:"
    @echo "  just test              - Run all tests"
    @echo "  just test-unit         - Run unit tests only"
    @echo "  just test-integration  - Run integration tests only"
    @echo "  just test-e2e          - Run end-to-end tests only"
    @echo "  just test-fast         - Run fast tests (exclude slow/gpu markers)"
    @echo "  just test-full         - Run all tests including slow ones"
    @echo ""
    @echo "Quality:"
    @echo "  just lint              - Run linting with ruff"
    @echo "  just format            - Format code with ruff"
    @echo "  just coverage          - Run tests with coverage report"
    @echo "  just coverage-html     - Generate HTML coverage report"
    @echo "  just quality-gate      - Run full quality gate (lint + test + coverage)"
    @echo ""
    @echo "Setup:"
    @echo "  just install-dev       - Install development dependencies"
    @echo "  just clean             - Clean up generated files"

# Install development dependencies
install-dev:
    pip install -r requirements-dev.txt

# Run all tests
test:
    PYTHONPATH=. pytest

# Run unit tests only
test-unit:
    PYTHONPATH=. pytest tests/unit/ -m "unit"

# Run integration tests only
test-integration:
    PYTHONPATH=. pytest tests/integration/ -m "integration"

# Run end-to-end tests only
test-e2e:
    PYTHONPATH=. pytest tests/e2e/ -m "e2e"

# Run fast tests (exclude slow and gpu markers)
test-fast:
    PYTHONPATH=. pytest -m "not slow and not gpu"

# Run all tests including slow ones
test-full:
    PYTHONPATH=. pytest -m "not gpu"

# Run linting
lint:
    ruff check .
    ruff check --select I .  # Import sorting

# Format code
format:
    ruff format .
    ruff check --fix .

# Run tests with coverage
coverage:
    PYTHONPATH=. pytest --cov=. --cov-report=term-missing --cov-fail-under=80

# Generate HTML coverage report
coverage-html:
    PYTHONPATH=. pytest --cov=. --cov-report=html:htmlcov --cov-fail-under=80
    @echo "HTML coverage report generated in htmlcov/index.html"

# Full quality gate
quality-gate: lint coverage
    @echo "✅ Quality gate passed!"

# Clean up generated files
clean:
    rm -rf htmlcov/
    rm -rf .coverage
    rm -rf .pytest_cache/
    rm -rf __pycache__/
    rm -rf tests/__pycache__/
    rm -rf tests/*/__pycache__/
    find . -name "*.pyc" -delete
    find . -name "*.pyo" -delete
    find . -name ".coverage" -delete

# Development workflow
dev-setup: install-dev
    @echo "Development environment setup complete!"
    @echo "Run 'just test-fast' to verify everything works"

# CI/CD pipeline simulation
ci-test: lint test-fast coverage
    @echo "✅ CI pipeline passed!"

# Pre-commit hook (can be used with pre-commit)
pre-commit: format lint test-fast
    @echo "✅ Pre-commit checks passed!"
