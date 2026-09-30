# Contributing to EdgeCascade

Thank you for your interest in contributing to **EdgeCascade**! EdgeCascade is an open-source, high-performance semantic fallback router and hybrid inference engine optimizing for privacy, token cost, and latency.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Running Tests](#running-tests)
- [Style Guide & Linting](#style-guide--linting)
- [Submitting a Pull Request](#submitting-a-pull-request)

---

## Code of Conduct
Please review and adhere to our [Code of Conduct](CODE_OF_CONDUCT.md) in all community interactions.

---

## How to Contribute
- **Reporting Bugs:** Open an issue with reproduction steps, system environment, and expected vs. actual behavior.
- **Suggesting Features:** Open a feature request detailing the use case, proposed API or configuration, and trade-offs.
- **Pull Requests:** PRs for bug fixes, performance optimizations, new provider adapters, or documentation are always welcome!

---

## Development Setup

### Prerequisites
- Python 3.10+
- Docker & Docker Compose
- Optional: Ollama or vLLM installed locally

### Quickstart
```bash
# Clone the repository
git clone https://github.com/your-org/edgecascade.git
cd edgecascade

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run the local server in reload mode
uvicorn edgecascade.main:app --host 0.0.0.0 --port 8080 --reload
```

---

## Running Tests
Run the test suite with `pytest`:

```bash
# Run all unit tests
pytest tests/ -v

# Run with coverage report
pytest --cov=edgecascade tests/ --cov-report=term-missing
```

---

## Style Guide & Linting
We use `ruff` and `black` for formatting and linting:
```bash
ruff check edgecascade tests
black --check edgecascade tests
```

---

## Submitting a Pull Request
1. Fork the repository and create your branch from `main`:
   ```bash
   git checkout -b feature/my-new-feature
   ```
2. Commit your changes with clear, descriptive commit messages.
3. Ensure all tests pass and coverage does not decrease.
4. Push your branch and open a PR against the `main` branch.
