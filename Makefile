.PHONY: help install clean train test lint format

# Default target
help:
	@echo "Available commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make train      - Train the model"
	@echo "  make test       - Run tests"
	@echo "  make lint       - Run linter"
	@echo "  make format     - Format code"
	@echo "  make clean      - Clean generated files"

# Install dependencies
install:
	pip install -r requirements.txt

# Train the model
train:
	python main.py --mode train

# Run tests
test:
	pytest tests/ -v

# Run linter
lint:
	flake8 src/ main.py --max-line-length=100

# Format code
format:
	black src/ main.py --line-length=100

# Clean generated files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ outputs/
