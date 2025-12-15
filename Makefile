.PHONY: help install install-torch clean train test lint format

# Default target
help:
	@echo "Available commands:"
	@echo "  make install       - Create conda environment and install dependencies"
	@echo "  make install-torch - Install PyTorch with CUDA support"
	@echo "  make train         - Train the model"
	@echo "  make test          - Run tests"
	@echo "  make lint          - Run linter"
	@echo "  make format        - Format code"
	@echo "  make clean         - Clean generated files"

# Create conda environment and install dependencies
install:
	conda create -n khmer-space-injector python=3.10 -y
	@echo "Conda environment created. Activate it with: conda activate khmer-space-injector"
	@echo "Then run: make install-torch"

# Install PyTorch with CUDA support
install-torch:
	pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
	pip install numpy scikit-learn pytest black flake8 matplotlib wandb

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
