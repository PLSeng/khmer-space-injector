.PHONY: help install install-torch clean train test lint format

# Default target
help:
	@echo "Available commands:"
	@echo "  make install       - Create conda environment and install dependencies"
	@echo "  make install-dependencies - Install PyTorch with CUDA support"
	@echo "  make train         - Train the model"
	@echo "  make test          - Run tests"
	@echo "  make lint          - Run linter"
	@echo "  make format        - Format code"

# Create conda environment and install dependencies
install:
	conda create -n khmer-space-injector python=3.11 -y
	@echo "Conda environment created. Activate it with: conda activate khmer-space-injector"
	@echo "Then run: make install-torch"

# Install PyTorch with CUDA support
install-dependencies:
	pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
	pip install -r requirements.txt

# Train the model
train:
	python main.py --mode train

# Run tests
test:
	pytest test/ -v

# Run linter
lint:
	flake8 src/ main.py --max-line-length=100

# Format code
format:
	black src/ main.py --line-length=100