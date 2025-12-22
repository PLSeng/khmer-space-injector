# khmer-space-injector
Khmer word segmentation with RNN

## Project Structure

```
khmer-space-injector/
├── main.py              # Entry point for training and inference
├── Makefile             # Development automation
├── requirements.txt     # Python dependencies
└── src/
    ├── dataloader.py    # Data loading and preprocessing
    ├── net.py           # Neural network models (RNN, GRU, LSTM)
    └── utils.py         # Utility functions
```

## Installation

```bash
make install # Create conda environment and install dependencies
conda activate khmer-space-injector # Activate the environment
```
```bash
make install-dependencies # Install PyTorch with CUDA support and other dependencies
```

Or manually:
```bash
conda create -n khmer-space-injector python=3.11 -y
conda activate khmer-space-injector
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## Usage

**Undergoing development!!!**