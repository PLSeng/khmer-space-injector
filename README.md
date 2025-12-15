# khmer-space-injector
Khmer word segmentation with RNN

## Project Structure

```
khmer-space-injector/
├── main.py              # Entry point for training and inference
├── Makefile             # Development automation
├── requirements.txt     # Python dependencies
└── src/
    ├── __init__.py      # Package initialization
    ├── dataloader.py    # Data loading and preprocessing
    ├── net.py           # Neural network models (RNN, GRU)
    └── utils.py         # Utility functions
```

## Installation

```bash
make install
```

Or manually:
```bash
pip install -r requirements.txt
```

## Usage

### Training

```bash
# Using make
make train

# Using python directly with custom arguments
python main.py --mode train \
    --data-path data/train.txt \
    --model rnn \
    --epochs 10 \
    --batch-size 32
```

### Inference

```bash
python main.py --mode inference \
    --input "YourKhmerTextHere" \
    --model-path outputs/final_model.pt
```

## Data Format

The training data should be in tab-separated format:
```
text_string	0 1 0 1 0
```

Where:
- First column: Input text
- Second column: Space-separated binary labels (1 for space, 0 for no space)

## Available Models

- **RNN (LSTM)**: Bidirectional LSTM for sequence labeling
- **GRU**: Bidirectional GRU for sequence labeling

## Development

### Available Make Commands

- `make install` - Install dependencies
- `make train` - Train the model
- `make test` - Run tests
- `make lint` - Run linter
- `make format` - Format code
- `make clean` - Clean generated files

## Model Configuration

Key hyperparameters (configurable via command line):

- `--embedding-dim`: Character embedding dimension (default: 128)
- `--hidden-dim`: Hidden layer dimension (default: 256)
- `--num-layers`: Number of RNN layers (default: 2)
- `--dropout`: Dropout probability (default: 0.3)
- `--batch-size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 0.001)

