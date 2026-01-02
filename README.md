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

## Training

Example command used for the best-performing model:

```bash
python main.py --mode train --train_path data --device cuda --batch_size 128 --epochs 20 --optimizer adamw --lr 0.01 --weight_decay 0 --grad_clip 5 --rnn_type lstm --embedding_dim 128 --hidden_dim 256 --num_layers 2 --dropout 0.3 --max_length 128 --bidirectional True --residual False
```

The vocabulary is automatically built and saved during training.


## Inference

```bash
python main.py --mode infer --device cuda --ckpt_path checkpoints/khmer_rnn.pt --vocab_path checkpoints/vocab.json --text_path infer_test.txt --output_path output_segmented.txt --rnn_type lstm --embedding_dim 128 --hidden_dim 256 --num_layers 2 --dropout 0.3 --max_length 128 --bidirectional True --residual False

```

## Streamlit Demo

Run the interactive demo:

```bash
streamlit run app.py
```

Features:

* Paste Khmer text or upload `.txt` files
* View segmented output
* Download results


## Results (Best Model)

* Token accuracy: **99.8%**
* Boundary F1-score: **≈ 99.5%**
* ROC–AUC: **0.9999**
* PR–AUC (AP): **0.9992**

Evaluation figures (loss curves, confusion matrix, ROC/PR curves) are available in the `Presentation/` folder.

## Notes

* Model: character-level BiLSTM
* Task: binary boundary prediction (space / no space)
* Future work: CRF decoding, out-of-domain evaluation, representation analysis
