"""
Main entry point for Khmer space injection RNN training and inference
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from src.net import KhmerRNN, KhmerGRU
from src.dataloader import create_dataloader, load_data
from src.utils import set_seed, build_vocab, save_model, load_model


def train(args):
    """
    Train the model
    
    Args:
        args: Command line arguments
    """
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    texts, labels = load_data(args.data_path)
    
    if not texts:
        print(f"Error: No data loaded from {args.data_path}")
        print("Please provide training data in the expected format.")
        return
    
    # Build vocabulary
    print("Building vocabulary...")
    char_to_index, index_to_char = build_vocab(texts)
    vocab_size = len(char_to_index)
    print(f"Vocabulary size: {vocab_size}")
    
    # Create dataloader
    print("Creating dataloader...")
    dataloader = create_dataloader(
        texts=texts,
        labels=labels,
        char_to_index=char_to_index,
        batch_size=args.batch_size,
        max_length=args.max_length,
        shuffle=True
    )
    
    # Initialize model
    print("Initializing model...")
    if args.model == 'rnn':
        model = KhmerRNN(
            vocab_size=vocab_size,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            bidirectional=args.bidirectional
        )
    elif args.model == 'gru':
        model = KhmerGRU(
            vocab_size=vocab_size,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            bidirectional=args.bidirectional
        )
    else:
        raise ValueError(f"Unknown model type: {args.model}")
    
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Reshape for loss calculation
            outputs = outputs.view(-1, 2)
            targets = targets.view(-1)
            
            # Compute loss
            loss = criterion(outputs, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % args.log_interval == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}], "
                      f"Batch [{batch_idx+1}/{len(dataloader)}], "
                      f"Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{args.epochs}], Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = Path(args.output_dir) / f"checkpoint_epoch_{epoch+1}.pt"
            save_model(model, str(checkpoint_path))
    
    # Save final model
    final_model_path = Path(args.output_dir) / "final_model.pt"
    save_model(model, str(final_model_path))
    print("Training completed!")


def inference(args):
    """
    Run inference on input text
    
    Args:
        args: Command line arguments
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # TODO: Implement inference logic
    print("Inference mode - not yet implemented")
    print(f"Input text: {args.input}")


def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser(description='Khmer Space Injection RNN')
    
    # Mode
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'inference'],
                        help='Mode: train or inference')
    
    # Data
    parser.add_argument('--data-path', type=str, default='data/train.txt',
                        help='Path to training data')
    parser.add_argument('--max-length', type=int, default=128,
                        help='Maximum sequence length')
    
    # Model
    parser.add_argument('--model', type=str, default='rnn', choices=['rnn', 'gru'],
                        help='Model type: rnn or gru')
    parser.add_argument('--embedding-dim', type=int, default=128,
                        help='Embedding dimension')
    parser.add_argument('--hidden-dim', type=int, default=256,
                        help='Hidden dimension')
    parser.add_argument('--num-layers', type=int, default=2,
                        help='Number of RNN layers')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout probability')
    parser.add_argument('--bidirectional', action='store_false', default=True,
                        help='Disable bidirectional RNN (enabled by default)')
    
    # Training
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='Log interval')
    parser.add_argument('--save-interval', type=int, default=5,
                        help='Save interval')
    parser.add_argument('--output-dir', type=str, default='outputs',
                        help='Output directory')
    
    # Inference
    parser.add_argument('--input', type=str, default='',
                        help='Input text for inference')
    parser.add_argument('--model-path', type=str, default='outputs/final_model.pt',
                        help='Path to trained model')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    if args.mode == 'train':
        train(args)
    elif args.mode == 'inference':
        inference(args)


if __name__ == '__main__':
    main()
