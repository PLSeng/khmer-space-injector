"""
Neural network models for Khmer space injection
"""

import torch
import torch.nn as nn
from typing import Tuple


class KhmerRNN(nn.Module):
    """
    RNN model for Khmer space injection
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        """
        Initialize RNN model
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of character embeddings
            hidden_dim: Dimension of hidden states
            num_layers: Number of RNN layers
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional RNN
        """
        super(KhmerRNN, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.rnn = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Output layer for binary classification (space or no space)
        rnn_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(rnn_output_dim, 2)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            
        Returns:
            Output tensor of shape (batch_size, sequence_length, 2)
        """
        # Embedding layer
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        embedded = self.dropout(embedded)
        
        # RNN layer
        rnn_output, _ = self.rnn(embedded)  # (batch_size, seq_len, hidden_dim * 2)
        rnn_output = self.dropout(rnn_output)
        
        # Fully connected layer
        output = self.fc(rnn_output)  # (batch_size, seq_len, 2)
        
        return output


class KhmerGRU(nn.Module):
    """
    GRU model for Khmer space injection
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        """
        Initialize GRU model
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of character embeddings
            hidden_dim: Dimension of hidden states
            num_layers: Number of GRU layers
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional GRU
        """
        super(KhmerGRU, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.gru = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Output layer for binary classification (space or no space)
        gru_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(gru_output_dim, 2)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            
        Returns:
            Output tensor of shape (batch_size, sequence_length, 2)
        """
        # Embedding layer
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        embedded = self.dropout(embedded)
        
        # GRU layer
        gru_output, _ = self.gru(embedded)  # (batch_size, seq_len, hidden_dim * 2)
        gru_output = self.dropout(gru_output)
        
        # Fully connected layer
        output = self.fc(gru_output)  # (batch_size, seq_len, 2)
        
        return output
