"""
Neural network models for Khmer space injection
"""

import torch
import torch.nn as nn
import random


class CRF(nn.Module):
    """
    Conditional Random Field (CRF) layer for sequence labeling
    """
    def __init__(self, num_tags):
        super().__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))

    def forward(self, emissions, tags, mask):
        log_num = self._score_sentence(emissions, tags, mask)
        log_den = self._log_partition(emissions, mask)
        return torch.mean(log_den - log_num)

    def _score_sentence(self, emissions, tags, mask):
        score = self.start_transitions[tags[:, 0]]

        for t in range(emissions.size(1) - 1):
            score += emissions[:, t, tags[:, t]]
            score += self.transitions[tags[:, t], tags[:, t + 1]] * mask[:, t + 1]

        last_idx = mask.sum(1).long() - 1
        last_tags = tags.gather(1, last_idx.unsqueeze(1)).squeeze()
        score += self.end_transitions[last_tags]
        return score

    def _log_partition(self, emissions, mask):
        alpha = self.start_transitions + emissions[:, 0]

        for t in range(1, emissions.size(1)):
            emit = emissions[:, t].unsqueeze(2)
            trans = self.transitions.unsqueeze(0)
            alpha = torch.logsumexp(alpha.unsqueeze(2) + emit + trans, dim=1)
            alpha *= mask[:, t].unsqueeze(1)

        return torch.logsumexp(alpha + self.end_transitions, dim=1)


class KhmerRNN(nn.Module):
    """
    RNN model for Khmer space injection
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int | list = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        rnn_type: str = "lstm",
        residual: bool = True,
        use_crf: bool = True,
    ):
        """
        Initialize RNN model

        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of character embeddings
            hidden_dim: Dimension of hidden states (int or list per layer)
            num_layers: Number of RNN layers
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional RNN
        """
        super().__init__()

        if isinstance(hidden_dim, int):
            self.hidden_dims = [hidden_dim] * num_layers
        else:
            self.hidden_dims = hidden_dim
            num_layers = len(hidden_dim)

        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.dropout = dropout
        self.residual = residual
        self.use_crf = use_crf
        self.rnn_type = rnn_type.lower()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0
        )

        self.rnn_layers = nn.ModuleList()

        for i in range(num_layers):
            input_dim = (
                embedding_dim
                if i == 0
                else self.hidden_dims[i - 1] * self.num_directions
            )

            rnn_cls = {
                "rnn": nn.RNN,
                "gru": nn.GRU,
                "lstm": nn.LSTM
            }[self.rnn_type]

            self.rnn_layers.append(
                rnn_cls(
                    input_size=input_dim,
                    hidden_size=self.hidden_dims[i],
                    batch_first=True,
                    bidirectional=bidirectional
                )
            )

        self.dropout_layer = nn.Dropout(dropout)

        final_dim = self.hidden_dims[-1] * self.num_directions
        self.fc = nn.Linear(final_dim, 2)

        if self.use_crf:
            self.crf = CRF(num_tags=2)

    def forward(self, x, tags=None, mask=None):
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            tags: Gold labels (optional, required for CRF training)
            mask: Padding mask (optional)

        Returns:
            Emission scores or CRF loss
        """
        out = self.embedding(x)

        for rnn in self.rnn_layers:
            residual = out
            out, _ = rnn(out)

            if self.residual and residual.shape == out.shape:
                out = out + residual

            out = self.dropout_layer(out)

        emissions = self.fc(out)

        if self.use_crf and tags is not None:
            return self.crf(emissions, tags, mask)

        return emissions
