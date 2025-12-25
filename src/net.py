"""
Neural network models for Khmer space injection
"""

import torch
import torch.nn as nn
import random

class CRF(nn.Module):
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

class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.Wxh = nn.Linear(input_dim, hidden_dim)
        self.Whh = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x_t, h_prev):
        return torch.tanh(self.Wxh(x_t) + self.Whh(h_prev))
    
class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.z = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.r = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.h = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(self, x_t, h_prev):
        concat = torch.cat([x_t, h_prev], dim=-1)
        z_t = torch.sigmoid(self.z(concat))
        r_t = torch.sigmoid(self.r(concat))

        concat_reset = torch.cat([x_t, r_t * h_prev], dim=-1)
        h_tilde = torch.tanh(self.h(concat_reset))

        return (1 - z_t) * h_prev + z_t * h_tilde
    
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.i = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.f = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.o = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.g = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(self, x_t, state):
        h_prev, c_prev = state
        concat = torch.cat([x_t, h_prev], dim=-1)

        i_t = torch.sigmoid(self.i(concat))
        f_t = torch.sigmoid(self.f(concat))
        o_t = torch.sigmoid(self.o(concat))
        g_t = torch.tanh(self.g(concat))

        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t

class BiRecurrentLayer(nn.Module):
    def __init__(self, cell_cls, input_dim, hidden_dim, bidirectional=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

        self.fw = cell_cls(input_dim, hidden_dim)
        if bidirectional:
            self.bw = cell_cls(input_dim, hidden_dim)

    def forward(self, x):
        B, T, _ = x.shape
        device = x.device
        H = self.hidden_dim

        # ---------- Forward ----------
        h_fw = []
        h = torch.zeros(B, H, device=device)
        c = torch.zeros_like(h) if isinstance(self.fw, LSTM) else None

        for t in range(T):
            if c is not None:
                h, c = self.fw(x[:, t], (h, c))
            else:
                h = self.fw(x[:, t], h)
            h_fw.append(h)

        h_fw = torch.stack(h_fw, dim=1)

        if not self.bidirectional:
            return h_fw

        # ---------- Backward ----------
        h_bw = []
        h = torch.zeros(B, H, device=device)
        c = torch.zeros_like(h) if isinstance(self.bw, LSTM) else None

        for t in reversed(range(T)):
            if c is not None:
                h, c = self.bw(x[:, t], (h, c))
            else:
                h = self.bw(x[:, t], h)
            h_bw.append(h)

        h_bw.reverse()
        h_bw = torch.stack(h_bw, dim=1)

        return torch.cat([h_fw, h_bw], dim=-1)
    
class KhmerRNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        hidden_dim=256,
        num_layers=2,
        dropout=0.3,
        bidirectional=True,
        rnn_type="lstm",
        residual=True,
        use_crf=True,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.residual = residual
        self.use_crf = use_crf

        cell_map = {
            "rnn": RNN,
            "gru": GRU,
            "lstm": LSTM,
        }
        cell_cls = cell_map[rnn_type.lower()]

        self.layers = nn.ModuleList()
        input_dim = embedding_dim

        for _ in range(num_layers):
            layer = BiRecurrentLayer(
                cell_cls=cell_cls,
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                bidirectional=bidirectional,
            )
            self.layers.append(layer)
            input_dim = hidden_dim * (2 if bidirectional else 1)

        self.fc = nn.Linear(input_dim, 2)

        if use_crf:
            self.crf = CRF(num_tags=2)

    def forward(self, x, tags=None, mask=None):
        out = self.embedding(x)

        for layer in self.layers:
            residual = out
            out = layer(out)

            if self.residual and out.shape == residual.shape:
                out = out + residual

            out = self.dropout(out)

        emissions = self.fc(out)

        if self.use_crf and tags is not None:
            return self.crf(emissions, tags, mask)

        return emissions
