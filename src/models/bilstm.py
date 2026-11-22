"""Bidirectional LSTM text classifier over GloVe embeddings.

The BiLSTM reads the review left-to-right and right-to-left; we pool its outputs with masked
mean + max pooling (padding positions are ignored) and classify with a linear layer.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiLSTMClassifier(nn.Module):
    def __init__(self, emb_matrix, hidden: int = 128, num_layers: int = 1,
                 dropout: float = 0.3, freeze_emb: bool = False):
        super().__init__()
        emb = torch.as_tensor(emb_matrix, dtype=torch.float32)
        self.emb = nn.Embedding.from_pretrained(emb, freeze=freeze_emb, padding_idx=0)
        self.lstm = nn.LSTM(emb.shape[1], hidden, num_layers=num_layers, batch_first=True,
                            bidirectional=True, dropout=dropout if num_layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(4 * hidden, 1)   # mean-pool (2h) + max-pool (2h)

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = self.emb(ids)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=ids.shape[1])  # [B, L, 2h]
        mask = (torch.arange(ids.shape[1], device=ids.device)[None, :] < lengths[:, None]).unsqueeze(-1)
        mean = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        mx = out.masked_fill(~mask, -1e9).max(1).values
        return self.fc(self.drop(torch.cat([mean, mx], dim=1))).squeeze(1)
