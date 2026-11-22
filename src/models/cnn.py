"""1D-CNN text classifier over GloVe embeddings (Kim, 2014 style).

Filters of width 3/4/5 act like learned n-gram detectors; max-pooling over time keeps the
strongest match of each filter anywhere in the review.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, emb_matrix, n_filters: int = 100, kernel_sizes=(3, 4, 5),
                 dropout: float = 0.5, freeze_emb: bool = False):
        super().__init__()
        emb = torch.as_tensor(emb_matrix, dtype=torch.float32)
        self.emb = nn.Embedding.from_pretrained(emb, freeze=freeze_emb, padding_idx=0)
        self.convs = nn.ModuleList(nn.Conv1d(emb.shape[1], n_filters, k) for k in kernel_sizes)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(kernel_sizes), 1)
        self.min_len = max(kernel_sizes)

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if ids.shape[1] < self.min_len:  # very short batch: pad so the widest filter fits
            ids = F.pad(ids, (0, self.min_len - ids.shape[1]))
        x = self.emb(ids).transpose(1, 2)                       # [B, emb_dim, L]
        pooled = [F.relu(conv(x)).max(dim=2).values for conv in self.convs]
        return self.fc(self.drop(torch.cat(pooled, dim=1))).squeeze(1)   # logits [B]
