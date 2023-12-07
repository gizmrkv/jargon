import math

import torch
from torch import Tensor, nn


class ScaledDotProductAttention(nn.Module):
    def __init__(
        self, embed_dim: int, dropout: float = 0.0, weight: bool = False
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.weight = weight

        self.dropout_layer = nn.Dropout(dropout)
        self._inv_sqrt_embed_dim = 1.0 / math.sqrt(embed_dim)

        if weight:
            self.w = nn.Parameter(torch.randn(embed_dim, embed_dim))

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor | None = None
    ) -> Tensor:
        if self.weight:
            query = torch.matmul(query, self.w)
        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores * self._inv_sqrt_embed_dim
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))
        scores = torch.softmax(scores, dim=-1)
        scores = self.dropout_layer(scores)
        return torch.matmul(scores, value), scores
