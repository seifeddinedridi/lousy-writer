from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ModelConfig:
    in_features: int
    out_features: int
    embedding_dim: int = 128


class RNNCell(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # TODO: matrix initialization
        self.w_xh_h = nn.Linear(in_features, out_features)

    def forward(self, x, h):
        xh = torch.cat((x, h), dim=1)
        return torch.tanh(self.w_xh_h(xh))


class RNN(nn.Module):
    def __init__(self, config: ModelConfig):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(config.in_features, config.embedding_dim)
        self.cell = RNNCell(config.embedding_dim + config.embedding_dim, config.embedding_dim)
        self.head = nn.Linear(config.embedding_dim, config.out_features)
        self.hidden = nn.Parameter(torch.zeros((1, config.embedding_dim)))

    def forward(self, x):
        # x has shape(batch_size, sequence_length)
        x_embedded = self.embedding(x)
        # x_embedded has shape(batch_size, sequence_length, embedding_dim)
        b, t, _ = x_embedded.size()
        ht = self.hidden.expand((b, -1))
        hidden_tensors = []
        for idx in range(t):
            ht = self.cell(x_embedded[:, idx, :], ht)
            hidden_tensors.append(ht)
        hidden_tensors = torch.stack(hidden_tensors, dim=1)
        # hidden_tensors has shape(batch_size, sequence_length, embedding_dim)
        # final output has shape(batch_size, sequence_length, out_features)
        return self.head(hidden_tensors)
