from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ModelConfig:
    in_features: int
    out_features: int
    embedding_dim: int = 256
    hidden_size: int = 1024
    layers_count: int = 1
    cell_type: str = 'rnn'


class RNNCell(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.w_x_h = nn.Linear(in_features, out_features)
        self.w_h_h = nn.Linear(out_features, out_features)

    def forward(self, x, h):
        return torch.tanh(self.w_x_h(x) + self.w_h_h(h))


class RNN(nn.Module):
    def __init__(self, in_features, hidden_size):
        super(RNN, self).__init__()
        self.cell = RNNCell(in_features, hidden_size)
        self.hidden = nn.Parameter(torch.zeros((1, hidden_size)))

    def forward(self, x):
        # (batch_size, sequence_length, embedding_dim)
        b, sequence_length, _ = x.size()
        hidden_tensors = []
        ht = self.hidden.expand((b, -1))
        for idx in range(sequence_length):
            ht = self.cell(x[:, idx, :], ht)
            hidden_tensors.append(ht)
        # (batch_size, sequence_length, out_features)
        hidden_tensors = torch.stack(hidden_tensors, dim=1)
        return hidden_tensors, hidden_tensors[:, -1]


class LSTM(nn.Module):
    def __init__(self, in_features, hidden_size):
        super(LSTM, self).__init__()
        self.w_x_f = nn.Linear(in_features, hidden_size, bias=False)
        self.w_c_f = nn.Linear(hidden_size, hidden_size)
        self.w_x_i = nn.Linear(in_features, hidden_size, bias=False)
        self.w_c_i = nn.Linear(hidden_size, hidden_size)
        self.w_x_o = nn.Linear(in_features, hidden_size, bias=False)
        self.w_c_o = nn.Linear(hidden_size, hidden_size)
        self.w_x_c = nn.Linear(in_features, hidden_size)
        self.c = nn.Parameter(torch.zeros((1, hidden_size)))

    def forward(self, x):
        # x has shape(batch_size, sequence_length)
        b, sequence_length, _ = x.size()
        ct = self.c.expand((b, -1))
        hidden_tensors = []
        for idx in range(sequence_length):
            xt = x[:, idx, :]
            ft = torch.sigmoid(self.w_x_f(xt) + self.w_c_f(ct))
            it = torch.sigmoid(self.w_x_i(xt) + self.w_c_i(ct))
            ot = torch.sigmoid(self.w_x_o(xt) + self.w_c_o(ct))
            ct = torch.mul(ft, ct) + torch.mul(it, torch.tanh(self.w_x_c(xt)))
            ht = torch.mul(ot, torch.tanh(ct))
            hidden_tensors.append(ht)
        # (batch_size, sequence_length, out_features)
        hidden_tensors = torch.stack(hidden_tensors, dim=1)
        return hidden_tensors, hidden_tensors[:, -1]  # (batch_size, sequence_length, out_features)


class PyTorchRNN(nn.Module):
    def __init__(self, config: ModelConfig):
        super(PyTorchRNN, self).__init__()
        self.embedding = nn.Embedding(config.in_features, config.embedding_dim)
        if config.cell_type == 'rnn':
            self.rnn = nn.RNN(config.embedding_dim, config.hidden_size, num_layers=config.layers_count, batch_first=True)
        elif config.cell_type == 'lstm':
            self.rnn = nn.LSTM(config.embedding_dim, config.hidden_size, num_layers=config.layers_count, batch_first=True)
        elif config.cell_type == 'gru':
            self.rnn = nn.GRU(config.embedding_dim, config.hidden_size, num_layers=config.layers_count, batch_first=True)
        else:
            raise TypeError('Unsupported cell type')
        self.head = nn.Linear(config.hidden_size, config.out_features)

    def forward(self, x):
        # x has shape(batch_size, sequence_length)
        x_embedded = self.embedding(x)  # (batch_size, sequence_length, embedding_dim)
        output, hn = self.rnn(x_embedded)
        return self.head(output), hn


class MultiLayerRNN(nn.Module):
    def __init__(self, config: ModelConfig):
        super(MultiLayerRNN, self).__init__()
        if config.cell_type == 'rnn':
            self.rnns = nn.ModuleList(
                [RNN(config.embedding_dim if idx == 0 else config.hidden_size, config.hidden_size) for idx in
                 range(config.layers_count)])
        elif config.cell_type == 'lstm':
            self.rnns = nn.ModuleList(
                [LSTM(config.embedding_dim if idx == 0 else config.hidden_size, config.hidden_size) for idx in
                 range(config.layers_count)])
        else:
            raise TypeError('Unsupported cell type')
        self.embedding = nn.Embedding(config.in_features, config.embedding_dim)
        self.head = nn.Linear(config.hidden_size, config.out_features)

    def forward(self, x):
        # x has shape(batch_size, sequence_length)
        in_tensor = self.embedding(x)
        for rnn in self.rnns:
            in_tensor, _ = rnn(in_tensor)
        in_tensor = self.head(in_tensor)
        return in_tensor, in_tensor[:, -1]
