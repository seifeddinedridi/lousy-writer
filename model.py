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
    def __init__(self, in_features, hidden_size, out_features):
        super().__init__()
        self.w_x_h = nn.Linear(in_features, hidden_size, bias=False)
        self.w_h_h = nn.Linear(hidden_size, hidden_size)
        self.w_h_x = nn.Linear(hidden_size, out_features)

    def forward(self, x, h):
        ht = torch.tanh(self.w_x_h(x) + self.w_h_h(h))
        return torch.tanh(self.w_h_x(ht)), ht


class RNN(nn.Module):
    def __init__(self, in_features, hidden_size):
        super(RNN, self).__init__()
        self.cell = RNNCell(in_features, hidden_size, in_features)
        self.hidden = nn.Parameter(torch.zeros((1, hidden_size)), requires_grad=False)
        self.hidden_size = hidden_size

    def forward(self, x, h=None):
        # (batch_size, sequence_length, embedding_dim)
        b, sequence_length, _ = x.size()
        hidden_tensors = []
        output_tensors = []
        if h is None:
            ht = self.hidden.expand((b, -1))
        else:
            ht = h
        for idx in range(sequence_length):
            output, ht = self.cell(x[:, idx, :], ht)
            hidden_tensors.append(ht)
            output_tensors.append(output)
        # (batch_size, sequence_length, out_features)
        return torch.stack(output_tensors, dim=1), ht

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        return weight.new_zeros(batch_size, self.hidden_size)


class LSTM(nn.Module):
    def __init__(self, in_features, hidden_size):
        super(LSTM, self).__init__()
        self.w_x_f = nn.Linear(in_features, hidden_size, bias=False)
        self.w_h_f = nn.Linear(hidden_size, hidden_size)
        self.w_x_i = nn.Linear(in_features, hidden_size, bias=False)
        self.w_h_i = nn.Linear(hidden_size, hidden_size)
        self.w_x_o = nn.Linear(in_features, hidden_size, bias=False)
        self.w_h_o = nn.Linear(hidden_size, hidden_size)
        self.w_x_c = nn.Linear(in_features, hidden_size, bias=False)
        self.w_h_c = nn.Linear(hidden_size, hidden_size)
        self.c = nn.Parameter(torch.zeros((1, hidden_size)), requires_grad=False)
        self.hidden = nn.Parameter(torch.zeros((1, hidden_size)), requires_grad=False)
        self.hidden_size = hidden_size

    def init_hidden(self, init_hidden):
        weight = next(self.parameters())
        return weight.new_zeros(init_hidden, self.hidden_size), weight.new_zeros(init_hidden, self.hidden_size)

    def forward(self, x, h=None):
        # x has shape(batch_size, sequence_length)
        b, sequence_length, _ = x.size()
        if h is None:
            ct = self.c.expand((b, -1))
            ht = self.hidden.expand((b, -1))
        else:
            ht = h[0]
            ct = h[1]
        output_tensors = []
        for idx in range(sequence_length):
            xt = x[:, idx, :]
            ft = torch.sigmoid(self.w_x_f(xt) + self.w_h_f(ht))
            it = torch.sigmoid(self.w_x_i(xt) + self.w_h_i(ht))
            ot = torch.sigmoid(self.w_x_o(xt) + self.w_h_o(ht))
            ct_hat = torch.sigmoid(self.w_x_c(xt) + self.w_h_c(ht))
            ct = torch.mul(ft, ct) + torch.mul(it, ct_hat)
            ht = torch.mul(ot, torch.tanh(ct))
            output_tensors.append(ot)
        # (batch_size, sequence_length, out_features)
        return torch.stack(output_tensors, dim=1), (ht, ct)


class PyTorchRNN(nn.Module):
    def __init__(self, config: ModelConfig):
        super(PyTorchRNN, self).__init__()
        self.embedding = nn.Embedding(config.in_features, config.embedding_dim)
        if config.cell_type == 'rnn':
            self.rnn = nn.RNN(config.embedding_dim, config.hidden_size, num_layers=config.layers_count,
                              batch_first=True)
        elif config.cell_type == 'lstm':
            self.rnn = nn.LSTM(config.embedding_dim, config.hidden_size, num_layers=config.layers_count,
                               batch_first=True)
        elif config.cell_type == 'gru':
            self.rnn = nn.GRU(config.embedding_dim, config.hidden_size, num_layers=config.layers_count,
                              batch_first=True)
        else:
            raise TypeError('Unsupported cell type')
        self.head = nn.Linear(config.hidden_size, config.out_features)
        self.init_weights()

    def forward(self, x, h=None):
        # x has shape(batch_size, sequence_length)
        x_embedded = self.embedding(x)  # (batch_size, sequence_length, embedding_dim)
        output, hn = self.rnn(x_embedded, h)
        return self.head(output), hn

    def init_hidden(self, batch_size):
        return None

    def init_weights(self):
        init_range = 0.1
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.head.bias.data.fill_(0)
        self.head.weight.data.uniform_(-init_range, init_range)


class MultiLayerRNN(nn.Module):
    def __init__(self, config: ModelConfig):
        super(MultiLayerRNN, self).__init__()
        if config.cell_type == 'rnn':
            self.rnns = nn.ModuleList(
                [RNN(config.embedding_dim, config.hidden_size) for _ in range(config.layers_count)])
            self.head = nn.Linear(config.embedding_dim, config.out_features)
        elif config.cell_type == 'lstm':
            self.rnns = nn.ModuleList(
                [LSTM(config.embedding_dim if idx == 0 else config.hidden_size, config.hidden_size) for idx in
                 range(config.layers_count)])
            self.head = nn.Linear(config.hidden_size, config.out_features)
        else:
            raise TypeError('Unsupported cell type')
        self.embedding = nn.Embedding(config.in_features, config.embedding_dim)
        self.hidden_size = config.hidden_size
        self.layers_count = config.layers_count

    def init_weights(self):
        init_range = 0.1
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.head.bias.data.fill_(0)
        self.head.weight.data.uniform_(-init_range, init_range)

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        return weight.new_zeros(self.layers_count, batch_size, self.hidden_size),\
            weight.new_zeros(self.layers_count, batch_size, self.hidden_size)

    def forward(self, x, h=None):
        # x has shape(batch_size, sequence_length)
        in_tensor = self.embedding(x)
        output_hidden = []
        output_c = []
        for idx, rnn in enumerate(self.rnns):
            in_tensor, h = rnn(in_tensor, (h[0][idx], h[1][idx]))
            if isinstance(rnn, LSTM):
                output_hidden.append(h[0])
                output_c.append(h[1])
            else:
                output_hidden.append(h)
        in_tensor = self.head(in_tensor)
        if isinstance(self.rnns[0], LSTM):
            state = (torch.stack(output_hidden, dim=0), torch.stack(output_c, dim=0))
        else:
            state = torch.stack(output_hidden, dim=0)
        return in_tensor, state
