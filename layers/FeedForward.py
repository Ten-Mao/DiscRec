from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(self, d_model, inner_dim, dropout=0.1, activation: Literal['relu', 'gelu'] = 'relu'):
        super(FeedForward, self).__init__()
        self.conv1 = nn.Conv1d(d_model, inner_dim, 1)
        self.dropout1 = nn.Dropout(dropout)
        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'gelu':
            self.act = nn.GELU()
        else:
            raise ValueError('activation should be either "relu" or "gelu"')
        self.conv2 = nn.Conv1d(inner_dim, d_model, 1)
        self.dropout2 = nn.Dropout(dropout)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv1d):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        residual = x
        x = x.permute(0, 2, 1)
        x = self.dropout1(self.act(self.conv1(x)))
        x = self.dropout2(self.conv2(x))
        x = x.permute(0, 2, 1)
        return x + residual
