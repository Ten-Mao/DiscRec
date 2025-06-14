from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPLayers(nn.Module):

    def __init__(
        self,

        dim_list,
        dropout=0.1,
        activation_fn:Literal["relu", "gelu", "sigmoid", "tanh"]="relu",
        bn=False,
        last_activation=True,
    ):
        super(MLPLayers, self).__init__()

        self.dim_list = dim_list
        self.dropout = dropout
        self.activation_fn = activation_fn
        self.bn = bn
        self.last_activation = last_activation


        self.layers = []
        for i in range(len(dim_list) - 1):
            self.layers.append(nn.Dropout(dropout))
            self.layers.append(nn.Linear(dim_list[i], dim_list[i + 1]))
            if bn:
                self.layers.append(nn.BatchNorm1d(dim_list[i + 1]))
            activation_fn = self.get_activation_fn(self.activation_fn)
            if i < len(dim_list) - 2 and activation_fn is not None:
                self.layers.append(activation_fn)

        activation_fn = self.get_activation_fn(self.activation_fn)
        if last_activation and activation_fn is not None:
            self.layers.append(activation_fn)
        
        self.mlp = nn.Sequential(*self.layers)
        self.apply(self.init_weights)
    
    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight.data)
            if module.bias is not None:
                module.bias.data.fill_(0.0)


    def get_activation_fn(self, activation):
        if activation == None:
            return None
        elif activation == "relu":
            return nn.ReLU()
        elif activation == "gelu":
            return nn.GELU()
        elif activation == "sigmoid":
            return nn.Sigmoid()
        elif activation == "tanh":
            return nn.Tanh()
        else:
            raise ValueError(f"activation should be one of [relu, gelu, sigmoid, tanh], but got {activation}")
    

    def forward(self, x):
        return self.mlp(x)