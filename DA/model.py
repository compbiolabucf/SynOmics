import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *

from layers import MoGCNLayer

class MoGCN(nn.Module):
    def __init__(self, input_features_u, input_features_v, num_layers=2, hidden_dim=64, bias=False, nc=1):
        super(MoGCN, self).__init__()

        self.layers = nn.ModuleList()
        self.nc = nc

        for i in range(num_layers):
            self.layers.append(MoGCNLayer(input_features_u, input_features_v, bias))

        self.fc = nn.Linear(input_features_u + input_features_v, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, nc)

    def forward(self, x_u, x_v, adjacency_matrix_u, adjacency_matrix_v, adjacency_matrix_uv):
        for layer in self.layers:
            x_u, x_v, _, _ = layer(x_u, x_v, adjacency_matrix_u, adjacency_matrix_v, adjacency_matrix_uv)

        x = torch.cat([x_u, x_v], dim=0)
        x = x.t()
        x = self.fc(x)
        x = F.relu(x)
        x = self.fc2(x)
        
        if nc > 2:
            softmax = nn.Softmax(dim=0)
            x = softmax(x) 
        else:
            x = torch.sigmoid(x)              # for binary classification

        return x, x_u, x_v