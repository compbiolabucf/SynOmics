import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import pickle
from utils import *
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import torch.nn.init as init
from torch.nn import BatchNorm1d

from layers import GCNCustomLayer, MoGCNLayer


class MoGCN(nn.Module):
    def __init__(self, input_features_u, input_features_v, input_features_w, 
                 num_layers=2, hidden_dim=64, bias=False, k1=0.3, k2=0.3):
        super(MoGCN, self).__init__()

        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            self.layers.append(MoGCNLayer(input_features_u, input_features_v, input_features_w, bias, k1, k2))

        self.fc = nn.Linear(input_features_u + input_features_v + input_features_w, hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, 1)     # for binary classification

    def forward(self, x_u, x_v, x_w, 
                adjacency_matrix_u, adjacency_matrix_v, adjacency_matrix_w, 
                adjacency_matrix_uv, adjacency_matrix_vw, adjacency_matrix_wu):
        
        for layer in self.layers:
            x_u, x_v, x_w = layer(x_u, x_v, x_w, 
                                  adjacency_matrix_u, adjacency_matrix_v, adjacency_matrix_w, 
                                  adjacency_matrix_uv, adjacency_matrix_vw, adjacency_matrix_wu)

        x = torch.cat([x_u, x_v], dim=0)
        x = torch.cat([x, x_w], dim=0)
        x = x.t()
        x = self.fc(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)   

        return x