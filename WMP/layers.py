import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
import torch.nn.init as init
from torch.nn import BatchNorm1d


class GCNCustomLayer(nn.Module):
    def __init__(self, in_features, bias=False):
        super(GCNCustomLayer, self).__init__()
        self.in_features = in_features  
        self.has_bias = bias
        self.W = nn.Parameter(torch.randn(in_features, in_features))        
        init.xavier_uniform_(self.W)

        if bias:
            self.bias = nn.Parameter(torch.randn(in_features))
            init.zeros_(self.bias)
    
    def forward(self, X, adjacency_matrix):
        
        A = adjacency_matrix
        X = torch.matmul(A, X)          # A(nxn), X(nxd) = X(nxd)
        X = torch.matmul(self.W, X)     # W(nxn), X(nxd) = X(nxd)

        if self.has_bias:
            X = X.t() + self.bias
            X = X.t()    
        
        return X
    

class SynOmicsLayer(nn.Module):
    def __init__(self, in_features_u, in_features_v, bias=False, k=0.5):
        super(SynOmicsLayer, self).__init__()
        
        self.in_features_u = in_features_u
        self.in_features_v = in_features_v
        self.k = k

        self.GCN_u = GCNCustomLayer(in_features_u, bias)
        self.GCN_v = GCNCustomLayer(in_features_v, bias)

        self.W_u = nn.Parameter(torch.randn(in_features_u, in_features_u))        
        self.W_v = nn.Parameter(torch.randn(in_features_v, in_features_v))

        init.xavier_uniform_(self.W_u)      # p x p
        init.xavier_uniform_(self.W_v)      # q x q

        self.bn_u = BatchNorm1d(in_features_u)
        self.bn_v = BatchNorm1d(in_features_v)

    def forward(self, X_u, X_v, adjacency_matrix_u, adjacency_matrix_v, adjacency_matrix_uv):
        H_u = self.GCN_u(X_u, adjacency_matrix_u)
        H_v = self.GCN_v(X_v, adjacency_matrix_v)

        W_u = self.W_u
        W_v = self.W_v

        H_vu = torch.matmul(adjacency_matrix_uv, X_v)       # V -> U
        H_vu = torch.matmul(W_u, H_vu)                      # nxd

        H_uv = torch.matmul(adjacency_matrix_uv.t(), X_u)   # U -> V
        H_uv = torch.matmul(W_v, H_uv)                      # mxd

        X_u = self.k * H_u + (1 - self.k) * H_vu
        X_v = self.k * H_v + (1 - self.k) * H_uv
        
        X_u = X_u.t()
        X_u = self.bn_u(X_u)
        X_u = X_u.t()

        X_v = X_v.t()
        X_v = self.bn_v(X_v)
        X_v = X_v.t()

        # X_u = F.relu(X_u)
        # X_v = F.relu(X_v)
        X_u = F.leaky_relu(X_u)
        X_v = F.leaky_relu(X_v)
    
        X_u = F.dropout(X_u, p=0.25, training=self.training)
        X_v = F.dropout(X_v, p=0.25, training=self.training)
        
        return X_u, X_v