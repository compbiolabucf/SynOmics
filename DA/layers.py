import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
import torch.nn.init as init
from torch.nn import BatchNorm1d

def dense_to_coo(adjacency_matrix):
    rows, cols = adjacency_matrix.nonzero(as_tuple=True)
    values = adjacency_matrix[rows, cols]
    coo_tensor = torch.sparse_coo_tensor(indices=torch.stack([rows, cols]),
                                          values=values,
                                          size=adjacency_matrix.size())
    
    return coo_tensor

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
        A = dense_to_coo(A)
        X = torch.sparse.mm(A, X)       # A(nxn), X(nxd) = X(nxd)
        X = torch.matmul(self.W, X)     # W(nxn), X(nxd) = X(nxd)

        if self.has_bias:
            X = X.t() + self.bias
            X = X.t()    
        
        return X
    

class SynOmicsLayer(nn.Module):
    def __init__(self, in_features_u, in_features_v, bias=False):
        super(SynOmicsLayer, self).__init__()
        
        self.in_features_u = in_features_u
        self.in_features_v = in_features_v

        self.GCN_u = GCNCustomLayer(in_features_u, bias)
        self.GCN_v = GCNCustomLayer(in_features_v, bias)

        self.W_u = nn.Parameter(torch.randn(in_features_u, in_features_u))        
        self.W_v = nn.Parameter(torch.randn(in_features_v, in_features_v))

        init.xavier_uniform_(self.W_u)     
        init.xavier_uniform_(self.W_v)      

        self.bn_u = BatchNorm1d(in_features_u)
        self.bn_v = BatchNorm1d(in_features_v)

    def forward(self, X_u, X_v, adjacency_matrix_u, adjacency_matrix_v, adjacency_matrix_uv):
        H_u = self.GCN_u(X_u, adjacency_matrix_u)
        H_v = self.GCN_v(X_v, adjacency_matrix_v)

        W_u = self.GCN_u.W
        W_v = self.GCN_v.W

        # convert adjacency_matrix_uv to COO format
        adjacency_matrix_uv = dense_to_coo(adjacency_matrix_uv)
        H_vu = torch.sparse.mm(adjacency_matrix_uv, X_v)       # U -> V
        H_vu = torch.matmul(W_u, H_vu)                      # nxd

        H_uv = torch.sparse.mm(adjacency_matrix_uv.t(), X_u)   # V -> U
        H_uv = torch.matmul(W_v, H_uv)

        H_u = H_u.t()
        H_u = self.bn_u(H_u)
        H_u = H_u.t()

        H_v = H_v.t()
        H_v = self.bn_v(H_v)
        H_v = H_v.t()

        H_vu = H_vu.t()
        H_vu = self.bn_u(H_vu)
        H_vu = H_vu.t()

        H_uv = H_uv.t()
        H_uv = self.bn_v(H_uv)
        H_uv = H_uv.t()

        H_u = F.relu(H_u)
        H_v = F.relu(H_v)
        H_vu = F.relu(H_vu)
        H_uv = F.relu(H_uv)

        H_u = F.dropout(H_u, p=0.5, training=self.training)
        H_v = F.dropout(H_v, p=0.5, training=self.training)
        H_vu = F.dropout(H_vu, p=0.5, training=self.training)
        H_uv = F.dropout(H_uv, p=0.5, training=self.training)
        
        return H_u, H_v, H_vu, H_uv