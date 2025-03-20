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
        X = torch.matmul(A, X)          # p x n
        X = torch.matmul(self.W, X)     # p x n

        if self.has_bias:
            X = X.t() + self.bias
            X = X.t()    
        
        return X                        # p x n
    

class MoGCNLayer(nn.Module):
    def __init__(self, in_features_u, in_features_v, in_features_w, bias=False, k1=0.3, k2=0.3):
        super(MoGCNLayer, self).__init__()
        
        self.in_features_u = in_features_u
        self.in_features_v = in_features_v
        self.in_features_w = in_features_w
        self.k1 = k1
        self.k2 = k2

        self.GCN_u = GCNCustomLayer(in_features_u, bias)
        self.GCN_v = GCNCustomLayer(in_features_v, bias)
        self.GCN_w = GCNCustomLayer(in_features_w, bias)

        self.W_u = nn.Parameter(torch.randn(in_features_u, in_features_u))        
        self.W_v = nn.Parameter(torch.randn(in_features_v, in_features_v))
        self.W_w = nn.Parameter(torch.randn(in_features_w, in_features_w))
        
        init.xavier_uniform_(self.W_u)      # p x p
        init.xavier_uniform_(self.W_v)      # q x q
        init.xavier_uniform_(self.W_w)      # r x r

        self.bn_u = BatchNorm1d(in_features_u)
        self.bn_v = BatchNorm1d(in_features_v)
        self.bn_w = BatchNorm1d(in_features_w)

    def forward(self, X_u, X_v, X_w, adjacency_matrix_u, adjacency_matrix_v, adjacency_matrix_w,
                 adjacency_matrix_uv, adjacency_matrix_vw, adjacency_matrix_wu):
        """
            X_u : p x n -> p is features, n is samples
            X_v : q x n -> q is features, n is samples
            X_w : r x n -> r is features, n is samples
        """
        H_u = self.GCN_u(X_u, adjacency_matrix_u)       # p x n
        H_v = self.GCN_v(X_v, adjacency_matrix_v)       # q x n
        H_w = self.GCN_w(X_w, adjacency_matrix_w)       # r x n

        W_u = self.W_u
        W_v = self.W_v
        W_w = self.W_w

        # Message passing between u and v
        H_vu = torch.matmul(adjacency_matrix_uv, X_v)       
        H_vu = torch.matmul(W_u, H_vu)                      # p x n                  

        H_uv = torch.matmul(adjacency_matrix_uv.t(), X_u)   
        H_uv = torch.matmul(W_v, H_uv)                      # q x n

        # Message passing between v and w
        H_wv = torch.matmul(adjacency_matrix_vw, X_w)       
        H_wv = torch.matmul(W_v, H_wv)                      # q x n

        H_vw = torch.matmul(adjacency_matrix_vw.t(), X_v)   
        H_vw = torch.matmul(W_w, H_vw)                      # r x n

        # Message passing between w and u
        H_uw = torch.matmul(adjacency_matrix_wu, X_u)       # U -> W
        H_uw = torch.matmul(W_w, H_uw)                      # r x n

        H_wu = torch.matmul(adjacency_matrix_wu.t(), X_w)   # W -> U
        H_wu = torch.matmul(W_u, H_wu)                      # p x n       

        X_u = self.k1 * H_vu + self.k2 * H_wu + (1 - self.k1 - self.k2) * H_u
        X_v = self.k1 * H_uv + self.k2 * H_wv + (1 - self.k1 - self.k2) * H_v
        X_w = self.k1 * H_vw + self.k2 * H_uw + (1 - self.k1 - self.k2) * H_w

        X_u = X_u.t()
        X_u = self.bn_u(X_u)
        X_u = X_u.t()

        X_v = X_v.t()
        X_v = self.bn_v(X_v)
        X_v = X_v.t()

        X_w = X_w.t()
        X_w = self.bn_w(X_w)
        X_w = X_w.t()

        X_u = F.leaky_relu(X_u)
        X_v = F.leaky_relu(X_v)
        X_w = F.leaky_relu(X_w)
    
        X_u = F.dropout(X_u, p=0.5, training=self.training)
        X_v = F.dropout(X_v, p=0.5, training=self.training)
        X_w = F.dropout(X_w, p=0.5, training=self.training)
        
        return X_u, X_v, X_w