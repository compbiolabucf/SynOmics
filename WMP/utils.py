import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn as nn
import torch.nn.functional as F


def get_scaler(scaler_name):
    if scaler_name == 'standard':
        return StandardScaler()
    elif scaler_name == 'minmax':
        return MinMaxScaler()
    elif scaler_name == 'robust':
        return RobustScaler()
    elif scaler_name == 'normalize':
        return Normalizer()
    else:
        raise ValueError("Invalid scaler name")
    

def normalize(data, scaler_name='standard'):
    scaler = get_scaler(scaler_name)
    
    data_np = data.detach().cpu().numpy()
    data_np = data_np.T 
    X_np = scaler.fit_transform(data_np)
    X_np = X_np.T 
    X = torch.tensor(X_np, dtype=torch.float32)

    if data.is_cuda:
        X = X.to(data.device)

    return X


def get_adjacency_matrix(data, threshold=0.5, metric='cosine'):
    """
    Create an adjacency matrix from the data
    :param data: numpy array
    :param threshold: float
    :return: numpy array
    """
    # Create the adjacency matrix
    if metric == 'cosine':
        A = cosine_similarity(data)
    elif metric == 'correlation':
        A = np.corrcoef(data)
    
    # Apply the threshold
    A[A < threshold] = 0
    A[A > threshold] = 1
    
    return A


def get_adjacency_matrix_bipartite(A_u, A_v):
    """
    Create a bipartite adjacency matrix from the data
    :param A_u: numpy array
    :param A_v: numpy array
    :return: numpy array
    """
    A = np.zeros((A_u.shape[0] + A_v.shape[0], A_u.shape[1] + A_v.shape[1]))

    u, _ = A_u.shape
    v, _ = A_v.shape

    A[:u, :u] = A_u
    A[u:, u:] = A_v

    return A


def normalized_adjacency(adjacency_matrix):
        # Add self-loops to the adjacency matrix
        adjacency_matrix = adjacency_matrix + torch.eye(adjacency_matrix.size(0), device=adjacency_matrix.device)

        # Compute degree matrix
        degree_matrix = torch.diag(torch.sum(adjacency_matrix, dim=1))
        
        # Compute D^(-1/2)
        degree_matrix_inv_sqrt = torch.pow(degree_matrix, -0.5)
        degree_matrix_inv_sqrt[torch.isinf(degree_matrix_inv_sqrt)] = 0. # handle division by zero

        # Normalize adjacency matrix
        norm_adj = torch.matmul(torch.matmul(degree_matrix_inv_sqrt, adjacency_matrix), degree_matrix_inv_sqrt)

        return norm_adj


def normalized_adjacency_bipartite(adjacency_matrix):
    n = adjacency_matrix.size(0)
    m = adjacency_matrix.size(1)
    adjacency_matrix_bipartite = torch.zeros(n+m, n+m, device=adjacency_matrix.device)
    adjacency_matrix_bipartite[:n, n:] = adjacency_matrix
    adjacency_matrix_bipartite[n:, :n] = adjacency_matrix.t()

    B = normalized_adjacency(adjacency_matrix_bipartite)
    B_u = B[:n, n:]         # n x m
    B_v = B[n:, :n]         # m x n

    return B_u, B_v


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Apply sigmoid to the inputs to get probabilities
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        targets = targets.float()  # Ensure targets are float type
        
        # Compute Focal Loss components
        pt = torch.exp(-BCE_loss)  # Probability of true class
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        
        # Apply reduction method
        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss


def get_best_threshold(fpr, tpr, thresholds, method='Youden'):
    """
    fpr: array
    tpr: array
    thresholds: array
    method: str
    """
    if method == 'Youden':
        # Youden's J statistic
        J = tpr - fpr
        idx = np.argmax(J)
    elif method =='distance':
        # Distance to the top-left corner
        dist = np.sqrt(fpr**2 + (1-tpr)**2)
        idx = np.argmin(dist)
    elif method == 'optimal':
        # Optimal threshold
        idx = np.argmax(tpr - fpr)
    else:
        raise ValueError("Invalid method")
    
    return thresholds[idx]