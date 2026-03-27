import math
import time
import pickle
from typing import Dict, Union

import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

from .utils import normalize, to_dense, to_sparse, sparse_to_tensor


class GraphConvolution(nn.Module):
    """Single graph convolutional layer: H' = A_norm * X * W.

    Implements the propagation rule from Kipf & Welling (2017):
    https://arxiv.org/abs/1609.02907

    Args:
        in_features: Input feature dimensionality.
        out_features: Output feature dimensionality.
        norm: Adjacency normalization mode — 'symmetric', 'asymmetric', or '' (none).
        bias: Whether to include a bias term.
        sparse: Unused flag retained for API compatibility.
    """

    def __init__(self, in_features: int, out_features: int,
                 norm: str = '', bias: bool = True, sparse: bool = True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.linear = nn.Linear(in_features, out_features, bias)
        self.norm = norm

    def forward(self, input: torch.Tensor,
                adj: Union[torch.Tensor, float, int] = 1.0) -> torch.Tensor:
        """Compute H' = A_norm * (X * W).

        Args:
            input: Node feature matrix of shape (N, in_features).
            adj: Adjacency matrix (N, N) or scalar multiplier.

        Returns:
            Output node embeddings of shape (N, out_features).
        """
        input = to_dense(input)
        support = self.linear(input)
        if isinstance(adj, (float, int)):
            return support * adj
        adj = (normalize(adj, True) if self.norm == 'symmetric'
               else normalize(adj, False) if self.norm == 'asymmetric'
               else adj)
        return torch.spmm(adj, support)

    def __repr__(self) -> str:
        return '{}(in_features={}, out_features={}, bias={}, norm={})'.format(
            self.__class__.__name__, self.in_features, self.out_features,
            self.bias, self.norm)


class CosineGraphAttentionLayer(nn.Module):
    """Graph attention layer using scaled cosine similarity as attention scores.

    Attention coefficient: a_{ij} = beta * (x_i . x_j) / (||x_i|| ||x_j|| + eps)
    Neighborhood masking ensures only adjacent nodes contribute.
    Propagation: h_i' = sum_j softmax(a_{ij}) * x_j

    Args:
        requires_grad: If True, the scale parameter beta is learnable.
    """

    def __init__(self, requires_grad: bool = True):
        super(CosineGraphAttentionLayer, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if requires_grad:
            self.beta = Parameter(torch.Tensor(1).uniform_(0, 1), requires_grad=True)
        else:
            self.beta = torch.autograd.Variable(
                torch.zeros(1), requires_grad=False).to(device)
        self.epoch_count = 0
        self.timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())

    def forward(self, xi: torch.Tensor, xj: torch.Tensor,
                adj: Union[torch.Tensor, float, int] = 1.0) -> torch.Tensor:
        """Compute attention-weighted aggregation from xj to xi's neighborhood.

        Args:
            xi: Query node features of shape (N, d).
            xj: Key/value node features of shape (M, d).
            adj: Adjacency mask (N, M). Scalar triggers self-loop identity mask.

        Returns:
            Aggregated output of shape (N, d).
        """
        xi_norm2 = torch.norm(xi, 2, 1).view(-1, 1)
        xj_norm2 = torch.norm(xj, 2, 1).view(-1, 1).t()
        cos = self.beta * torch.div(
            torch.mm(xi, xj.t()),
            torch.mm(xi_norm2, xj_norm2) + 1e-7)

        if isinstance(adj, (float, int)):
            adj = torch.eye(xi.shape[0], device=xi.device)
        else:
            adj = to_dense(adj)
        mask = (1. - to_dense(adj)) * -1e9
        P = F.softmax(cos + mask, dim=1)

        self.epoch_count += 1
        return torch.mm(P, xj)

    def save_attention(self, P: torch.Tensor) -> None:
        """Periodically serialize the attention matrix P to disk.

        Args:
            P: Attention weight matrix of shape (N, M).
        """
        if self.epoch_count % 25 == 0:
            print('Saving Attention for {}{}'.format(P.shape[0], P.shape[1]))
            save_file = 'tmp/MGATRx_attention_weight_{}_{}_{}.pkl'.format(
                self.timestamp, P.shape[0], P.shape[1])
            with open(save_file, 'wb') as f:
                pickle.dump([P.cpu().detach().numpy(),
                             self.beta.cpu().detach().numpy()], f)

    def __repr__(self) -> str:
        return self.__class__.__name__ + ' ()'


class DictReLU(nn.ReLU):
    """ReLU that accepts either a tensor or a dict of tensors keyed by node type."""

    def forward(self, input: Union[torch.Tensor, Dict]) -> Union[torch.Tensor, Dict]:
        if isinstance(input, dict):
            return {key: F.relu(fea) for key, fea in input.items()}
        return F.relu(input)


class DictDropout(nn.Dropout):
    """Dropout that accepts either a tensor or a dict of tensors keyed by node type."""

    def forward(self, input: Union[torch.Tensor, Dict]) -> Union[torch.Tensor, Dict]:
        if isinstance(input, dict):
            return {key: F.dropout(fea, self.p, self.training, self.inplace)
                    for key, fea in input.items()}
        return F.dropout(input, self.p, self.training, self.inplace)
