import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import numpy as np
import torch.nn.functional as F
import random
from torch import nn
from .utils import normalize, to_dense, to_sparse,sparse_to_tensor
import time
import pickle
from torch_geometric.utils import softmax
from torch_sparse import spmm


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, norm='', bias=True, sparse=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.linear = nn.Linear(in_features, out_features, bias)
        self.norm = norm

    def forward(self, input, adj=1.0):
        input = to_dense(input)
        support = self.linear(input)
        if isinstance(adj, (float, int)):
            output = support * adj
        else:
            adj = normalize(adj, True) if self.norm == 'symmetric' else normalize(adj,
                                                                                False) if self.norm == 'asymmetric' else adj
            output = torch.spmm(adj, support)
        return output

    def __repr__(self):
        return self.__class__.__name__ + '(in_features={}, out_features={}, bias={}, norm={})'.format(
            self.in_features, self.out_features, self.bias, self.norm)

class CosineGraphAttentionLayer(nn.Module):

    def __init__(self,  requires_grad=True):
        super(CosineGraphAttentionLayer, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if requires_grad:
            # unifrom initialization
            self.beta = Parameter(torch.Tensor(1).uniform_(0, 1), requires_grad=requires_grad)
        else:
            self.beta = torch.autograd.Variable(torch.zeros(1), requires_grad=requires_grad).to(device)
        self.epoch_count = 0
        self.timestamp  = time.strftime('%Y%m%d_%H%M%S', time.localtime())


    def forward(self, xi, xj, adj=1.0):
        xi_norm2 = torch.norm(xi, 2, 1).view(-1, 1)
        xj_norm2 = torch.norm(xj, 2, 1).view(-1, 1).t()

        # add a minor constant (1e-7) to denominator to prevent division by
        # zero error
        cos = self.beta * torch.div(torch.mm(xi,xj.t()), torch.mm(xi_norm2, xj_norm2) + 1e-7)

        # neighborhood masking (inspired by this repo:
        # https://github.com/danielegrattarola/keras-gat)
        if isinstance(adj, (float, int)):
            adj = torch.eye(xi.shape[0]).cuda()
        else:
            adj = to_dense(adj)
        mask = (1. - to_dense(adj)) * -1e9
        masked = cos + mask
        # masked = to_sparse(masked)
        # propagation matrix
        # sparsemax = Sparsemax(dim=1)
        # P = sparsemax(masked)
        P = F.softmax(masked, dim=1)
        # attention-guided propagation

        self.epoch_count += 1

        output = torch.mm(P, xj)
        return output

    def save_attention(self, P):
        if self.epoch_count % 25 == 0:
            print('Saving Attention for {}{}'.format(P.shape[0],P.shape[1]))
            save_file = 'tmp/GCNRx_attention_weight_{}_{}_{}'.format(self.timestamp,P.shape[0],P.shape[1]) + '.pkl'
            with open(save_file, 'wb') as f:
                pickle.dump([P.cpu().detach().numpy(), self.beta.cpu().detach().numpy()], f)

    def __repr__(self):
        return self.__class__.__name__ + ' ()'




class DictReLU(nn.ReLU):
    def forward(self , input):
        return {key: F.relu(fea) for key, fea in input.items()} if isinstance(input, dict) else F.relu(input)


class DictDropout(nn.Dropout):
    def forward(self, input):
        if isinstance(input, dict):
            return {key: F.dropout(fea, self.p, self.training, self.inplace) for key, fea in input.items()}
        else:
            return F.dropout(input, self.p, self.training, self.inplace)