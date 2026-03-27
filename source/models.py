from typing import Callable, Dict, Tuple

import torch
import torch.nn as nn

from .layers import GraphConvolution, CosineGraphAttentionLayer, DictReLU, DictDropout
from .utils import issymmetric, to_sparse, to_dense, normalize


class MGATRx(nn.Module):
    """Multi-view Graph Attention network for drug Repositioning (MGATRx).

    Encodes a heterogeneous graph via stacked HeteroGCN or HeteroGAT layers,
    then decodes per-task adjacency reconstructions via linear projections.

    Args:
        in_dim: Dict mapping node type id -> input feature dimensionality.
        out_dims: Tuple of hidden/output dimensions, one per encoder layer.
        tasks: Iterable of (src_type, dst_type) edge types to reconstruct.
        enc_act: Activation applied after each encoder layer.
        dropout: Dropout probability applied in the predictor head (GCN only).
        model: Encoder backbone — 'GCN' or 'GAT'.
    """

    def __init__(self, in_dim: Dict[int, int], out_dims: Tuple[int, ...],
                 tasks=((0, 3),), enc_act: Callable = lambda x: x,
                 dropout: float = 0.5, model: str = 'GCN'):
        super(MGATRx, self).__init__()
        self.model = model
        self.heteroGNN = HeteroGCN if model == 'GCN' else HeteroGAT
        self.encoder = nn.ModuleList(
            [self.heteroGNN(in_dim, out_dims[0], act=enc_act)])
        self.Dropout = DictDropout(p=dropout)
        self.tasks = tasks
        self.predicter = nn.ModuleDict(
            {str(i): nn.Linear(out_dims[-1], in_dim[i[1]]) for i in tasks})
        for i in range(len(out_dims)):
            if i + 1 < len(out_dims):
                self.encoder.append(self.heteroGNN(out_dims[i], out_dims[i + 1]))

    def encode(self, fea_mats: Dict, adj_mats: Dict) -> Dict:
        """Stack encoder layers over the heterogeneous graph.

        Args:
            fea_mats: Dict mapping node type -> feature tensor.
            adj_mats: Dict mapping edge type -> [adjacency tensor].

        Returns:
            Dict mapping node type -> embedding tensor after all encoder layers.
        """
        for m in self.encoder:
            fea_mats = m(fea_mats, adj_mats)
        return fea_mats

    def predict(self, z: Dict) -> Dict:
        """Apply per-task linear decoders to produce adjacency logits.

        Args:
            z: Dict mapping node type -> embedding tensor.

        Returns:
            Dict mapping edge type -> [logit tensor].
        """
        if self.model == 'GCN':
            z = DictReLU()(z)
            z = self.Dropout(z)
        return {t: [self.predicter[str(t)](z[t[0]])] for t in self.tasks}

    def forward(self, fea_mats: Dict, adj_mats: Dict,
                adj_masks) -> Tuple[Dict, Dict]:
        """Encode the graph and decode task-specific adjacency reconstructions.

        Args:
            fea_mats: Dict mapping node type -> feature tensor.
            adj_mats: Dict mapping edge type -> [adjacency tensor].
            adj_masks: Unused — reserved for future masking support.

        Returns:
            Tuple of (adj_recon, z) where adj_recon maps edge type -> [logit tensor]
            and z maps node type -> embedding tensor.
        """
        z = self.encode(fea_mats, adj_mats)
        adj_recon = self.predict(z)
        return adj_recon, z


class HeteroGCN(nn.Module):
    """Heterogeneous GCN layer with sum aggregation across edge types.

    For each node type i: h_i = GCN_i(x_i) + sum_{(i,j)} GCN_j(x_j, A_{ij})

    Args:
        in_dim: Dict mapping node type -> input dim, or int for homogeneous.
        out_dim: Output embedding dimensionality.
        aggregation_type: Message aggregation strategy (only 'sum' supported).
        act: Post-aggregation activation function.
    """

    def __init__(self, in_dim=[], out_dim: int = 64,
                 aggregation_type: str = 'sum', act: Callable = lambda x: x):
        super(HeteroGCN, self).__init__()
        if isinstance(in_dim, dict):
            self.gcn = nn.ModuleDict(
                {str(i): GraphConvolution(in_dim[i], out_dim, sparse=True)
                 for i in in_dim})
            self.ishg = True
        elif isinstance(in_dim, int):
            self.gcn = GraphConvolution(in_dim, out_dim)
            self.ishg = False
        else:
            raise ValueError('in_dim must be an int or dict of int')
        self.leakyrelu = nn.LeakyReLU(0.01)
        self.aggregation_type = aggregation_type

    def forward(self, fea_mats: Dict, adj_mats: Dict) -> Dict:
        """Propagate and aggregate node features across all edge types.

        Args:
            fea_mats: Dict mapping node type -> feature tensor.
            adj_mats: Dict mapping edge type -> [adjacency tensor].

        Returns:
            Dict mapping node type -> updated embedding tensor.
        """
        gcn = self.gcn if self.ishg else {str(i): self.gcn for i in fea_mats}
        out_fea = {i: gcn[str(i)](fea_mats[i]) for i in fea_mats}

        if self.aggregation_type == 'sum':
            for (i, j), adjs in adj_mats.items():
                adj = adjs[0]
                out_fea[i] = out_fea[i] + gcn[str(j)](fea_mats[j], adj)
                if i != j or not issymmetric(adj):
                    out_fea[j] = out_fea[j] + gcn[str(i)](fea_mats[i], adj.t())
        return out_fea


class HeteroGAT(nn.Module):
    """Heterogeneous Graph Attention layer with cosine similarity attention.

    For each node type i: h_i = sum_{(i,j)} CosAttn_j(h_i, h_j, A_{ij})
    followed by self-attention: h_i = CosAttn_i(h_i, h_i).

    Args:
        in_dim: Dict mapping node type -> input dim (int not supported).
        out_dim: Output embedding dimensionality.
        nlayers: Number of attention sub-layers.
        aggregation_type: Message aggregation strategy (only 'sum' supported).
        act: Post-aggregation activation function.
        dropout: Dropout probability (unused in forward, reserved).
    """

    def __init__(self, in_dim=[], out_dim: int = 64, nlayers: int = 1,
                 aggregation_type: str = 'sum', act: Callable = lambda x: x,
                 dropout: float = 0.1):
        super(HeteroGAT, self).__init__()
        self.layers = nlayers
        if isinstance(in_dim, dict):
            self.linear = nn.ModuleDict(
                {str(i): nn.Linear(in_dim[i], out_dim) for i in in_dim})
            self.ishg = True
            self.attentionlayers = nn.ModuleList()
            self.attentionlayers.append(nn.ModuleDict(
                {str(i): CosineGraphAttentionLayer(requires_grad=True)
                 for i in in_dim}))
            for _ in range(1, self.layers):
                self.attentionlayers.append(nn.ModuleDict(
                    {str(i): CosineGraphAttentionLayer() for i in in_dim}))
        elif isinstance(in_dim, int):
            self.ishg = False
            raise NotImplementedError('HeteroGAT requires dict in_dim.')
        else:
            raise ValueError('in_dim must be a dict of int')

        self.dropout = dropout
        self.activation = act
        self.out_fea = None
        self.aggregation_type = aggregation_type

    def forward(self, fea_mats: Dict, adj_mats: Dict) -> Dict:
        """Apply linear projection then cosine attention aggregation.

        Args:
            fea_mats: Dict mapping node type -> feature tensor.
            adj_mats: Dict mapping edge type -> [adjacency tensor].

        Returns:
            Dict mapping node type -> updated embedding tensor after activation.
        """
        support = self.linear if self.ishg else {str(i): self.linear for i in fea_mats}
        self.out_fea = {i: support[str(i)](fea_mats[i]) for i in fea_mats}

        if self.aggregation_type == 'sum':
            for layer_idx in range(self.layers):
                att = (self.attentionlayers[layer_idx] if self.ishg
                       else {str(i): self.attentionlayers[layer_idx]
                             for i in self.out_fea})
                self.out_fea = {
                    i: att[str(i)](self.out_fea[i], self.out_fea[i])
                    for i in self.out_fea}
                for (i, j), adjs in adj_mats.items():
                    adj = adjs[0]
                    self.out_fea[i] = (self.out_fea[i]
                                       + att[str(j)](self.out_fea[i],
                                                     self.out_fea[j], adj))
                    if i != j or not issymmetric(adj):
                        self.out_fea[j] = (self.out_fea[j]
                                           + att[str(i)](self.out_fea[j],
                                                         self.out_fea[i],
                                                         adj.t()))

        self.out_fea = {i: self.activation(self.out_fea[i]) for i in self.out_fea}
        return self.out_fea
