from .layers import *
from .utils import issymmetric, to_sparse, to_dense, normalize

class MGATRx(nn.Module):
    def __init__(self, in_dim, out_dims, tasks=((0, 3),), enc_act=lambda x:x, dropout=0.5, model='GCN'):
        super(MGATRx, self).__init__()
        self.model = model
        if model == 'GCN':
            self.heteroGNN = HeteroGCN
        elif model == 'GAT':
            self.heteroGNN = HeteroGAT
        self.encoder = nn.ModuleList([self.heteroGNN(in_dim, out_dims[0], act=enc_act)])
        self.Dropout = DictDropout(p=dropout)
        self.tasks = tasks
        self.predicter = nn.ModuleDict({str(i): nn.Linear(out_dims[-1], in_dim[i[1]]) for i in tasks})
        for i in range(len(out_dims)):
            if i + 1 < len(out_dims):
                self.encoder.append(self.heteroGNN(out_dims[i], out_dims[i + 1]))


    def encode(self, fea_mats, adj_mats):
        for i, m in enumerate(self.encoder):
            fea_mats = m(fea_mats, adj_mats)
            # if i + 1 < len(self.encoder):
            #     fea_mats = DictReLU()(fea_mats)
            #     fea_mats = self.Dropout(fea_mats)

        return fea_mats

    def predict(self, z):
        if self.model == 'GCN':
            z = DictReLU()(z)
            z = self.Dropout(z)
        adj_recon = {}
        for t in self.tasks:
            adj_recon[t] = [self.predicter[str(t)](z[t[0]])]

        return adj_recon

    def forward(self, fea_mats, adj_mats, adj_masks):
        # adj_mats = copy.deepcopy(adj_mats)
        # for key in adj_masks.keys() & adj_mats.keys():
        #     mask = torch.zeros(adj_mats[key][0].shape, device='cuda')
        #     mask[adj_masks[key][0]] = 1
        #     adj_mats[key] = [to_sparse(mask).float() * to_sparse(adj_mats[key][0])]
            # adj_mats[key] = [to_sparse(adj_mats[key][0])]

        z = self.encode(fea_mats, adj_mats)
        adj_recon = self.predict(z)
        return adj_recon, z

class HeteroGCN(nn.Module):
    def __init__(self, in_dim=[], out_dim=64, aggregation_type='sum', act = lambda x:x):
        super(HeteroGCN, self).__init__()
        if isinstance(in_dim, dict):
            self.gcn = nn.ModuleDict({str(i): GraphConvolution(in_dim[i],out_dim, sparse=True) for i in in_dim})
            self.ishg = True
        elif isinstance(in_dim, int):
            # This is incase of using more than one GCN layer
            self.gcn = GraphConvolution(in_dim, out_dim)
            self.ishg = False
        else:
            raise ValueError('in_dim must be integer or dict of integer')
        alpha = 0.01
        self.leakyrelu = torch.nn.LeakyReLU(alpha)
        self.aggregation_type = aggregation_type


    def forward(self, fea_mats, adj_mats):
        gcn = self.gcn if self.ishg else {str(i): self.gcn for i in fea_mats}
        out_fea = {i: gcn[str(i)](fea_mats[i]) for i in fea_mats}

        if self.aggregation_type == 'sum':
            for (i,j), adjs in adj_mats.items():
                adj = adjs[0] # There is only one adjacency matrix for type (i,j)
                out_fea[i] = out_fea[i] + gcn[str(j)](fea_mats[j], adj)
                if i!=j or not issymmetric(adj):
                    out_fea[j] = out_fea[j] + gcn[str(i)](fea_mats[i], adj.t())

        # out_fea = {i: self.leakyrelu(out_fea[i]) for i in out_fea}
        return out_fea

class HeteroGAT(nn.Module):
    def __init__(self, in_dim=[], out_dim=64, nlayers=1, aggregation_type='sum', act=lambda x:x, dropout=0.1):
        super(HeteroGAT, self).__init__()
        self.layers = nlayers
        if isinstance(in_dim, dict):
            self.linear = nn.ModuleDict({str(i): nn.Linear(in_dim[i], out_dim) for i in in_dim})
            self.ishg = True
            self.attentionlayers = nn.ModuleList()
            self.attentionlayers.append(nn.ModuleDict({str(i): CosineGraphAttentionLayer(requires_grad=True) for i in in_dim}))
            for i in range(1,self.layers):
                self.attentionlayers.append(nn.ModuleDict({str(i): CosineGraphAttentionLayer() for i in in_dim}))

        elif isinstance(in_dim, int):
            # This is incase of using more than one GCN layer
            # self.gcn = GraphAttentionLayer(in_dim, out_dim)
            self.ishg = False
            raise NotImplementedError
        else:
            raise ValueError('in_dim must be integer or dict of integer')

        # self.out_att = nn.ModuleDict({str(i): GraphAttentionLayer(out_dim, out_dim, alpha=alpha, concat=False) for i in in_dim})

        self.dropout=dropout
        # self.leakyrelu = torch.nn.LeakyReLU(alpha)
        # self.activation = nn.LeakyReLU(0.5)
        self.activation = act
        self.out_fea = None
        self.aggregation_type = aggregation_type




    def forward(self, fea_mats, adj_mats):
        support = self.linear if self.ishg else {str(i): self.linear for i in fea_mats}
        self.out_fea = {i: support[str(i)](fea_mats[i]) for i in fea_mats}

        if self.aggregation_type == 'sum':
            for i in range(self.layers):
                att = self.attentionlayers[i] if self.ishg else {str(i): self.attentionlayers[i] for i in self.out_fea}
                self.out_fea = {i: att[str(i)](self.out_fea[i], self.out_fea[i]) for i in self.out_fea}
                for (i, j), adjs in adj_mats.items():
                    adj = adjs[0]
                    self.out_fea[i] = self.out_fea[i] + att[str(j)](self.out_fea[i], self.out_fea[j], adj)

                    if i != j or not issymmetric(adj):
                        self.out_fea[j] = self.out_fea[j] + att[str(i)](self.out_fea[j], self.out_fea[i], adj.t())


        self.out_fea = {i: self.activation(self.out_fea[i]) for i in self.out_fea}




        return self.out_fea