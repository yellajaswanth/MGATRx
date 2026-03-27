from typing import Union

import numpy as np
import scipy.sparse as sp
import torch
import pandas as pd
import os
import pickle


def normalize(adj: Union[torch.Tensor, np.ndarray, sp.spmatrix],
              issymmetric: bool = True) -> Union[np.ndarray, torch.Tensor]:
    """Row-normalize an adjacency matrix.

    Symmetric normalization: D^{-1/2} A D^{-1/2} (returns dense numpy array).
    Asymmetric normalization: D^{-1} A (returns sparse CPU tensor).

    Args:
        adj: Adjacency matrix as a dense/sparse tensor or numpy array.
        issymmetric: If True applies symmetric normalization, else row normalization.

    Returns:
        Normalized adjacency. Symmetric -> numpy ndarray; asymmetric -> sparse FloatTensor.
    """
    if torch.is_tensor(adj):
        adj = to_dense(adj).cpu().numpy()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    adj = sp.csc_matrix(adj)

    if issymmetric:
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.linalg.pinv(np.diag(np.power(rowsum, 0.5).flatten()))
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        adj_normalized = adj.dot(d_inv_sqrt).transpose().dot(d_inv_sqrt)
        return adj_normalized
    else:
        rowsum = np.array(adj.sum(1))
        degree_mat_inv_sqrt = sp.diags(np.nan_to_num(np.power(rowsum, -1)).flatten())
        adj_normalized = degree_mat_inv_sqrt.dot(adj)
        return sparse_to_tensor(adj_normalized).to(device)


def sparse_to_tensor(sparse_mx: sp.spmatrix) -> torch.Tensor:
    """Convert a scipy sparse matrix to a PyTorch sparse FloatTensor.

    Args:
        sparse_mx: scipy sparse matrix of any format.

    Returns:
        torch.sparse.FloatTensor with the same values and shape.
    """
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = torch.LongTensor(np.vstack((sparse_mx.row, sparse_mx.col)))
    values = torch.FloatTensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(coords, values, shape)


def load_drugbank_data():
    """Load the DrugBank/KEGG heterogeneous graph dataset.

    Reads six bipartite adjacency matrices from data/DB-KEGG/ and constructs
    identity-matrix node features. Results are cached in tmp/drugbank-train.pkl.

    Returns:
        adj_mats: Dict mapping (node_type_i, node_type_j) -> [FloatTensor].
        fea_mats: Dict mapping node_type -> identity FloatTensor.
        fea_nums: Dict mapping node_type -> int (number of nodes).
        adj_losstype: Dict mapping edge type -> [(loss_fn str, weight int)].
    """
    if os.path.exists('tmp/drugbank-train.pkl'):
        print('Using pickle data...')
        with open('tmp/drugbank-train.pkl', 'rb') as f:
            adj_mats, fea_mats, fea_nums, adj_losstype = pickle.load(f)
        return adj_mats, fea_mats, fea_nums, adj_losstype

    chem = np.loadtxt('data/DB-KEGG/drug-chemfp.txt')
    targets = np.loadtxt('data/DB-KEGG/drug-targets.txt')
    se = np.loadtxt('data/DB-KEGG/drug-se.txt')
    meshcat = np.loadtxt('data/DB-KEGG/drug-meshcat.txt')
    dis_targets = np.loadtxt('data/DB-KEGG/disease-targets.txt')
    labels = np.loadtxt('data/DB-KEGG/drug-disease.txt')

    adjs = [chem, targets, se, meshcat]
    assert len(adjs) > 0
    assert labels.shape[0] > 0

    print('Using as-is adjacency matrix')
    num_drugs, num_diseases = labels.shape
    _, num_targets = targets.shape
    _, num_substructs = chem.shape
    _, num_ses = se.shape
    _, num_meshcats = meshcat.shape

    adj_mats = {
        (0, 1): [torch.FloatTensor(labels)],
        (0, 2): [torch.FloatTensor(targets)],
        (0, 3): [torch.FloatTensor(chem)],
        (0, 4): [torch.FloatTensor(se)],
        (0, 5): [torch.FloatTensor(meshcat)],
        (1, 2): [torch.FloatTensor(dis_targets)],
    }

    fea_mats = {
        0: torch.eye(num_drugs),
        1: torch.eye(num_diseases),
        2: torch.eye(num_targets),
        3: torch.eye(num_substructs),
        4: torch.eye(num_ses),
        5: torch.eye(num_meshcats),
    }

    fea_nums = {
        0: num_drugs,
        1: num_diseases,
        2: num_targets,
        3: num_substructs,
        4: num_ses,
        5: num_meshcats,
    }

    adj_losstype = {
        (0, 1): [('BCE', 1)],
        (0, 2): [('MSE', 1)],
        (0, 3): [('MSE', 1)],
        (0, 4): [('MSE', 1)],
        (0, 5): [('MSE', 1)],
        (1, 2): [('MSE', 1)],
    }

    os.makedirs('tmp', exist_ok=True)
    with open('tmp/drugbank-train.pkl', 'wb') as f:
        pickle.dump([adj_mats, fea_mats, fea_nums, adj_losstype], f)

    return adj_mats, fea_mats, fea_nums, adj_losstype


def to_sparse(x: torch.Tensor) -> torch.Tensor:
    """Convert a dense tensor to sparse format (no-op if already sparse)."""
    return x if x.is_sparse else x.to_sparse()


def to_dense(x: torch.Tensor) -> torch.Tensor:
    """Convert a sparse tensor to dense format (no-op if already dense)."""
    return x if not x.is_sparse else x.to_dense()


def issymmetric(mat: Union[torch.Tensor, np.ndarray, sp.spmatrix]) -> bool:
    """Check whether a matrix is symmetric.

    Args:
        mat: Square matrix as a tensor, numpy array, or scipy sparse matrix.

    Returns:
        True if mat == mat.T, False otherwise.
    """
    if torch.is_tensor(mat):
        mat = mat.to_dense().cpu().numpy() if mat.is_sparse else mat.cpu().numpy()
    if mat.shape != mat.T.shape:
        return False
    if sp.issparse(mat):
        return not (mat != mat.T).todense().any()
    return not (mat != mat.T).any()
