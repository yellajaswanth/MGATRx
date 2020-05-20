import numpy as np
import scipy.sparse as sp
import torch
import pandas as pd
import networkx as nx

import os
import pickle


def normalize(adj, issymmetric=True):

    if torch.is_tensor(adj):
        adj = to_dense(adj).cpu().numpy()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    adj = sp.csc_matrix(adj)

    if issymmetric:
        """Using Tensorflow method - preprocess_adj_dense(adj)"""
        # adj = np.eye(adj.shape[0]) + adj # Uncomment this if you're not adding self connections in adj
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.linalg.pinv(np.diag(np.power(rowsum, 0.5).flatten()))
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = d_inv_sqrt
        adj_normalized = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
        return adj_normalized
    else:
        rowsum = np.array(adj.sum(1))
        degree_mat_inv_sqrt = sp.diags(np.nan_to_num(np.power(rowsum, -1)).flatten())
        adj_normalized = degree_mat_inv_sqrt.dot(adj)
        return sparse_to_tensor(adj_normalized).to(device)


def sparse_to_tensor(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = torch.LongTensor(np.vstack((sparse_mx.row, sparse_mx.col)))
    values = torch.FloatTensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(coords, values, shape)


def load_drugbank_data():
    """
    :return: adjs, features, labels
    """

    if os.path.exists('tmp/drugbank-train.pkl'):
        print('Using pickle data...')
        f = open('tmp/drugbank-train.pkl', 'rb')
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
    num_drugs = adjs[0].shape[0]

    print('Using as-is adjacency matrix')
    num_drugs, num_diseases = labels.shape
    _, num_targets = targets.shape
    _, num_substructs = chem.shape
    _, num_ses = se.shape
    _, num_meshcats = meshcat.shape
    # _, num_pathways = pathways.shape

    adj_mats = {
            (0,1): [torch.FloatTensor(labels)],
            (0,2): [torch.FloatTensor(targets)],
            (0,3): [torch.FloatTensor(chem)],
            (0,4): [torch.FloatTensor(se)],
            (0,5): [torch.FloatTensor(meshcat)],
            (1,2): [torch.FloatTensor(dis_targets)],
         }
    drug_feat = torch.eye(num_drugs)
    dis_feat = torch.eye(num_diseases)
    target_feat = torch.eye(num_targets)
    chem_feat = torch.eye(num_substructs)
    se_feat = torch.eye(num_ses)
    mesh_feat = torch.eye(num_meshcats)

    fea_mats = {
        0: drug_feat,
        1: dis_feat,
        2: target_feat,
        3: chem_feat,
        4: se_feat,
        5: mesh_feat,
        # 6: pathway_feat
    }

    fea_nums = {
        0: num_drugs,
        1: num_diseases,
        2: num_targets,
        3: num_substructs,
        4: num_ses,
        5: num_meshcats,
        # 6: num_pathways
    }

    adj_losstype = {
        (0, 1): [('BCE', 1)],
        (0, 2): [('MSE', 1)],
        (0, 3): [('MSE', 1)],
        (0, 4): [('MSE', 1)],
        (0, 5): [('MSE', 1)],
        (1, 2): [('MSE', 1)],
    }

    if not os.path.exists('tmp/'): os.mkdir('tmp/')


    with open('tmp/drugbank-train.pkl', 'wb') as f:
        pickle.dump([adj_mats, fea_mats, fea_nums, adj_losstype], f)
    return adj_mats, fea_mats, fea_nums, adj_losstype




def to_sparse(x):
    return x if x.is_sparse else x.to_sparse()

def to_dense(x):
    return x if not x.is_sparse else x.to_dense()

def issymmetric(mat):
    if torch.is_tensor(mat):
        mat = mat.to_dense().cpu().numpy() if mat.is_sparse else mat.cpu().numpy()
    if mat.shape!=mat.T.shape:
        return False
    if sp.issparse(mat):
        return not (mat!=mat.T).todense().any()
    else:
        return not (mat!=mat.T).any()