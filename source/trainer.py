import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score

from .models import MGATRx
from .utils import normalize, to_dense
from .metrics import aupr_threshold


def get_activation(activation: str = 'none') -> nn.Module:
    activations = {
        'leaky': nn.LeakyReLU(0.1),
        'selu': nn.SELU(),
        'relu': nn.ReLU(),
        'tanh': nn.Tanh(),
        'sigmoid': nn.Sigmoid(),
        'elu': nn.ELU(),
        'prelu': nn.PReLU(),
    }
    if activation in activations:
        return activations[activation]
    return lambda x: x


def build_model(fea_nums, adj_mats, args, device):
    edge_decoder = {key: (args.decoder, 1) for key in adj_mats}
    num_layers = tuple(args.embed_size for _ in range(args.num_layers))
    if args.decoder != 'linear':
        raise NotImplementedError(f"Decoder '{args.decoder}' is not implemented.")
    enc_activation = get_activation(args.encoder_activation)
    model = MGATRx(
        fea_nums, num_layers, edge_decoder,
        enc_act=enc_activation, dropout=args.dropout, model=args.encoder
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return model, optimizer


def calculate_loss(adj_recon, adj_mats, mask, adj_losstype):
    total_loss = 0
    for key, item in adj_losstype.items():
        loss_type = item[0][0]
        loss_weight = item[0][1]

        inp = to_dense(adj_recon[key][0])
        target = to_dense(adj_mats[key][0])
        if len(mask[key]):
            inp = inp * mask[key][0]
            target = target * mask[key][0]

        if loss_type == 'BCE':
            inp = inp.view(-1)
            target = target.view(-1)
            pos_weight = (target == 0).sum() / (target != 0).sum()
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='mean')
            loss = criterion(inp, target)
        elif loss_type == 'MSE':
            inp = torch.sigmoid(inp).view(-1)
            target = target.view(-1)
            criterion = nn.MSELoss(reduction='mean')
            loss = criterion(inp, target)
        else:
            loss = 0

        total_loss += loss * loss_weight
    return total_loss


def train_fold(
    train_data, valid_data, test_data,
    adj_mats, fea_mats, adj_losstype,
    model, optimizer, model_init_state, opt_init_state,
    args, device, fold_num, timestamp
):
    model.load_state_dict(model_init_state)
    optimizer.load_state_dict(opt_init_state)

    shape = adj_mats[(0, 1)][0].shape
    drug_disease_train = np.zeros(shape)
    train_mask = np.zeros(shape)
    val_mask = np.zeros(shape)

    for row in train_data:
        drug_disease_train[row[0], row[1]] = row[2]
        train_mask[row[0], row[1]] = 1
    for row in valid_data:
        val_mask[row[0], row[1]] = 1

    drug_disease_train = torch.from_numpy(drug_disease_train).float()

    adj_mats_train = {}
    train_mask_dict = {}
    val_mask_dict = {}
    epoch_vs_aupr = []

    for key, item in adj_mats.items():
        if key == (0, 1):
            adj_mats_train[key] = [normalize(drug_disease_train, issymmetric=False).to(device)]
            train_mask_dict[key] = [torch.from_numpy(train_mask).float().to(device)]
            val_mask_dict[key] = [torch.from_numpy(val_mask).float().to(device)]
        else:
            adj_mats_train[key] = [normalize(item[0], issymmetric=False).to(device)]
            train_mask_dict[key] = []
            val_mask_dict[key] = []
        adj_mats[key] = [item[0].to(device)]

    best_valid_aupr = 0
    best_valid_auc = 0
    best_adj_recon = None
    test_aupr = 0
    test_auc = 0
    counter = 0

    pbar = tqdm(range(args.epochs))
    for epoch in pbar:
        model.train()
        optimizer.zero_grad()
        adj_recon, _ = model(fea_mats, adj_mats_train, None)
        loss_train = calculate_loss(adj_recon, adj_mats, train_mask_dict, adj_losstype)
        loss_val = calculate_loss(adj_recon, adj_mats, val_mask_dict, adj_losstype)
        loss_train.backward()
        optimizer.step()

        counter += 1
        if epoch % 25 == 0:
            print('\nEpoch:{}, Loss:{}'.format(epoch, loss_train))
            with torch.no_grad():
                results = adj_recon[(0, 1)][0].sigmoid().cpu().numpy()

            pred_list = [results[e[0], e[1]] for e in valid_data]
            ground_truth = [e[2] for e in valid_data]

            valid_auc = roc_auc_score(ground_truth, pred_list)
            valid_aupr = average_precision_score(ground_truth, pred_list)
            epoch_vs_aupr.append([epoch, valid_aupr])

            if valid_aupr >= best_valid_aupr:
                counter = 0
                best_valid_aupr = valid_aupr
                best_valid_auc = valid_auc
                best_adj_recon = {k: [adj_recon[k][0].clone().detach()] for k in adj_recon}

                test_pred = [results[e[0], e[1]] for e in test_data]
                test_gt = [e[2] for e in test_data]
                test_auc = roc_auc_score(test_gt, test_pred)
                test_aupr = average_precision_score(test_gt, test_pred)

            if counter > 0:
                print('No best AUPR for {} epochs'.format(counter))
            print('\nValid AUC: {:.4f} Valid AUPR: {:.4f} Test AUC: {:.4f} Test AUPR: {:.4f}'.format(
                valid_auc, valid_aupr, test_auc, test_aupr))

        pbar.set_description('Fold-{} loss_train:{:.4f} loss_val:{:.4f}'.format(fold_num, loss_train, loss_val))

        if counter > 50:
            break

        pd.DataFrame(epoch_vs_aupr, columns=['Epoch', 'AUPR']).to_csv(
            'logs/epoch_plots/MGATRx_EpochvsAUPR_fold{}_{}_{}.txt'.format(
                fold_num, args.encoder_activation, timestamp),
            sep='\t')

    return best_valid_auc, best_valid_aupr, test_auc, test_aupr, best_adj_recon
