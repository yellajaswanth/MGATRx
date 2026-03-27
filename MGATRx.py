import time
import os
import copy
import pickle
import warnings

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

import torch

from source.argparser import parse_args
from source.utils import load_drugbank_data
from source.trainer import build_model, train_fold
from source.evaluate import aggregate_fold_predictions, compute_and_log_metrics

warnings.filterwarnings("ignore")

timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())

if not os.path.exists('logs/epoch_plots'):
    os.makedirs('logs/epoch_plots')

args = parse_args()
print(args)

adj_mats, fea_mats, fea_nums, adj_losstype = load_drugbank_data()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')

fea_mats = {k: v.to(device) for k, v in fea_mats.items()}

model, optimizer = build_model(fea_nums, adj_mats, args, device)
model_init_state = copy.deepcopy(model.state_dict())
opt_init_state = copy.deepcopy(optimizer.state_dict())

drug_disease_mat = copy.deepcopy(adj_mats[(0, 1)][0].cpu().numpy())

whole_positive_index = []
whole_negative_index = []
for i in range(drug_disease_mat.shape[0]):
    for j in range(drug_disease_mat.shape[1]):
        if int(drug_disease_mat[i][j]) == 1:
            whole_positive_index.append([i, j])
        else:
            whole_negative_index.append([i, j])

negative_sample_index = np.random.choice(
    np.arange(len(whole_negative_index)), size=len(whole_negative_index), replace=False)

data_set = np.zeros((len(negative_sample_index) + len(whole_positive_index), 3), dtype=int)
count = 0
for i in whole_positive_index:
    data_set[count] = [i[0], i[1], 1]
    count += 1
for i in negative_sample_index:
    data_set[count] = [whole_negative_index[i][0], whole_negative_index[i][1], 0]
    count += 1

test_auc_fold = []
test_aupr_fold = []
test_results_fold = []
test_set_fold = []

kf = StratifiedKFold(n_splits=args.kfolds, shuffle=True, random_state=args.seed)

for fold_count, (train_index, test_index) in enumerate(
        kf.split(data_set[:, [0, 1]], data_set[:, 2]), start=1):

    train_data, test_data = data_set[train_index], data_set[test_index]
    train_data, validation_data = train_test_split(
        train_data, test_size=args.valid_size, stratify=train_data[:, 2])

    print('Train:{:.2%} Val:{:.2%} Test:{:.2%}'.format(
        train_data.size / data_set.size,
        validation_data.size / data_set.size,
        test_data.size / data_set.size))

    v_auc, v_aupr, t_auc, t_aupr, results = train_fold(
        train_data=train_data,
        valid_data=validation_data,
        test_data=test_data,
        adj_mats=copy.deepcopy(adj_mats),
        fea_mats=fea_mats,
        adj_losstype=adj_losstype,
        model=model,
        optimizer=optimizer,
        model_init_state=model_init_state,
        opt_init_state=opt_init_state,
        args=args,
        device=device,
        fold_num=fold_count,
        timestamp=timestamp,
    )

    print('Fold-{} Results'.format(fold_count))
    print('Valid AUC: {:.4f} Valid AUPR: {:.4f} Test AUC: {:.4f} Test AUPR: {:.4f}'.format(
        v_auc, v_aupr, t_auc, t_aupr))

    test_auc_fold.append(t_auc)
    test_aupr_fold.append(t_aupr)

    if args.save_model:
        os.makedirs('tmp', exist_ok=True)
        save_file = 'tmp/MGATRx_{}_{}_{}_fold{}.pkl'.format(
            args.encoder, args.decoder, timestamp, fold_count)
        with open(save_file, 'wb') as f:
            pickle.dump([results, test_data], f)

    test_results_fold.append(results)
    test_set_fold.append(test_data)

    if args.fold_test:
        break

if args.save_model:
    os.makedirs('tmp', exist_ok=True)
    save_file = 'tmp/MGATRx_{}_{}_{}_{}.pkl'.format(
        args.encoder, args.decoder, timestamp, 'all')
    with open(save_file, 'wb') as f:
        pickle.dump([test_results_fold, test_set_fold], f)

for adj in test_results_fold:
    for key, item in adj.items():
        adj[key] = [item[0].detach().cpu()]

num_folds = 1 if args.fold_test else args.kfolds
y_real, y_proba = aggregate_fold_predictions(
    test_results_fold, test_set_fold, num_folds, args.fold_test)

compute_and_log_metrics(y_real, y_proba, args, timestamp)
