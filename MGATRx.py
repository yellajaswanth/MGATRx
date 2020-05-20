import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

import warnings
from tqdm import tqdm

import torch
import torch.optim as optim

from source.metrics import *
from source.utils import *
from source.argparser import parse_args
from source.models import *
import copy


warnings.filterwarnings("ignore")

timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())

if not os.path.exists('logs/epoch_plots'): os.makedirs('logs/epoch_plots')


args = parse_args()
print(args)



##############################
######## Load Dataset ########
##############################

adj_mats, fea_mats, fea_nums, adj_losstype = load_drugbank_data()
# adj_mats, fea_mats, fea_nums, adj_losstype = load_orpha_data()
# adj_losstype = {
#             (0, 1): [('BCE', 1)],
#             (0, 2): [('MSE', 0)],
#             (0, 3): [('MSE', 0)],
#             (0, 4): [('MSE', 0)],
#             (0, 5): [('MSE', 0)],
#             (1, 2): [('MSE', 0)]
#         }
tasks = [task for task, loss in adj_losstype.items() if loss[0][1] > 0]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.empty_cache()  # entirely clear cache in GPU
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0) / 1024 ** 3, 1), 'GB')
    # for key, item in adj_mats.items():
    #     adj_mats[key] = [adj_mats[key][0].to(device)]

    for key, item in fea_mats.items():
        fea_mats[key] = fea_mats[key].to(device)


##############################
######## Model Config ########
##############################

def get_activation(activation='none'):
    # choices = ['leaky','selu', 'relu','tanh','sigmoid','elu','none']
    if activation == 'leaky':
        return torch.nn.LeakyReLU(0.1)
    elif activation == 'selu':
        return torch.nn.SELU()
    elif activation =='relu':
        return torch.nn.ReLU()
    elif activation == 'tanh':
        return torch.nn.tanh()
    elif activation == 'sigmoid':
        return torch.nn.Sigmoid()
    elif activation == 'elu':
        return torch.nn.ELU()
    elif activation == 'prelu':
        return torch.nn.PReLU()
    elif activation == 'none':
        return lambda x:x

def load_model():
    edge_decoder = {}
    encoder = args.encoder
    decoder = args.decoder
    for i, item in adj_mats.items(): edge_decoder[i] = (decoder,1)
    # for task in tasks: edge_decoder[task] = ('dismult',1)
    # count_dict = {}
    # for x, y in adj_mats.keys():
    #     count_dict[x] = count_dict.get(x, 0) + 1
    #     count_dict[y] = count_dict.get(y, 0) + 1
    num_layers = tuple([args.embed_size for layer in range(args.num_layers)])
    if decoder == 'linear':
        enc_activation = get_activation(args.encoder_activation)
        model = MGATRx(fea_nums, num_layers, edge_decoder, enc_act=enc_activation, dropout=args.dropout, model=encoder).to(device)
    else:
        raise NotImplementedError
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    return model, optimizer

model, optimizer = load_model()
model_init_state = copy.deepcopy(model.state_dict())
opt_init_state = copy.deepcopy(optimizer.state_dict())

def calculate_loss(adj_recon, adj_mats, mask, adj_losstype):
    total_loss = 0
    for key, item in adj_losstype.items():
        loss = 0
        loss_type = item[0][0]
        loss_weight = item[0][1]

        input = to_dense(adj_recon[key][0])
        target = to_dense(adj_mats[key][0])
        if len(mask[key]):
            input = input * mask[key][0]
            target = target * mask[key][0]


        if loss_type == 'BCE':
            input = input.view(-1)
            target = target.view(-1)
            pos_weight = (target == 0).sum() / (target != 0).sum()

            loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight,reduction='mean')
            loss = loss(input, target)
        elif loss_type == 'MSE':
            input = torch.sigmoid(input)
            input = input.view(-1)
            target = target.view(-1)

            loss = torch.nn.MSELoss(reduction='mean')
            loss = loss(input, target)
            # loss = 0

        total_loss += loss*loss_weight
    return total_loss

##############################
######## Train Model  ########
##############################



def gcn_train(TRAIN_DATA,VALID_DATA, TEST_DATA, adj_mats, NUM_EPOCHS, FOLD_NUM):
    print(model)

    model.load_state_dict(model_init_state)
    optimizer.load_state_dict(opt_init_state)

    ORIGINAL_DRUG_DIS_SHAPE = adj_mats[(0,1)][0].shape

    drug_disease_train = np.zeros(ORIGINAL_DRUG_DIS_SHAPE)
    drug_disease_val = np.zeros(ORIGINAL_DRUG_DIS_SHAPE)

    train_mask = np.zeros((ORIGINAL_DRUG_DIS_SHAPE))
    val_mask = np.zeros((ORIGINAL_DRUG_DIS_SHAPE))

    for row in TRAIN_DATA:
        drug_disease_train[row[0], row[1]] = row[2]
        train_mask[row[0], row[1]] = 1

    for row in VALID_DATA:
        drug_disease_val[row[0], row[1]] = row[2]
        val_mask[row[0], row[1]] = 1


    drug_disease_train = torch.from_numpy(drug_disease_train).float()

    adj_mats_train = {}

    train_mask_dict = {}
    val_mask_dict = {}

    epoch_vs_aupr = []

    for key,item in adj_mats.items():
        if key == (0,1):
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
    pbar = tqdm(range(NUM_EPOCHS))
    for epoch in pbar:
        model.train()
        optimizer.zero_grad()
        adj_recon, adj_embedding = model(fea_mats, adj_mats_train, None)
        loss_train = calculate_loss(adj_recon, adj_mats, train_mask_dict, adj_losstype)
        loss_val = calculate_loss(adj_recon, adj_mats, val_mask_dict, adj_losstype)

        loss_train.backward()
        optimizer.step()

        # earlystopper(val_loss=loss_val, model=model, criteria='loss')
        # if earlystopper.early_stop: break


        counter +=1
        if epoch % 25 == 0:
            print('\nEpoch:{}, Loss:{}'.format(epoch, loss_train))
            pred_list = []
            ground_truth = []
            with torch.no_grad():
                results = adj_recon[(0,1)][0].sigmoid().cpu().numpy()

            for ele in VALID_DATA:
                pred_list.append(results[ele[0], ele[1]])
                ground_truth.append(ele[2])

            valid_auc = roc_auc_score(ground_truth, pred_list)
            valid_aupr = average_precision_score(ground_truth, pred_list)


            epoch_vs_aupr.append([epoch,valid_aupr])

            if valid_aupr >= best_valid_aupr:
                counter = 0
                best_valid_aupr = valid_aupr
                best_valid_auc = valid_auc
                best_adj_recon = {}
                for key in adj_recon:
                    best_adj_recon[key] = [adj_recon[key][0].clone().detach()]
                # best_epoch = epoch
                pred_list = []
                ground_truth = []
                for ele in TEST_DATA:
                    pred_list.append(results[ele[0], ele[1]])
                    ground_truth.append(ele[2])

                test_auc = roc_auc_score(ground_truth, pred_list)
                test_aupr = average_precision_score(ground_truth, pred_list)
            if counter > 0: print('No best AUPR for '+str(counter) + ' epochs')

            print('\nValid AUC: {:.4f} Valid AUPR: {:.4f} Test AUC: {:.4f} Test AUPR: {:.4f}'.format(valid_auc, valid_aupr, test_auc, test_aupr))
        pbar.set_description('Fold-{} loss_train:{:.4f} loss_val:{:.4f}'.format(FOLD_NUM, loss_train, loss_val))
        if counter > 50: break
        pd.DataFrame(epoch_vs_aupr, columns=['Epoch','AUPR']).to_csv('logs/epoch_plots/GCNRx_EpochvsAUPR_fold{}_{}_{}.txt'.format(FOLD_NUM,args.encoder_activation, timestamp), sep='\t')
    return best_valid_auc, best_valid_aupr, test_auc, test_aupr, best_adj_recon


##############################
######## K-Fold setup  #######
##############################

test_auc_round = []
test_aupr_round = []



TOTAL_DRUGS = adj_mats[(0,1)][0].cpu().numpy().shape[0]
TOTAL_DISEASES = adj_mats[(0, 1)][0].cpu().numpy().shape[1]


dataset_indices = None

drug_disease_mat = np.zeros((TOTAL_DRUGS, TOTAL_DISEASES))

drug_disease_mat = copy.deepcopy(adj_mats[(0,1)][0].cpu().numpy())



whole_positive_index = []
whole_negative_index = []

for i in range(np.shape(drug_disease_mat)[0]):
    for j in range(np.shape(drug_disease_mat)[1]):
        if int(drug_disease_mat[i][j]) == 1:
            whole_positive_index.append([i,j])
        elif int(drug_disease_mat[i][j]) == 0:
            whole_negative_index.append([i, j])


negative_sample_index = np.random.choice(np.arange(len(whole_negative_index)), size=len(whole_negative_index),
                                         replace=False)

data_set = np.zeros((len(negative_sample_index) + len(whole_positive_index), 3), dtype=int)
count = 0
for i in whole_positive_index:
    data_set[count][0] = i[0]
    data_set[count][1] = i[1]
    data_set[count][2] = 1
    count += 1
for i in negative_sample_index:
    data_set[count][0] = whole_negative_index[i][0]
    data_set[count][1] = whole_negative_index[i][1]
    data_set[count][2] = 0
    count += 1


test_auc_fold = []
test_aupr_fold = []

test_results_fold = []
test_set_fold = []
kf = StratifiedKFold(n_splits=args.kfolds, shuffle=True, random_state=args.seed)

fold_count = 1
for train_index, test_index in kf.split(data_set[:,[0,1]],data_set[:,2]):

    train_data, test_data = data_set[train_index], data_set[test_index]
    train_data, validation_data = train_test_split(train_data, test_size=0.15, stratify=train_data[:,2])
    print('Train:{} Val:{} Test:{}'.format((train_data.size / data_set.size), (validation_data.size / data_set.size), (test_data.size / data_set.size)))

    v_auc, v_aupr, t_auc, t_aupr,results = gcn_train(TRAIN_DATA=train_data, VALID_DATA=validation_data, TEST_DATA=test_data,
                                                     adj_mats=adj_mats,  NUM_EPOCHS=args.epochs, FOLD_NUM= fold_count)
    print('Fold-{} Results'.format(fold_count))
    print('Valid AUC: {:.4f} Valid AUPR: {:.4f} Test AUC: {:.4f} Test AUPR: {:.4f}'.format(v_auc, v_aupr, t_auc,
                                                                                           t_aupr))
    test_auc_fold.append(t_auc)
    test_aupr_fold.append(t_aupr)

    if args.save_model:
        save_file = 'tmp/GCNRx_' + args.encoder + '_' + args.decoder + '_' + str(timestamp) + '_'+ str(fold_count)+ '.pkl'
        with open(save_file, 'wb') as f:
            pickle.dump([results, test_data], f)

    test_results_fold.append(results)
    test_set_fold.append(test_data)
    fold_count +=1



    if args.fold_test: break


if args.save_model:
    save_file = 'tmp/GCNRx_' + args.encoder+'_'+ args.decoder+ '_'+str(timestamp)+ '.pkl'

    with open(save_file, 'wb') as f:
        pickle.dump([test_results_fold, test_set_fold], f)

for adj in test_results_fold:
    for key, item in adj.items():
        adj[key] = [item[0].detach().cpu()]




y_real = []
y_proba = []
for i in range(args.kfolds):
    y_true = []
    y_score = []
    predictions = test_results_fold[i][(0, 1)][0].sigmoid().numpy()
    for row in test_set_fold[i]:
        y_true.append(row[2])
        y_score.append(predictions[row[0], row[1]])
    y_real.append(y_true)
    y_proba.append(y_score)
    if args.fold_test: break
y_real = np.concatenate(y_real)
y_proba = np.concatenate(y_proba)
precision, recall, pr_thresholds = precision_recall_curve(y_real, y_proba, pos_label=1)
ap = average_precision_score(y_real, y_proba)
threshold = aupr_threshold(precision, recall, pr_thresholds)


fpr, tpr, _ = roc_curve(y_real, y_proba)
aucroc = auc(fpr, tpr)
print('Average Precision:{}, AUC:{}'.format(ap, aucroc))

predicted_score = np.copy(y_proba)
predicted_score[predicted_score > threshold] = 1
predicted_score[predicted_score <= threshold] = 0

f1_micro=f1_score(y_real,predicted_score, 'micro')


rows = []
rows.append(['AUPR',ap])
rows.append(['AUC',aucroc])
rows.append(['F1',f1_micro])
for arg in vars(args):
    rows.append([arg, getattr(args, arg)])

pd.DataFrame(rows, columns=['Attribute','Value']).to_csv('logs/GCNRx_{}.log'.format(timestamp), sep='\t', index=False)




