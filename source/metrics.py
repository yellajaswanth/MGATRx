import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, hamming_loss, auc, roc_curve, precision_recall_curve, \
    precision_score, recall_score, f1_score, mean_squared_error, average_precision_score

def get_indexes(y):
	indexes = []
	for r in y:
		indexes.append(np.where(r == 1)[0].tolist())
	return indexes


import numpy as np


def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.

    This function computes the average precision at k between two lists of
    items.

    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The average precision at k over the input lists

    """
    if set(actual) == {0, 1} and len(actual) == len(predicted):
        actual = np.nonzero(actual)[0]
        predicted = np.argsort(predicted)[-k:][::-1]

    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    # if not actual:
    #     return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.

    This function computes the mean average precision at k between two lists
    of lists of items.

    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The mean average precision at k over the input lists

    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])



def optimal_cutoff(y_true, y_predicted):
    '''
    The following implementation is based on youden's index. It is the maximum
    vertical distance between ROC curve and diagonal line. The idea is to maximize
    the difference between True Positive and False Positive.
    J = Sensitivity - (1-Specificity)

    Other optimal cutoff points: https://www.listendata.com/2015/03/sas-calculating-optimal-predicted.html

    However, it is suggested that Youdens Index works well. http://www.medicalbiostatistics.com/roccurve.pdf

    :return: optimal point (float)
    '''
    fpr, tpr, threshold = roc_curve(y_true, y_predicted)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(threshold, index=i)})
    roc_t = roc.ix[(roc.tf - 0).abs().argsort()[:1]]

    return roc_t['threshold'].values[0]

def auc_prc_multilabel(y_true, y_pred):
    precision_micro, recall_micro, _ = precision_recall_curve(y_true.ravel(), y_pred.ravel())
    try:
        prc_auc_micro = auc(recall_micro, precision_micro)
    except ValueError:
        return 0.0
    return prc_auc_micro

def aupr_threshold(precision, recall, pr_thresholds):
    all_F_measure = np.zeros(len(pr_thresholds))
    for k in range(0, len(pr_thresholds)):
        if (precision[k] + precision[k]) > 0:
            all_F_measure[k] = 2 * precision[k] * recall[k] / (precision[k] + recall[k])
        else:
            all_F_measure[k] = 0
    max_index = all_F_measure.argmax()
    threshold = pr_thresholds[max_index]
    return threshold

def model_eval(predicted_labels, true_labels):
    true_labels = true_labels.ravel()
    predicted_labels = np.nan_to_num(predicted_labels.ravel())


    fpr, tpr, auc_thresholds = roc_curve(true_labels, predicted_labels)
    auc_score = auc(fpr, tpr)


    precision, recall, pr_thresholds = precision_recall_curve(true_labels, predicted_labels)
    aupr_score = auc(recall, precision)

    # Threshold is calculated based on AUPR
    threshold = aupr_threshold(precision, recall, pr_thresholds)
    # threshold = 0.5
    predicted_score = np.copy(predicted_labels)
    predicted_score[predicted_score > threshold] = 1
    predicted_score[predicted_score <= threshold] = 0


    f1_micro=f1_score(true_labels,predicted_score, 'micro')
    mse = mean_squared_error(true_labels, predicted_score)
    # f1_macro = f1_score(true_labels, predicted_score, 'macro')
    # f1_weigted = f1_score(true_labels, predicted_score, 'weighted')
    # f1_samples = f1_score(true_labels, predicted_score, 'samples')
    # accuracy=accuracy_score(true_labels,predicted_score)
    # precision=precision_score(true_labels,predicted_score, 'micro')
    # recall=recall_score(true_labels,predicted_score, 'micro')

    # indx_truth = get_indexes(true_labels)
    # indx_prediction = get_indexes(predicted_score)
    # map5 = mapk(true_labels, predicted_labels, k=5)
    # map10 = mapk(true_labels, predicted_labels, k=10)
    # map50 = mapk(true_labels, predicted_labels, k=50)
    # map50 = mapk(indx_truth, indx_prediction, k=50)
    # print(aupr_score,auc_score,f,accuracy,precision,recall)

    result_eval = {}
    result_eval['aupr'] = aupr_score
    result_eval['auc'] = auc_score
    result_eval['f1_micro'] = f1_micro
    result_eval['mse'] = mse
    # result_eval['f1_macro'] = f1_macro
    # result_eval['f1_weighted'] = f1_weigted
    # result_eval['f1_samples'] = f1_samples
    # result_eval['accuracy'] = accuracy
    # result_eval['precision'] = precision
    # result_eval['recall'] = recall
    # result_eval['map5'] = map5
    # result_eval['map10'] = map10
    # result_eval['map50'] = map50
    return result_eval

def calculate_auc(predicted_labels, true_labels):
    true_labels = true_labels.ravel()
    predicted_labels = np.nan_to_num(predicted_labels.ravel())

    fpr, tpr, auc_thresholds = roc_curve(true_labels, predicted_labels)
    auc_score = auc(fpr, tpr)
    return  auc_score

