import numpy as np
import pandas as pd

from sklearn.metrics import average_precision_score, roc_curve, auc, f1_score, precision_recall_curve

from .metrics import aupr_threshold


def aggregate_fold_predictions(test_results_fold: list, test_set_fold: list,
                               num_folds: int, fold_test: bool):
    """Collect and concatenate predictions across all cross-validation folds.

    Args:
        test_results_fold: List of per-fold best adj_recon dicts (CPU tensors).
        test_set_fold: List of per-fold test arrays of shape (N, 3).
        num_folds: Total number of folds to aggregate.
        fold_test: If True, only the first fold is aggregated.

    Returns:
        Tuple (y_real, y_proba) of concatenated 1-D numpy arrays.
    """
    y_real = []
    y_proba = []
    for i in range(num_folds):
        predictions = test_results_fold[i][(0, 1)][0].sigmoid().numpy()
        y_true = [row[2] for row in test_set_fold[i]]
        y_score = [predictions[row[0], row[1]] for row in test_set_fold[i]]
        y_real.append(y_true)
        y_proba.append(y_score)
        if fold_test:
            break
    return np.concatenate(y_real), np.concatenate(y_proba)


def compute_and_log_metrics(y_real: np.ndarray, y_proba: np.ndarray,
                            args, timestamp: str) -> None:
    """Compute AUPR, AUC-ROC, and F1 then write a TSV log file.

    The decision threshold is selected as the point maximising F1 on the
    precision-recall curve (via aupr_threshold).

    Args:
        y_real: Ground-truth binary labels.
        y_proba: Predicted probability scores.
        args: Parsed argparse namespace (all fields written to the log).
        timestamp: Run timestamp string used in the output filename.
    """
    precision, recall, pr_thresholds = precision_recall_curve(y_real, y_proba, pos_label=1)
    ap = average_precision_score(y_real, y_proba)
    threshold = aupr_threshold(precision, recall, pr_thresholds)

    fpr, tpr, _ = roc_curve(y_real, y_proba)
    aucroc = auc(fpr, tpr)
    print('Average Precision:{}, AUC:{}'.format(ap, aucroc))

    predicted_score = np.copy(y_proba)
    predicted_score[predicted_score > threshold] = 1
    predicted_score[predicted_score <= threshold] = 0
    f1_micro = f1_score(y_real, predicted_score, average='micro')

    rows = [['AUPR', ap], ['AUC', aucroc], ['F1', f1_micro]]
    for arg in vars(args):
        rows.append([arg, getattr(args, arg)])

    pd.DataFrame(rows, columns=['Attribute', 'Value']).to_csv(
        'logs/MGATRx_{}.log'.format(timestamp), sep='\t', index=False)
