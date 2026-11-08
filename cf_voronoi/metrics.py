"""Open-set metrics and cross-validation helpers."""
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split


def openset_metrics(known_scores, unknown_scores):
    """Compute OOD detection metrics.

    Args:
        known_scores: anomaly scores for in-distribution samples
        unknown_scores: anomaly scores for OOD samples
        Convention: higher score = more OOD

    Returns:
        dict with AUROC, FPR@95, AUPR-In, AUPR-Out
    """
    y_true = np.r_[np.zeros(len(known_scores)), np.ones(len(unknown_scores))]
    y_score = np.r_[known_scores, unknown_scores]

    auroc = roc_auc_score(y_true, y_score)
    thr = np.percentile(known_scores, 95)
    fpr95 = (unknown_scores <= thr).mean()
    aupr_in = average_precision_score(1 - y_true, -y_score)
    aupr_out = average_precision_score(y_true, y_score)

    return dict(AUROC=auroc, FPR95=fpr95, AUPRin=aupr_in, AUPRout=aupr_out)


def evaluate_5seeds(method_fn, X_np, y_np, normalize=False,
                    n_seeds=5, test_size=0.2, **kwargs):
    """Run method_fn over multiple seeds, return mean ± std."""
    X_np = X_np.astype('float32')
    if normalize:
        X_np = (X_np - X_np.mean(0)) / (X_np.std(0) + 1e-8)

    accs = []
    for seed in range(n_seeds):
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_np, y_np, test_size=test_size,
            random_state=seed, stratify=y_np)
        accs.append(method_fn(
            torch.tensor(X_tr),
            torch.tensor(y_tr, dtype=torch.long),
            torch.tensor(X_te),
            torch.tensor(y_te, dtype=torch.long),
            seed, **kwargs))
    return np.mean(accs), np.std(accs)
