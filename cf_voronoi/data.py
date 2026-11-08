"""Dataset loading utilities."""
import numpy as np
from sklearn.datasets import load_digits, load_wine, load_breast_cancer


def load_dataset(name):
    """Load a standard dataset with consistent preprocessing."""
    if name == 'digits':
        X, y = load_digits(return_X_y=True)
        return X.astype('float32') / 16.0, y.astype('int64')
    if name == 'wine':
        X, y = load_wine(return_X_y=True)
        return X.astype('float32'), y.astype('int64')
    if name == 'breast_cancer':
        X, y = load_breast_cancer(return_X_y=True)
        return X.astype('float32'), y.astype('int64')
    raise ValueError(f"Unknown dataset: {name}")


def split_known_unknown(X, y, known_classes, unknown_classes,
                        test_size=0.2, seed=0):
    """Split into train (known only) + test_known + test_unknown."""
    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y)

    tr_mask = np.isin(y_tr, known_classes)
    te_k_mask = np.isin(y_te, known_classes)
    te_u_mask = np.isin(y_te, unknown_classes)

    X_tr_k = X_tr[tr_mask]
    y_tr_k = y_tr[tr_mask]
    remap = {c: i for i, c in enumerate(known_classes)}
    y_tr_k_remap = np.array([remap[c] for c in y_tr_k])

    return {
        'X_train_known': X_tr_k,
        'y_train_known': y_tr_k_remap,
        'X_test_known': X_te[te_k_mask],
        'y_test_known_orig': y_te[te_k_mask],
        'X_test_unknown': X_te[te_u_mask],
        'y_test_unknown_orig': y_te[te_u_mask],
        'remap': remap,
    }
