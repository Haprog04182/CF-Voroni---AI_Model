"""Open-set recognition: detect unknown classes."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from cf_voronoi import CF_Voronoi, MLP, openset_metrics
from cf_voronoi.data import load_dataset, split_known_unknown
from sklearn.covariance import EmpiricalCovariance
import numpy as np


def energy_score(model, X, T=1.0):
    with torch.no_grad():
        return -(T * torch.logsumexp(model(X) / T, dim=1)).numpy()


def mahalanobis_score(model, X_tr, y_tr, X_te):
    with torch.no_grad():
        Z_tr = model.features(X_tr).numpy()
        Z_te = model.features(X_te).numpy()
    classes = np.unique(y_tr.numpy())
    centroids = np.stack([Z_tr[y_tr.numpy() == k].mean(0) for k in classes])
    cov = EmpiricalCovariance().fit(Z_tr - centroids[y_tr.numpy()])
    prec = cov.precision_
    scores = []
    for z in Z_te:
        d = np.array([(z - c) @ prec @ (z - c) for c in centroids])
        scores.append(d.min())
    return np.array(scores)


def fmt(m, tag):
    return (f"{tag:>20}: AUROC={m['AUROC']:.4f} | "
            f"FPR@95={m['FPR95']:.4f} | "
            f"AUPR-In={m['AUPRin']:.4f} | AUPR-Out={m['AUPRout']:.4f}")


def main():
    known = [0, 1, 2, 3, 4, 5]
    unknown = [6, 7, 8, 9]
    X, y = load_dataset('digits')
    splits = split_known_unknown(X, y, known, unknown)

    X_tr = torch.tensor(splits['X_train_known'])
    y_tr = torch.tensor(splits['y_train_known'], dtype=torch.long)
    X_te_k = torch.tensor(splits['X_test_known'])
    X_te_u = torch.tensor(splits['X_test_unknown'])

    print(f"Train known: {len(X_tr)}")
    print(f"Test known:  {len(X_te_k)}")
    print(f"Test unknown: {len(X_te_u)}\n")

    # MLP
    torch.manual_seed(0)
    mlp = MLP(64, 6, 32)
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-2)
    for _ in range(500):
        opt.zero_grad()
        F.cross_entropy(mlp(X_tr), y_tr).backward()
        opt.step()

    # Scores
    with torch.no_grad():
        s_k = -F.softmax(mlp(X_te_k), 1).max(1).values.numpy()
        s_u = -F.softmax(mlp(X_te_u), 1).max(1).values.numpy()
    print(fmt(openset_metrics(s_k, s_u), "MLP+softmax"))

    s_k = energy_score(mlp, X_te_k)
    s_u = energy_score(mlp, X_te_u)
    print(fmt(openset_metrics(s_k, s_u), "MLP+Energy"))

    s_k = mahalanobis_score(mlp, X_tr, y_tr, X_te_k)
    s_u = mahalanobis_score(mlp, X_tr, y_tr, X_te_u)
    print(fmt(openset_metrics(s_k, s_u), "MLP+Mahalanobis"))

    # CF-Voronoi
    torch.manual_seed(0)
    cf = CF_Voronoi(64, 6, 32, 3)
    cf.fit(X_tr, y_tr, steps=300)
    s_k = cf.score(X_te_k)
    s_u = cf.score(X_te_u)
    print(fmt(openset_metrics(s_k, s_u), "CF-Voronoi"))


if __name__ == "__main__":
    main()
