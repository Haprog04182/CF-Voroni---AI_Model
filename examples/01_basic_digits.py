"""Quick start: train CF-Voronoi on digits, compare with MLP."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from cf_voronoi import CF_Voronoi, MLP
from cf_voronoi.data import load_dataset
from sklearn.model_selection import train_test_split


def main():
    print("CF-Voronoi on digits\n")

    X, y = load_dataset('digits')
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y)
    X_tr = torch.tensor(X_tr)
    y_tr = torch.tensor(y_tr, dtype=torch.long)
    X_te = torch.tensor(X_te)
    y_te = torch.tensor(y_te, dtype=torch.long)

    print(f"Train: {len(X_tr)}, Test: {len(X_te)}, "
          f"In-dim: {X_tr.shape[1]}, Classes: {len(set(y.tolist()))}\n")

    print(f"{'model':>30} | {'params':>7} | {'test acc':>9}")
    print("-" * 55)

    # MLP baseline
    torch.manual_seed(0)
    mlp = MLP(in_dim=64, K=10, D=32)
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-2)
    for _ in range(500):
        opt.zero_grad()
        F.cross_entropy(mlp(X_tr), y_tr).backward()
        opt.step()
    acc_mlp = (mlp(X_te).argmax(1) == y_te).float().mean().item()
    p_mlp = sum(p.numel() for p in mlp.parameters())
    print(f"{'MLP':>30} | {p_mlp:>7} | {acc_mlp:>9.4f}")

    # CF-Voronoi sweep
    for D, P in [(8, 3), (16, 5), (32, 3)]:
        torch.manual_seed(0)
        cf = CF_Voronoi(in_dim=64, K=10, D=D, P=P)
        cf.fit(X_tr, y_tr, steps=300)
        acc = (cf.predict(X_te) == y_te).float().mean().item()
        p = cf.count_params()
        print(f"{f'CF-Voronoi D={D}, P={P}':>30} | {p:>7} | {acc:>9.4f} "
              f"({p/p_mlp:.2f}x)")


if __name__ == "__main__":
    main()
