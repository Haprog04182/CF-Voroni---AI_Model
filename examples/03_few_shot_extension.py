"""Few-shot: add new classes without retraining."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from cf_voronoi import CF_Voronoi, MLP
from cf_voronoi.data import load_dataset
from sklearn.model_selection import train_test_split


def main():
    X, y = load_dataset('digits')
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y)

    known = [0, 1, 2, 3, 4, 5]
    unknown = [6, 7, 8, 9]

    m_tr = torch.tensor([c in known for c in y_tr])
    X_known = torch.tensor(X_tr[m_tr])
    y_known_raw = y_tr[m_tr]
    remap = {c: i for i, c in enumerate(known)}
    y_known = torch.tensor([remap[c] for c in y_known_raw], dtype=torch.long)

    # Train on 6 classes
    torch.manual_seed(0)
    cf = CF_Voronoi(64, 6, 32, 3)
    cf.fit(X_known, y_known, steps=300)
    print(f"Trained CF-Voronoi on classes 0-5 ({len(X_known)} samples)")

    # 5-shot new classes
    n_shot = 5
    new_X, new_y = [], []
    for i, cls in enumerate(unknown):
        mask = (y_te == cls)
        new_X.append(X_te[mask][:n_shot])
        new_y.extend([6 + i] * n_shot)

    X_new = torch.tensor(np.concatenate(new_X) if isinstance(new_X[0], np.ndarray)
                         else np.stack(new_X))
    y_new = torch.tensor(new_y, dtype=torch.long)

    # Extend CF-Voronoi
    cf_ext = CF_Voronoi(64, 6, 32, 3)
    cf_ext.load_state_dict(cf.state_dict())
    cf_ext.extend(X_new, y_new)

    X_full = torch.tensor(X_te)
    y_full = torch.tensor(y_te, dtype=torch.long)
    acc_cf = (cf_ext.predict(X_full) == y_full).float().mean().item()

    # MLP full retrain
    torch.manual_seed(0)
    mlp = MLP(64, 10, 32)
    X_full_tr = torch.tensor(X_tr)
    y_full_tr = torch.tensor(y_tr, dtype=torch.long)
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-2)
    for _ in range(500):
        opt.zero_grad()
        F.cross_entropy(mlp(X_full_tr), y_full_tr).backward()
        opt.step()
    acc_mlp = (mlp(X_full).argmax(1) == y_full).float().mean().item()

    print(f"\nResults:")
    print(f"  MLP full retrain ({len(X_full_tr)} samples): {acc_mlp:.4f}")
    print(f"  CF-Voronoi 5-shot ({len(X_new)} samples):    {acc_cf:.4f}")
    print(f"  → ratio: {acc_cf/acc_mlp:.1%}")


if __name__ == "__main__":
    import numpy as np
    main()
