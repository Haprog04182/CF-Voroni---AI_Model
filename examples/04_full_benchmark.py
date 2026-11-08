"""Full benchmark: closed-set + open-set + 5-seed stability + datasets."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from cf_voronoi import CF_Voronoi, MLP
from cf_voronoi.data import load_dataset
from cf_voronoi.metrics import evaluate_5seeds


def run_dataset(name, normalize, configs):
    X, y = load_dataset(name)
    in_dim = X.shape[1]
    K = len(set(y))

    print(f"\n{'='*70}")
    print(f"Dataset: {name} (in_dim={in_dim}, K={K}, N={len(X)})")
    print(f"{'='*70}")
    print(f"{'method':>25} | {'params':>7} | {'acc (5 seeds)':>16}")
    print("-" * 60)

    def mlp_fn(X_tr, y_tr, X_te, y_te, seed):
        torch.manual_seed(seed)
        m = MLP(in_dim, K, D=32)
        opt = torch.optim.Adam(m.parameters(), lr=1e-2)
        for _ in range(500):
            opt.zero_grad()
            F.cross_entropy(m(X_tr), y_tr).backward()
            opt.step()
        with torch.no_grad():
            return (m(X_te).argmax(1) == y_te).float().mean().item()

    m, s = evaluate_5seeds(mlp_fn, X, y, normalize=normalize)
    print(f"{'MLP':>25} | {sum(p.numel() for p in MLP(in_dim, K).parameters()):>7} | "
          f"{m:.4f} ± {s:.4f}")

    for D, P in configs:
        def cf_fn(X_tr, y_tr, X_te, y_te, seed):
            torch.manual_seed(seed)
            model = CF_Voronoi(in_dim, K, D=D, P=P)
            model.fit(X_tr, y_tr, steps=200)
            return (model.predict(X_te) == y_te).float().mean().item()
        m, s = evaluate_5seeds(cf_fn, X, y, normalize=normalize)
        p = CF_Voronoi(in_dim, K, D=D, P=P).count_params()
        print(f"{f'CF-Voronoi D={D}, P={P}':>25} | {p:>7} | {m:.4f} ± {s:.4f}")


def main():
    run_dataset('digits', False, [(8, 3), (16, 5), (32, 3)])
    run_dataset('wine', True, [(8, 3), (16, 5)])
    run_dataset('breast_cancer', True, [(8, 3), (16, 5)])


if __name__ == "__main__":
    main()
