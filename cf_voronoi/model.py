"""CF-Voronoi model implementation."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_encoder(in_dim, D):
    """Non-linear encoder: in_dim → D."""
    hidden = max(16, D * 2)
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.ReLU(),
        nn.Linear(hidden, D))


class CF_Voronoi(nn.Module):
    """Prototype-based classifier with softmin distances.

    Args:
        in_dim: input feature dimension
        K: number of classes
        D: latent space dimension
        P: prototypes per class

    Key fix notes:
        - log_tau is LEARNABLE (not fixed constant)
        - logits = +tau * logsumexp(-d2/tau)  (positive sign!)
        - prototype init uses random noise to break symmetry for P>1
    """

    def __init__(self, in_dim, K, D=32, P=3):
        super().__init__()
        self.K, self.D, self.P = K, D, P
        self.enc = make_encoder(in_dim, D)
        self.prototypes = nn.Parameter(torch.randn(K * P, D))
        self.log_tau = nn.Parameter(torch.tensor(0.0))

    def _logits(self, z):
        tau = self.log_tau.exp()
        d2 = torch.cdist(z, self.prototypes).pow(2).view(-1, self.K, self.P)
        return tau * torch.logsumexp(-d2 / tau, dim=2)

    def fit(self, X, y, steps=300, lr=1e-2, init_noise=0.1, reg_lambda=0.01):
        """Train with CE + regularizer pulling examples toward their prototypes."""
        with torch.no_grad():
            Z = self.enc(X)
            for k in range(self.K):
                Zk = Z[y == k]
                if len(Zk) == 0:
                    continue
                centroid = Zk.mean(dim=0)
                for p in range(self.P):
                    idx = k * self.P + p
                    self.prototypes.data[idx] = (
                        centroid + init_noise * torch.randn_like(centroid))
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            z = self.enc(X)
            logits = self._logits(z)
            d2 = torch.cdist(z, self.prototypes).pow(2).view(-1, self.K, self.P)
            reg = d2.min(dim=2)[0][torch.arange(len(y)), y].mean()
            (F.cross_entropy(logits, y) + reg_lambda * reg).backward()
            opt.step()
        return self

    def predict(self, X):
        with torch.no_grad():
            return self._logits(self.enc(X)).argmax(dim=1)

    def score(self, X):
        """Anomaly score: min distance to any prototype. Higher = more OOD."""
        with torch.no_grad():
            return torch.cdist(self.enc(X), self.prototypes).min(1)[0].numpy()

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

    def extend(self, X_new, y_new, n_steps=50, lr=5e-3):
        """Add new classes from few examples without full retraining.

        New prototypes initialized as centroids of new examples,
        then fine-tuned (encoder + old prototypes frozen).
        """
        old_K = self.K
        old_protos = self.prototypes.data.clone()

        with torch.no_grad():
            z_new = self.enc(X_new)
            new_protos = []
            for cls in range(old_K, int(y_new.max().item()) + 1):
                z_cls = z_new[y_new == cls]
                if len(z_cls) == 0:
                    continue
                centroid = z_cls.mean(dim=0)
                for _ in range(self.P):
                    new_protos.append(centroid + 0.01 * torch.randn_like(centroid))
            all_protos = torch.cat([old_protos, torch.stack(new_protos)], dim=0)

        self.prototypes = nn.Parameter(all_protos)
        self.K = old_K + len(new_protos) // self.P

        # Fine-tune only new prototypes
        new_start = old_K * self.P
        frozen = self.prototypes.data[:new_start].clone()
        new_param = nn.Parameter(self.prototypes.data[new_start:].clone())
        opt = torch.optim.Adam([new_param, self.log_tau], lr=lr)

        for _ in range(n_steps):
            opt.zero_grad()
            self.prototypes = nn.Parameter(torch.cat([frozen, new_param], dim=0))
            z = self.enc(X_new)
            F.cross_entropy(self._logits(z), y_new).backward()
            opt.step()

        self.prototypes = nn.Parameter(
            torch.cat([frozen, new_param.data], dim=0))
        return self
