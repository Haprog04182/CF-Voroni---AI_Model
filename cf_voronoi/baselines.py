"""Baseline models: MLP, ProtoNet."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Standard 2-layer MLP with ReLU."""
    def __init__(self, in_dim, K, D=32):
        super().__init__()
        hidden = max(16, D * 2)
        self.enc = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, D))
        self.head = nn.Linear(D, K)

    def forward(self, x):
        return self.head(self.enc(x))

    def features(self, x):
        return self.enc(x)


class ProtoNet(nn.Module):
    """Nearest-centroid classifier (supervised ProtoNet)."""
    def __init__(self, in_dim, K, D=32):
        super().__init__()
        hidden = max(16, D * 2)
        self.enc = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, D))
        self.centroids = nn.Parameter(torch.randn(K, D))
        self.K = K

    def _logits(self, z):
        return -torch.cdist(z, self.centroids).pow(2)

    def fit(self, X, y, steps=300, lr=1e-2, init_noise=0.1, reg_lambda=0.01):
        with torch.no_grad():
            Z = self.enc(X)
            for k in range(self.K):
                Zk = Z[y == k]
                if len(Zk):
                    c = Zk.mean(0)
                    self.centroids.data[k] = c + init_noise * torch.randn_like(c)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            z = self.enc(X)
            d2 = torch.cdist(z, self.centroids).pow(2)
            reg = d2[torch.arange(len(y)), y].mean()
            (F.cross_entropy(self._logits(z), y) + reg_lambda * reg).backward()
            opt.step()
        return self

    def predict(self, X):
        with torch.no_grad():
            return self._logits(self.enc(X)).argmax(1)

    def score(self, X):
        with torch.no_grad():
            return torch.cdist(self.enc(X), self.centroids).min(1)[0].numpy()
