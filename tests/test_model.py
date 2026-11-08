"""Basic sanity tests for CF-Voronoi."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pytest
from cf_voronoi import CF_Voronoi, MLP
from cf_voronoi.data import load_dataset


def test_model_creates():
    model = CF_Voronoi(in_dim=10, K=3, D=8, P=2)
    assert model.K == 3
    assert model.D == 8
    assert model.P == 2
    assert model.prototypes.shape == (6, 8)


def test_model_trains_on_small_data():
    """XOR-like tiny data — should reach perfect accuracy."""
    X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    y = torch.tensor([0, 1, 1, 0], dtype=torch.long)

    torch.manual_seed(0)
    model = CF_Voronoi(in_dim=2, K=2, D=8, P=2)
    model.fit(X, y, steps=500)
    preds = model.predict(X)
    assert (preds == y).all(), f"Failed XOR: {preds} vs {y}"


def test_predict_works():
    X, y = load_dataset('digits')
    X_t = torch.tensor(X[:100])
    y_t = torch.tensor(y[:100], dtype=torch.long)
    model = CF_Voronoi(in_dim=X.shape[1], K=10, D=16, P=2)
    model.fit(X_t, y_t, steps=100)
    preds = model.predict(X_t)
    assert preds.shape == y_t.shape
    assert set(preds.numpy()).issubset(set(y_t.numpy()))


def test_score_monotone():
    """Distance score should be non-negative."""
    X, y = load_dataset('digits')
    X_t = torch.tensor(X[:100])
    y_t = torch.tensor(y[:100], dtype=torch.long)
    model = CF_Voronoi(in_dim=X.shape[1], K=10, D=16, P=2)
    model.fit(X_t, y_t, steps=100)
    scores = model.score(X_t)
    assert (scores >= 0).all()


def test_extend_adds_classes():
    X, y = load_dataset('digits')
    X_t = torch.tensor(X[:200])
    y_t = torch.tensor(y[:200], dtype=torch.long)

    known = y_t < 5
    X_k, y_k = X_t[known], y_t[known]

    model = CF_Voronoi(in_dim=X.shape[1], K=5, D=16, P=2)
    model.fit(X_k, y_k, steps=100)
    original_K = model.K

    # Add one new class
    new_mask = y_t == 5
    X_new = X_t[new_mask][:5]
    y_new = torch.full((5,), 5, dtype=torch.long)
    model.extend(X_new, y_new)

    assert model.K == original_K + 1


def test_learnable_tau():
    """log_tau must actually change during training."""
    model = CF_Voronoi(in_dim=64, K=3, D=8, P=2)
    tau_init = model.log_tau.item()

    X, y = load_dataset('digits')
    X_t = torch.tensor(X[:50])
    y_t = torch.tensor(y[:50] % 3, dtype=torch.long)
    model.fit(X_t, y_t, steps=200)

    assert model.log_tau.item() != tau_init, "log_tau did not change"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
