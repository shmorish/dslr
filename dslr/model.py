import sys

import numpy as np

from dslr.io import HOUSES
from dslr.stats import mean, std

# Chosen from pair_plot: drop Arithmancy / Care of Magical Creatures (no house signal)
# and Defense Against the Dark Arts (exact duplicate of Astronomy).
FEATURES = [
    "Astronomy", "Herbology", "Divination", "Muggle Studies", "Ancient Runes",
    "History of Magic", "Transfiguration", "Potions", "Charms", "Flying",
]


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def prepare(cols, features, means, stds):
    """Matrix with a bias column; missing values filled with the training mean, then standardized."""
    X = np.array([[means[f] if v is None else v for v in cols[f]] for f in features]).T
    X = (X - [means[f] for f in features]) / [stds[f] for f in features]
    return np.hstack([np.ones((len(X), 1)), X])


# name: (learning rate, epochs, batch size; None = whole dataset)
ALGOS = {"batch": (0.5, 1000, None), "minibatch": (0.1, 100, 32), "sgd": (0.01, 20, 1)}


def log_loss(X, y, theta):
    """J(θ) = −(1/m) Σ [y log h + (1−y) log(1−h)], clipped so log never sees 0."""
    h = np.clip(sigmoid(X @ theta), 1e-15, 1 - 1e-15)
    return float(-(y * np.log(h) + (1 - y) * np.log(1 - h)).sum() / len(y))


def gradient_descent(X, y, algo="batch"):
    """Minimize J(θ) with dJ/dθ = Xᵀ(g(Xθ) − y) / m, over all rows or shuffled batches.

    Returns (θ, [(epoch, J)]) with J recorded at the start, every 1/5 of training and the end.
    """
    lr, epochs, size = ALGOS[algo]
    rng = np.random.default_rng(0)
    theta = np.zeros(X.shape[1])
    history = [(0, log_loss(X, y, theta))]
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(y)) if size else np.arange(len(y))
        for start in range(0, len(y), size or len(y)):
            b = order[start:start + (size or len(y))]
            theta -= lr * X[b].T @ (sigmoid(X[b] @ theta) - y[b]) / len(b)
        if epoch % (epochs // 5) == 0:
            history.append((epoch, log_loss(X, y, theta)))
    return theta, history


def train(cols, houses, algo="batch"):
    for f in FEATURES:
        if f not in cols:
            sys.exit(f"error: missing feature column '{f}'")
    means = {f: mean([v for v in cols[f] if v is not None]) for f in FEATURES}
    # a constant column would divide by zero; leaving it unscaled makes it a harmless constant
    stds = {f: std([v for v in cols[f] if v is not None]) or 1.0 for f in FEATURES}
    X = prepare(cols, FEATURES, means, stds)
    labels = np.array(houses)
    thetas, losses = {}, {}
    for h in HOUSES:
        theta, losses[h] = gradient_descent(X, (labels == h).astype(float), algo)
        thetas[h] = theta.tolist()
    return {"features": FEATURES, "means": means, "stds": stds, "thetas": thetas, "loss": losses}


def check_model(model):
    """Error message if model is not what train() produces, else None."""
    try:
        features = model["features"]
        ok = (all(isinstance(model[k][f], (int, float)) for k in ("means", "stds") for f in features)
              and all(len(model["thetas"][h]) == len(features) + 1 for h in HOUSES))
    except (KeyError, TypeError, IndexError):
        ok = False
    return None if ok else "weights file does not contain a trained model"


def predict(model, cols):
    X = prepare(cols, model["features"], model["means"], model["stds"])
    scores = np.array([X @ np.array(model["thetas"][h]) for h in HOUSES])
    return [HOUSES[i] for i in scores.argmax(axis=0)]
