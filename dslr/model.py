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


def prepare(cols, means, stds):
    """Matrix with a bias column; missing values filled with the training mean, then standardized."""
    X = np.array([[means[f] if v is None else v for v in cols[f]] for f in FEATURES]).T
    X = (X - [means[f] for f in FEATURES]) / [stds[f] for f in FEATURES]
    return np.hstack([np.ones((len(X), 1)), X])


# name: (learning rate, epochs, batch size; None = whole dataset)
ALGOS = {"batch": (0.5, 1000, None), "minibatch": (0.1, 100, 32), "sgd": (0.01, 20, 1)}


def gradient_descent(X, y, algo="batch"):
    """Minimize the log loss J(θ) with dJ/dθ = Xᵀ(g(Xθ) − y) / m, over all rows or shuffled batches."""
    lr, epochs, size = ALGOS[algo]
    rng = np.random.default_rng(0)
    theta = np.zeros(X.shape[1])
    for _ in range(epochs):
        order = rng.permutation(len(y)) if size else np.arange(len(y))
        for start in range(0, len(y), size or len(y)):
            b = order[start:start + (size or len(y))]
            theta -= lr * X[b].T @ (sigmoid(X[b] @ theta) - y[b]) / len(b)
    return theta


def train(cols, houses, algo="batch"):
    for f in FEATURES:
        if f not in cols:
            sys.exit(f"error: missing feature column '{f}'")
    means = {f: mean([v for v in cols[f] if v is not None]) for f in FEATURES}
    stds = {f: std([v for v in cols[f] if v is not None]) for f in FEATURES}
    X = prepare(cols, means, stds)
    labels = np.array(houses)
    thetas = {h: gradient_descent(X, (labels == h).astype(float), algo).tolist() for h in HOUSES}
    return {"features": FEATURES, "means": means, "stds": stds, "thetas": thetas}


def predict(model, cols):
    X = prepare(cols, model["means"], model["stds"])
    scores = np.array([X @ np.array(model["thetas"][h]) for h in HOUSES])
    return [HOUSES[i] for i in scores.argmax(axis=0)]
