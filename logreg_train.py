import json
import sys

import numpy as np

from utils import HOUSES, load_houses, mean, std

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


def gradient_descent(X, y, lr=0.5, epochs=1000):
    """Minimize the log loss J(θ) with dJ/dθ = Xᵀ(g(Xθ) − y) / m."""
    theta = np.zeros(X.shape[1])
    for _ in range(epochs):
        theta -= lr * X.T @ (sigmoid(X @ theta) - y) / len(y)
    return theta


def train(cols, houses):
    for f in FEATURES:
        if f not in cols:
            sys.exit(f"error: missing feature column '{f}'")
    means = {f: mean([v for v in cols[f] if v is not None]) for f in FEATURES}
    stds = {f: std([v for v in cols[f] if v is not None]) for f in FEATURES}
    X = prepare(cols, means, stds)
    labels = np.array(houses)
    thetas = {h: gradient_descent(X, (labels == h).astype(float)).tolist() for h in HOUSES}
    return {"features": FEATURES, "means": means, "stds": stds, "thetas": thetas}


def predict(model, cols):
    X = prepare(cols, model["means"], model["stds"])
    scores = np.array([X @ np.array(model["thetas"][h]) for h in HOUSES])
    return [HOUSES[i] for i in scores.argmax(axis=0)]


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: logreg_train.py dataset_train.csv")
    cols, houses = load_houses(sys.argv[1])
    keep = [i for i, h in enumerate(houses) if h in HOUSES]
    cols = {f: [v[i] for i in keep] for f, v in cols.items()}
    houses = [houses[i] for i in keep]
    model = train(cols, houses)
    with open("weights.json", "w") as f:
        json.dump(model, f, indent=2)
    acc = sum(p == h for p, h in zip(predict(model, cols), houses)) / len(houses)
    print(f"weights saved to weights.json (training accuracy {acc:.4f})")


if __name__ == "__main__":
    main()
