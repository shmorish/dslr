"""Self-check: uv run python test_dslr.py"""
import numpy as np

from describe import describe
from logreg_train import ALGOS, predict, train
from utils import load_csv, load_houses, numeric_columns


def test_describe_matches_numpy():
    header, rows = load_csv("datasets/dataset_train.csv")
    cols = numeric_columns(header, rows)
    assert "Hogwarts House" not in cols and "Arithmancy" in cols
    table = describe(cols)
    for name, values in cols.items():
        xs = np.array([v for v in values if v is not None])
        expected = {
            "Count": len(xs), "Mean": xs.mean(), "Std": xs.std(ddof=1), "Min": xs.min(),
            "25%": np.percentile(xs, 25), "50%": np.percentile(xs, 50),
            "75%": np.percentile(xs, 75), "Max": xs.max(),
            "Var": xs.var(ddof=1), "Range": np.ptp(xs),
            "IQR": np.percentile(xs, 75) - np.percentile(xs, 25),
            "Skew": np.mean(((xs - xs.mean()) / xs.std(ddof=1)) ** 3),
            "Missing": len(values) - len(xs),
        }
        for stat, v in expected.items():
            assert np.isclose(table[name][stat], v), (name, stat, table[name][stat], v)


def holdout_accuracy(algo):
    cols, houses = load_houses("datasets/dataset_train.csv")
    idx = np.random.default_rng(0).permutation(len(houses))
    cut = len(idx) * 4 // 5
    split = lambda ids: ({f: [v[i] for i in ids] for f, v in cols.items()}, [houses[i] for i in ids])
    (tr_cols, tr_h), (te_cols, te_h) = split(idx[:cut]), split(idx[cut:])
    pred = predict(train(tr_cols, tr_h, algo), te_cols)
    return np.mean([p == h for p, h in zip(pred, te_h)])


def test_logreg_holdout_accuracy():
    for algo in ALGOS:
        acc = holdout_accuracy(algo)
        assert acc >= 0.98, (algo, acc)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
