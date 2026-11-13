"""Self-check: uv run python test_dslr.py"""
import numpy as np

from describe import describe
from utils import load_csv, numeric_columns


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
        }
        for stat, v in expected.items():
            assert np.isclose(table[name][stat], v), (name, stat, table[name][stat], v)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
