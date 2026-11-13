import math

import numpy as np
import pytest

from conftest import TEST, TRAIN
from dslr.io import load_csv, numeric_columns
from dslr.stats import STATS, describe


def test_matches_numpy_on_train():
    header, rows = load_csv(TRAIN)
    cols = numeric_columns(header, rows)
    assert len(cols) == 13 and "Hogwarts House" not in cols
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
            assert table[name][stat] == pytest.approx(v), (name, stat)


def test_empty_column_gives_zero_count_and_nan():
    t = describe({"E": [None, None]})["E"]
    assert t["Count"] == 0 and t["Missing"] == 2
    assert all(math.isnan(t[s]) for s, _ in STATS if s != "Count")


def test_single_value_gives_nan_instead_of_crashing():
    t = describe({"A": [5.0, None]})["A"]
    assert t["Count"] == 1 and t["Mean"] == 5.0 and t["Missing"] == 1
    assert math.isnan(t["Std"])


def test_cli_output_layout(run):
    r = run("analysis/describe.py", TRAIN)
    assert r.returncode == 0, r.stderr
    lines = r.stdout.splitlines()
    # same columns as pandas.read_csv(TRAIN).describe(): Index plus the 13 courses
    assert lines[0].split()[:2] == ["Index", "Arithmancy"] and lines[0].endswith("Flying")
    assert [l.split()[0] for l in lines[1:]] == [s for s, _ in STATS] + ["Missing"]
    assert lines[1].split()[1:3] == ["1600.000000", "1566.000000"]


def test_cli_columns_never_run_together(run):
    lines = run("analysis/describe.py", TRAIN).stdout.splitlines()
    var = next(l for l in lines if l.startswith("Var"))
    assert len(var.split()) == 1 + 14  # Arithmancy's 9-digit variance stays separated


def test_cli_unlabelled_test_set_reports_empty_house_column_like_pandas(run):
    r = run("analysis/describe.py", TEST)
    assert r.returncode == 0
    header, count = r.stdout.splitlines()[:2]
    names = header.replace("Hogwarts House", "Hogwarts_House").split()
    assert names[:2] == ["Index", "Hogwarts_House"]
    assert count.split()[1:3] == ["400.000000", "0.000000"]


@pytest.mark.parametrize("args, msg", [
    ((), "usage"),
    (("missing.csv",), "error"),
])
def test_cli_errors(run, args, msg):
    r = run("analysis/describe.py", *args)
    assert r.returncode != 0 and msg in r.stderr


def test_cli_no_numeric_features(run, write_csv):
    r = run("analysis/describe.py", write_csv("Name,House\na,b\n"))
    assert r.returncode != 0 and "no numerical features" in r.stderr


def test_cli_undecodable_file(run, tmp_path):
    p = tmp_path / "bin.csv"
    p.write_bytes(b"\xff\xfe\x00binary")
    r = run("analysis/describe.py", p)
    assert r.returncode != 0 and "cannot read" in r.stderr and "Traceback" not in r.stderr


def test_cli_directory(run, tmp_path):
    r = run("analysis/describe.py", tmp_path)
    assert r.returncode != 0 and "cannot read" in r.stderr


def test_cli_empty_file(run, write_csv):
    r = run("analysis/describe.py", write_csv(""))
    assert r.returncode != 0 and "empty" in r.stderr
