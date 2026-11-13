import numpy as np
import pytest

from dslr.io import numeric_columns, to_float
from dslr.stats import count, maximum, mean, minimum, percentile, skew, sort, std, var

XS = [3.0, -1.5, 7.25, 0.0, 2.0, 2.0, 10.0]


@pytest.mark.parametrize("fn, ref", [
    (count, len),
    (mean, np.mean),
    (std, lambda xs: np.std(xs, ddof=1)),
    (var, lambda xs: np.var(xs, ddof=1)),
    (minimum, np.min),
    (maximum, np.max),
])
def test_stats_match_numpy(fn, ref):
    assert fn(XS) == pytest.approx(ref(XS))


@pytest.mark.parametrize("p", [0, 10, 25, 50, 75, 99, 100])
def test_percentile_matches_numpy(p):
    assert percentile(XS, p) == pytest.approx(np.percentile(XS, p))


def test_percentile_single_value():
    assert percentile([4.0], 25) == 4.0


def test_percentile_does_not_mutate_input():
    xs = [3.0, 1.0, 2.0]
    percentile(xs, 50)
    assert xs == [3.0, 1.0, 2.0]


def test_skew_sign():
    assert skew([1, 1, 1, 1, 10]) > 0
    assert skew([1, 10, 10, 10, 10]) < 0
    assert skew([1, 2, 3]) == pytest.approx(0)


@pytest.mark.parametrize("fn, xs", [(mean, []), (std, [1.0])])
def test_too_few_values_raise(fn, xs):
    with pytest.raises(ZeroDivisionError):
        fn(xs)


def test_to_float():
    assert to_float("1.5") == 1.5
    assert to_float("-2") == -2.0
    assert to_float("abc") is None


def test_numeric_columns_skips_text_index_and_keeps_missing_as_none():
    header = ["Index", "Name", "A", "B", "Mixed"]
    rows = [["0", "x", "1", "", "1"], ["1", "y", "2.5", "3", "oops"]]
    cols = numeric_columns(header, rows)
    assert set(cols) == {"A", "B"}
    assert cols["A"] == [1.0, 2.5]
    assert cols["B"] == [None, 3.0]


def test_numeric_columns_drops_all_empty_column():
    cols = numeric_columns(["Index", "E", "A"], [["0", "", "1"], ["1", "", "2"]])
    assert set(cols) == {"A"}


def test_numeric_columns_pandas_like_keeps_index_and_empty_columns():
    cols = numeric_columns(["Index", "E", "T", "A"], [["0", "", "x", "1"], ["1", "", "y", "2"]],
                           pandas_like=True)
    assert cols == {"Index": [0.0, 1.0], "E": [None, None], "A": [1.0, 2.0]}


def test_numeric_columns_handles_short_rows():
    cols = numeric_columns(["A", "B"], [["1"], ["2", "3"]])
    assert cols == {"A": [1.0, 2.0], "B": [None, 3.0]}


@pytest.mark.parametrize("xs", [[], [1.0], [2, 1], [3, 1, 2, 1, 5, -4, 0.5], list(range(50, 0, -1))])
def test_sort(xs):
    assert sort(xs) == sorted(xs)


def test_numeric_columns_treats_nan_and_inf_as_missing():
    cols = numeric_columns(["A"], [["1"], ["nan"], ["inf"], ["-Infinity"], ["2"]])
    assert cols == {"A": [1.0, None, None, None, 2.0]}


def test_numeric_columns_drops_column_of_only_nan():
    assert numeric_columns(["A", "B"], [["nan", "1"], ["NaN", "2"]]) == {"B": [1.0, 2.0]}
