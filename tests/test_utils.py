import numpy as np
import pytest

from utils import (count, maximum, mean, minimum, numeric_columns, percentile, skew, std,
                   to_float, var)

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


def test_numeric_columns_handles_short_rows():
    cols = numeric_columns(["A", "B"], [["1"], ["2", "3"]])
    assert cols == {"A": [1.0, 2.0], "B": [None, 3.0]}
