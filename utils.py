import csv
import sys

HOUSES = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]
COLORS = {"Gryffindor": "red", "Hufflepuff": "gold", "Ravenclaw": "blue", "Slytherin": "green"}


def load_csv(path):
    """Return (header, rows) or exit with a readable error."""
    try:
        with open(path, newline="") as f:
            rows = list(csv.reader(f))
    except OSError as e:
        sys.exit(f"error: {e}")
    if not rows:
        sys.exit(f"error: {path} is empty")
    return rows[0], rows[1:]


def to_float(s):
    try:
        return float(s)
    except ValueError:
        return None


def numeric_columns(header, rows):
    """{name: [float|None]} for every column whose non-empty values all parse as numbers."""
    cols = {}
    for j, name in enumerate(header):
        if name == "Index":
            continue
        values = [to_float(r[j]) if j < len(r) and r[j] != "" else None for r in rows]
        non_empty = sum(1 for r in rows if j < len(r) and r[j] != "")
        parsed = [v for v in values if v is not None]
        if parsed and len(parsed) == non_empty:
            cols[name] = values
    return cols


def count(xs):
    n = 0
    for _ in xs:
        n += 1
    return n


def mean(xs):
    total = 0.0
    for x in xs:
        total += x
    return total / count(xs)


def var(xs):
    m = mean(xs)
    return sum((x - m) ** 2 for x in xs) / (count(xs) - 1)


def std(xs):
    return var(xs) ** 0.5


def minimum(xs):
    lo = xs[0]
    for x in xs:
        if x < lo:
            lo = x
    return lo


def maximum(xs):
    hi = xs[0]
    for x in xs:
        if x > hi:
            hi = x
    return hi


def percentile(xs, p):
    """Linear interpolation between closest ranks (same as numpy/pandas default)."""
    s = sorted(xs)
    k = (count(s) - 1) * p / 100
    lo = int(k)
    hi = lo + 1 if lo + 1 < count(s) else lo
    return s[lo] + (s[hi] - s[lo]) * (k - lo)
